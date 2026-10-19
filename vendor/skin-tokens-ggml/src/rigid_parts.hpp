#pragma once

// Rigid parts. Many models keep hard parts as separate closed solids: the
// bones of an anatomical skeleton, a robot's plates, a belt buckle, a hair
// clip. Learned skin weights spread such a part over two to eight joints, so
// it bends "like bubble gum" (or twists like a candy wrapper) when the joints
// turn. A part that must not bend has to follow one joint at 100%.
//
// Each part is judged on its own, so the same rule serves skeletons, robots
// and dressed characters. A part becomes rigid when
//   1. it is closed (no border edge): hair cards, cloth panels and most
//      skins are open sheets;
//   2. it sits on one joint: at least half of its vertices are nearest to a
//      bone segment that joint moves (a segment belongs to its parent joint;
//      a point past the child end belongs to the child joint). A body,
//      a whole limb, a glove or a sleeve spreads over several joints;
//   3. its learned weights reach only the strongest joint and the joints one
//      bone away (a cord tied to a long hair chain stays soft);
// and the snap may not take more than half of any joint's total weight away
// from it (an eye or jaw joint that only exists for that part); parts that
// would do so stay soft. Size and orientation never enter the rule.
//
// Toon models add an inverted-hull outline around each part: a copy with the
// same vertex and face counts, pushed out along the normals. The outline must
// move with the part inside it, or it slides off: each part is grouped with
// the outline that encloses it before the rule runs, and the outline of a
// part that stays soft copies the weights of the nearest vertex inside.
//
// Everything works on the welded mesh of the weight transfer (seam twins
// share one vertex), so a test reaches it without the model.

#include <skintokens/skintokens.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <unordered_map>
#include <vector>

namespace skintokens::detail::rigid {

inline constexpr std::uint32_t no_part = std::numeric_limits<std::uint32_t>::max();

struct parts {
    // Part of each welded vertex; `no_part` for a vertex on no face.
    std::vector<std::uint32_t> of;
    std::vector<std::vector<std::uint32_t>> members;
    std::vector<std::size_t> faces;
    // A closed part has every edge on two or more faces (no border).
    std::vector<char> closed;
};

// Connected parts of a welded triangle mesh and whether each one is closed.
inline parts find_parts(std::size_t count, std::span<const std::array<std::uint32_t, 3>> faces) {
    std::vector<std::uint32_t> parent(count);
    std::iota(parent.begin(), parent.end(), 0U);
    const auto root = [&](std::uint32_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    std::vector<char> on_face(count, 0);
    std::unordered_map<std::uint64_t, std::uint32_t> edge_uses;
    edge_uses.reserve(faces.size() * 3U);
    for (const auto &face : faces) {
        if (face[0] >= count || face[1] >= count || face[2] >= count) continue;
        for (std::size_t k = 0; k < 3U; ++k) {
            const auto a = face[k], b = face[(k + 1U) % 3U];
            on_face[a] = 1;
            parent[root(a)] = root(b);
            const auto low = std::min(a, b), high = std::max(a, b);
            ++edge_uses[(static_cast<std::uint64_t>(low) << 32U) | high];
        }
    }
    parts out;
    out.of.assign(count, no_part);
    std::vector<std::uint32_t> id_of_root(count, no_part);
    for (std::uint32_t v = 0; v < count; ++v) {
        if (!on_face[v]) continue;
        const auto r = root(v);
        if (id_of_root[r] == no_part) {
            id_of_root[r] = static_cast<std::uint32_t>(out.members.size());
            out.members.emplace_back();
        }
        out.of[v] = id_of_root[r];
        out.members[id_of_root[r]].push_back(v);
    }
    out.faces.assign(out.members.size(), 0U);
    out.closed.assign(out.members.size(), 1);
    for (const auto &face : faces) {
        if (face[0] >= count || face[1] >= count || face[2] >= count) continue;
        ++out.faces[out.of[face[0]]];
    }
    for (const auto &[key, uses] : edge_uses)
        if (uses < 2U) out.closed[out.of[static_cast<std::uint32_t>(key >> 32U)]] = 0;
    return out;
}

// Group each part with the outline shell around it: a part with the same
// vertex and face counts whose box encloses this part's box (within 1% of the
// shell's diagonal), whose diagonal is larger, and whose centre lies within
// 10% of its diagonal. Returns a group id per part.
inline std::vector<std::uint32_t> pair_shells(const parts &found, std::span<const vec3> position) {
    const std::size_t count = found.members.size();
    struct box {
        std::array<double, 3> low, high, centre;
        double diagonal;
    };
    std::vector<box> boxes(count);
    for (std::size_t part = 0; part < count; ++part) {
        box b{};
        b.low = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};
        b.high = {-b.low[0], -b.low[1], -b.low[2]};
        for (const auto v : found.members[part]) {
            const std::array<double, 3> p{position[v].x, position[v].y, position[v].z};
            for (std::size_t k = 0; k < 3U; ++k) {
                b.low[k] = std::min(b.low[k], p[k]);
                b.high[k] = std::max(b.high[k], p[k]);
                b.centre[k] += p[k];
            }
        }
        const double size = static_cast<double>(std::max<std::size_t>(found.members[part].size(), 1U));
        double d2 = 0.0;
        for (std::size_t k = 0; k < 3U; ++k) {
            b.centre[k] /= size;
            d2 += (b.high[k] - b.low[k]) * (b.high[k] - b.low[k]);
        }
        b.diagonal = std::sqrt(d2);
        boxes[part] = b;
    }
    // Candidates share vertex and face counts, so bucket by them.
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> same_shape;
    for (std::uint32_t part = 0; part < count; ++part)
        same_shape[(static_cast<std::uint64_t>(found.members[part].size()) << 32U) | found.faces[part]].push_back(part);
    std::vector<std::uint32_t> group(count);
    std::iota(group.begin(), group.end(), 0U);
    const auto root = [&](std::uint32_t x) {
        while (group[x] != x) {
            group[x] = group[group[x]];
            x = group[x];
        }
        return x;
    };
    for (const auto &[shape, list] : same_shape) {
        if (list.size() < 2U) continue;
        for (const auto inner : list) {
            const auto &in = boxes[inner];
            std::uint32_t best = no_part;
            double best_distance = std::numeric_limits<double>::infinity();
            for (const auto outer : list) {
                if (outer == inner) continue;
                const auto &out = boxes[outer];
                if (!(out.diagonal > in.diagonal)) continue;
                const double tolerance = 0.01 * out.diagonal;
                bool encloses = true;
                double d2 = 0.0;
                for (std::size_t k = 0; k < 3U; ++k) {
                    encloses = encloses && out.low[k] <= in.low[k] + tolerance && out.high[k] >= in.high[k] - tolerance;
                    d2 += (out.centre[k] - in.centre[k]) * (out.centre[k] - in.centre[k]);
                }
                const double distance = std::sqrt(d2);
                if (!encloses || distance >= 0.1 * out.diagonal || distance >= best_distance) continue;
                best = outer;
                best_distance = distance;
            }
            if (best != no_part) group[root(inner)] = root(best);
        }
    }
    for (std::uint32_t part = 0; part < count; ++part) group[part] = root(part);
    return group;
}

// Joint that moves each point at rest: the nearest bone segment's parent
// joint, or its child joint when the point lies past the segment's child
// end (a head beyond the neck, a fingertip). `joints` are rest positions in
// the same space as the points; `parents[j] < 0` marks a root.
inline std::vector<std::uint32_t> owners(std::span<const vec3> points, std::span<const vec3> joints,
    std::span<const std::int32_t> parents) {
    struct segment {
        std::uint32_t parent, child;
        vec3 a, ab;
        float length2;
    };
    std::vector<segment> segments;
    for (std::size_t joint = 0; joint < joints.size() && joint < parents.size(); ++joint) {
        const auto parent = parents[joint];
        if (parent < 0 || static_cast<std::size_t>(parent) >= joints.size()) continue;
        const vec3 a = joints[static_cast<std::size_t>(parent)], b = joints[joint];
        const vec3 ab{b.x - a.x, b.y - a.y, b.z - a.z};
        segments.push_back({static_cast<std::uint32_t>(parent), static_cast<std::uint32_t>(joint), a, ab,
            ab.x * ab.x + ab.y * ab.y + ab.z * ab.z});
    }
    std::vector<std::uint32_t> out(points.size(), 0U);
    if (segments.empty()) return out;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto p = points[i];
        float best = std::numeric_limits<float>::infinity();
        for (const auto &s : segments) {
            const vec3 ap{p.x - s.a.x, p.y - s.a.y, p.z - s.a.z};
            float t = s.length2 > 0.0F ? (ap.x * s.ab.x + ap.y * s.ab.y + ap.z * s.ab.z) / s.length2 : 0.0F;
            t = std::clamp(t, 0.0F, 1.0F);
            const vec3 d{ap.x - t * s.ab.x, ap.y - t * s.ab.y, ap.z - t * s.ab.z};
            const float d2 = d.x * d.x + d.y * d.y + d.z * d.z;
            if (d2 < best) {
                best = d2;
                out[i] = t >= 1.0F ? s.child : s.parent;
            }
        }
    }
    return out;
}

// A row as the exported mesh keeps it: the four greatest weights, summed to
// one. The guard counts weight this way so it matches the exported skin.
inline void kept_four(const float *row, std::size_t joint_count, std::vector<double> &out) {
    out.assign(joint_count, 0.0);
    std::array<std::size_t, 4> best{};
    std::size_t found = 0U;
    for (std::size_t joint = 0; joint < joint_count; ++joint) {
        if (!(row[joint] > 0.0F)) continue;
        std::size_t at = found < 4U ? found++ : 4U;
        if (at == 4U) {
            if (row[joint] <= row[best[3]]) continue;
            at = 3U;
        }
        best[at] = joint;
        while (at > 0U && row[best[at]] > row[best[at - 1U]]) {
            std::swap(best[at], best[at - 1U]);
            --at;
        }
    }
    double total = 0.0;
    for (std::size_t k = 0; k < found; ++k) total += row[best[k]];
    if (!(total > 0.0)) return;
    for (std::size_t k = 0; k < found; ++k) out[best[k]] = row[best[k]] / total;
}

struct rule {
    // Share of a part's vertices that must sit on one joint's segments.
    double owner_share = 0.5;
    // A joint counts in the weight span when it holds this share of the part.
    double significant = 0.05;
    // Joint-loss guard: no joint may keep less than this share of its weight.
    double keep_joint = 0.5;
};

// Apply the rule to every group of parts (from pair_shells), give each
// rigid group its strongest joint at 100%, and make the outline shells of
// soft groups copy their inner part. `copies[v]` is how many source vertices
// share welded vertex v (empty = one each), so the joint-loss guard counts
// weight the way the exported mesh holds it. `rows` is the dense welded weight
// matrix, `joint_count` floats per vertex. Returns the number of parts made
// rigid.
inline std::size_t snap_rigid_parts(const parts &found, std::span<const std::uint32_t> group,
    std::span<const vec3> position, std::span<float> rows, std::size_t joint_count,
    std::span<const std::int32_t> parents, std::span<const vec3> joints, const rule &settings = {},
    std::span<const std::uint32_t> copies = {}) {
    const auto copies_of = [&](std::size_t v) {
        return v < copies.size() ? static_cast<double>(copies[v]) : 1.0;
    };
    const std::size_t count = found.members.size();
    if (joint_count == 0U || group.size() != count || parents.size() != joint_count || joints.size() != joint_count)
        return 0U;
    std::vector<std::vector<std::uint32_t>> neighbours(joint_count);
    for (std::size_t joint = 0; joint < joint_count; ++joint) {
        const auto parent = parents[joint];
        if (parent < 0 || static_cast<std::size_t>(parent) >= joint_count) continue;
        neighbours[joint].push_back(static_cast<std::uint32_t>(parent));
        neighbours[static_cast<std::size_t>(parent)].push_back(static_cast<std::uint32_t>(joint));
    }
    // Members and closedness per group.
    std::vector<std::vector<std::uint32_t>> groups(count);
    std::vector<char> closed(count, 1);
    for (std::size_t part = 0; part < count; ++part) {
        groups[group[part]].push_back(static_cast<std::uint32_t>(part));
        if (!found.closed[part]) closed[group[part]] = 0;
    }
    const auto owner = owners(position, joints, parents);
    std::vector<double> before(joint_count, 0.0), kept;
    for (std::size_t v = 0; v < position.size(); ++v) {
        kept_four(rows.data() + v * joint_count, joint_count, kept);
        for (std::size_t joint = 0; joint < joint_count; ++joint) before[joint] += copies_of(v) * kept[joint];
    }
    struct choice {
        std::uint32_t group, joint;
        std::vector<double> sum; // weight per joint, one per welded vertex (the rule)
        std::vector<double> counted; // weight per joint, counting copies (the guard)
        double vertices;
    };
    std::vector<choice> chosen;
    std::vector<double> sum(joint_count);
    std::vector<std::uint32_t> tally(joint_count);
    for (std::uint32_t g = 0; g < count; ++g) {
        if (groups[g].empty() || !closed[g]) continue;
        std::fill(sum.begin(), sum.end(), 0.0);
        std::fill(tally.begin(), tally.end(), 0U);
        std::vector<double> counted(joint_count, 0.0);
        double weighted = 0.0;
        std::size_t vertices = 0U;
        for (const auto part : groups[g])
            for (const auto v : found.members[part]) {
                const float *row = rows.data() + static_cast<std::size_t>(v) * joint_count;
                kept_four(row, joint_count, kept);
                for (std::size_t joint = 0; joint < joint_count; ++joint) {
                    sum[joint] += row[joint];
                    counted[joint] += copies_of(v) * kept[joint];
                }
                ++tally[owner[v]];
                ++vertices;
                weighted += copies_of(v);
            }
        const auto strongest = static_cast<std::uint32_t>(std::max_element(sum.begin(), sum.end()) - sum.begin());
        const double total = std::accumulate(sum.begin(), sum.end(), 0.0);
        if (!(total > 0.0) || vertices == 0U) continue;
        // 2. One joint moves most of the part.
        if (static_cast<double>(*std::max_element(tally.begin(), tally.end())) <
            settings.owner_share * static_cast<double>(vertices))
            continue;
        // 3. Every significant joint is the strongest or one bone from it.
        bool near = true;
        for (std::size_t joint = 0; joint < joint_count && near; ++joint) {
            if (joint == strongest || sum[joint] < settings.significant * total) continue;
            near = std::find(neighbours[strongest].begin(), neighbours[strongest].end(), joint) !=
                neighbours[strongest].end();
        }
        if (!near) continue;
        chosen.push_back({g, strongest, sum, std::move(counted), weighted});
    }
    // Joint-loss guard: weight each joint keeps after every chosen snap. A
    // joint left with less than `keep_joint` of its weight gets back the
    // snapped groups that held most of it, largest first, until it keeps
    // enough.
    std::vector<double> after = before;
    for (const auto &c : chosen) {
        for (std::size_t joint = 0; joint < joint_count; ++joint) after[joint] -= c.counted[joint];
        after[c.joint] += c.vertices;
    }
    std::vector<char> reverted(chosen.size(), 0);
    for (std::size_t joint = 0; joint < joint_count; ++joint) {
        if (!(before[joint] >= 1.0)) continue;
        std::vector<std::size_t> holders;
        for (std::size_t k = 0; k < chosen.size(); ++k)
            if (!reverted[k] && chosen[k].joint != joint && chosen[k].counted[joint] > 0.0) holders.push_back(k);
        std::sort(holders.begin(), holders.end(),
            [&](std::size_t a, std::size_t b) { return chosen[a].counted[joint] > chosen[b].counted[joint]; });
        for (const auto k : holders) {
            if (after[joint] >= settings.keep_joint * before[joint]) break;
            reverted[k] = 1;
            for (std::size_t j = 0; j < joint_count; ++j) after[j] += chosen[k].counted[j];
            after[chosen[k].joint] -= chosen[k].vertices;
        }
    }
    std::size_t snapped = 0U;
    std::vector<char> done(count, 0);
    for (std::size_t k = 0; k < chosen.size(); ++k) {
        const auto &c = chosen[k];
        if (reverted[k]) continue;
        done[c.group] = 1;
        for (const auto part : groups[c.group]) {
            for (const auto v : found.members[part]) {
                float *row = rows.data() + static_cast<std::size_t>(v) * joint_count;
                std::fill(row, row + joint_count, 0.0F);
                row[c.joint] = 1.0F;
            }
            ++snapped;
        }
    }
    // A soft part keeps its weights, but its outline shell must bend with
    // it: every shell vertex takes the weights of the nearest vertex of the
    // innermost part (the group member with the smallest box).
    for (std::uint32_t g = 0; g < count; ++g) {
        if (done[g] || groups[g].size() < 2U) continue;
        std::uint32_t core = groups[g].front();
        double smallest = std::numeric_limits<double>::infinity();
        for (const auto part : groups[g]) {
            vec3 low = position[found.members[part].front()], high = low;
            for (const auto v : found.members[part]) {
                const auto q = position[v];
                low = {std::min(low.x, q.x), std::min(low.y, q.y), std::min(low.z, q.z)};
                high = {std::max(high.x, q.x), std::max(high.y, q.y), std::max(high.z, q.z)};
            }
            const double d = static_cast<double>((high.x - low.x) * (high.x - low.x) +
                (high.y - low.y) * (high.y - low.y) + (high.z - low.z) * (high.z - low.z));
            if (d < smallest) {
                smallest = d;
                core = part;
            }
        }
        for (const auto part : groups[g]) {
            if (part == core) continue;
            for (const auto v : found.members[part]) {
                const auto q = position[v];
                std::uint32_t nearest = found.members[core].front();
                float best = std::numeric_limits<float>::infinity();
                for (const auto u : found.members[core]) {
                    const auto r = position[u];
                    const float d2 = (q.x - r.x) * (q.x - r.x) + (q.y - r.y) * (q.y - r.y) + (q.z - r.z) * (q.z - r.z);
                    if (d2 < best) {
                        best = d2;
                        nearest = u;
                    }
                }
                std::copy_n(rows.data() + static_cast<std::size_t>(nearest) * joint_count, joint_count,
                    rows.data() + static_cast<std::size_t>(v) * joint_count);
            }
        }
    }
    return snapped;
}

} // namespace skintokens::detail::rigid
