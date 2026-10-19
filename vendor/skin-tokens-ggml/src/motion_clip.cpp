// Kimodo SOMA30 clips for the viewer: build a motion from motion.elf's transforms, cut it into a
// loop or a one-shot, and add it to a rigged GLB.
#include <skintokens/skintokens.hpp>
#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace skintokens {
namespace {

quat quat_from_rows(const float * m) {
    const float m00 = m[0], m01 = m[1], m02 = m[2], m10 = m[3], m11 = m[4], m12 = m[5], m20 = m[6],
                m21 = m[7], m22 = m[8];
    const float tr = m00 + m11 + m22;
    quat q;
    if (tr > 0.0F) {
        const float s = std::sqrt(tr + 1.0F) * 2.0F;
        q = {(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25F * s};
    } else if (m00 > m11 && m00 > m22) {
        const float s = std::sqrt(1.0F + m00 - m11 - m22) * 2.0F;
        q = {0.25F * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s};
    } else if (m11 > m22) {
        const float s = std::sqrt(1.0F + m11 - m00 - m22) * 2.0F;
        q = {(m01 + m10) / s, 0.25F * s, (m12 + m21) / s, (m02 - m20) / s};
    } else {
        const float s = std::sqrt(1.0F + m22 - m00 - m11) * 2.0F;
        q = {(m02 + m20) / s, (m12 + m21) / s, 0.25F * s, (m10 - m01) / s};
    }
    const float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return {q.x / n, q.y / n, q.z / n, q.w / n};
}

float dot(const quat & a, const quat & b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

float angle_degrees(const quat & a, const quat & b) {
    const double d = std::min(1.0, std::abs(static_cast<double>(dot(a, b))));
    return static_cast<float>(2.0 * std::acos(d) * 180.0 / std::numbers::pi);
}

quat slerp(const quat & a, quat b, float t) {
    float d = dot(a, b);
    if (d < 0.0F) {
        b = {-b.x, -b.y, -b.z, -b.w};
        d = -d;
    }
    quat q;
    if (d > 0.9995F) {
        q = {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t};
    } else {
        const float th = std::acos(d), s = std::sin(th);
        const float wa = std::sin((1.0F - t) * th) / s, wb = std::sin(t * th) / s;
        q = {wa * a.x + wb * b.x, wa * a.y + wb * b.y, wa * a.z + wb * b.z, wa * a.w + wb * b.w};
    }
    const float n = std::sqrt(dot(q, q));
    return {q.x / n, q.y / n, q.z / n, q.w / n};
}

float median(std::vector<float> v) {
    if (v.empty()) return 0.0F;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
    return v[v.size() / 2];
}

} // namespace

result<motion> soma30_motion_from_transforms(
    const std::vector<std::string> & names,
    const std::vector<std::int32_t> & parents,
    float frames_per_second,
    std::span<const float> transforms) {
    const std::size_t joints = names.size();
    if (joints == 0U || parents.size() != joints || transforms.size() % (joints * 12U) != 0U ||
        transforms.empty() || !(frames_per_second > 0.0F))
        return std::unexpected(detail::fail(error_code::invalid_argument,
            "motion transforms do not match the skeleton"));
    for (std::size_t j = 0; j < joints; ++j)
        if (parents[j] >= static_cast<std::int32_t>(j) || (j == 0U) != (parents[j] < 0))
            return std::unexpected(detail::fail(error_code::invalid_argument,
                "motion skeleton parents must precede children, root first"));
    motion out;
    out.frames = transforms.size() / (joints * 12U);
    out.frames_per_second = frames_per_second;
    out.rig.names = names;
    out.rig.parents = parents;
    out.rig.rest_positions.resize(joints);
    // Rest = T-pose: identity rotations, so a joint's global rest is its parent's plus its offset.
    // The root rests where the first frame puts it (as soma_chain_glb.py writes the GLB).
    for (std::size_t j = 0; j < joints; ++j) {
        const float * t = &transforms[j * 12U + 9U];
        const vec3 offset{t[0], t[1], t[2]};
        out.rig.rest_positions[j] = parents[j] < 0 ? offset
            : vec3{out.rig.rest_positions[static_cast<std::size_t>(parents[j])].x + offset.x,
                   out.rig.rest_positions[static_cast<std::size_t>(parents[j])].y + offset.y,
                   out.rig.rest_positions[static_cast<std::size_t>(parents[j])].z + offset.z};
    }
    out.root_translations.resize(out.frames);
    out.local_rotations.resize(out.frames * joints);
    for (std::size_t f = 0; f < out.frames; ++f) {
        const float * root = &transforms[f * joints * 12U];
        out.root_translations[f] = {root[9], root[10], root[11]};
        for (std::size_t j = 0; j < joints; ++j)
            out.local_rotations[f * joints + j] = quat_from_rows(&transforms[(f * joints + j) * 12U]);
    }
    return out;
}

result<std::pair<motion, clip_cut_report>> cut_clip(const motion & in, clip_cut kind, float seconds) {
    const std::size_t J = in.rig.names.size(), F = in.frames;
    if (J == 0U || F < 16U || in.local_rotations.size() != F * J || in.root_translations.size() != F)
        return std::unexpected(detail::fail(error_code::invalid_argument, "clip is too short to cut"));
    const float fps = in.frames_per_second;
    const auto Q = [&](std::size_t f, std::size_t j) -> const quat & { return in.local_rotations[f * J + j]; };
    const auto & R = in.root_translations;
    std::vector<float> energy(F - 1U), speed(F - 1U);
    for (std::size_t f = 0; f + 1U < F; ++f) {
        float e = 0.0F;
        for (std::size_t j = 0; j < J; ++j) e += angle_degrees(Q(f, j), Q(f + 1U, j));
        energy[f] = e;
        speed[f] = std::hypot(R[f + 1U].x - R[f].x, R[f + 1U].z - R[f].z) * fps;
    }
    const auto seam = [&](std::size_t a, std::size_t b, float & max_deg) {
        float sum = 0.0F;
        max_deg = 0.0F;
        for (std::size_t j = 0; j < J; ++j) {
            const float d = angle_degrees(Q(a, j), Q(b, j));
            sum += d;
            max_deg = std::max(max_deg, d);
        }
        const std::size_t a1 = a + 1U, b1 = b + 1U < F ? b + 1U : b;
        const std::size_t b0 = b + 1U < F ? b : b - 1U;
        const float dvx = (R[a1].x - R[a].x) - (R[b1].x - R[b0].x);
        const float dvy = (R[a1].y - R[a].y) - (R[b1].y - R[b0].y);
        const float dvz = (R[a1].z - R[a].z) - (R[b1].z - R[b0].z);
        const float dv = std::sqrt(dvx * dvx + dvy * dvy + dvz * dvz) * fps;
        return sum + 200.0F * std::abs(R[a].y - R[b].y) + 20.0F * dv;
    };

    clip_cut_report report;
    std::vector<std::size_t> frames;
    if (kind == clip_cut::once) {
        const float peak = *std::max_element(energy.begin(), energy.end());
        std::size_t first = F, last = 0;
        for (std::size_t f = 0; f + 1U < F; ++f)
            if (energy[f] > 0.15F * peak) {
                first = std::min(first, f);
                last = f;
            }
        if (first == F) first = 0, last = F - 2U;
        const std::size_t lo = first >= 6U ? first - 6U : 0U, hi = std::min(F, last + 7U);
        for (std::size_t f = lo; f < hi; ++f) frames.push_back(f);
        report.start = lo;
    } else {
        const std::size_t L0 = static_cast<std::size_t>(std::lround(seconds * fps));
        const std::size_t q0 = F / 4U, q1 = 3U * F / 4U;
        const float mid = median({speed.begin() + static_cast<std::ptrdiff_t>(q0), speed.begin() + static_cast<std::ptrdiff_t>(q1)});
        const float emid = median({energy.begin() + static_cast<std::ptrdiff_t>(q0), energy.begin() + static_cast<std::ptrdiff_t>(q1)});
        std::size_t a = F, b = 0;
        for (std::size_t f = 0; f + 1U < F; ++f)
            if (speed[f] >= 0.75F * mid && energy[f] >= 0.5F * emid) {
                a = std::min(a, f);
                b = f;
            }
        if (a == F) a = 6U, b = F - 2U;
        a = std::max<std::size_t>(a, 6U);
        b = std::min<std::size_t>(b, F - 2U);
        const bool still = mid < 0.2F && emid < 15.0F;
        const std::size_t lmin = still ? L0 : std::max<std::size_t>(8U, static_cast<std::size_t>(L0 * 0.6F));
        const std::size_t lmax = still ? L0 : static_cast<std::size_t>(L0 * 1.4F);
        float best = std::numeric_limits<float>::infinity();
        std::size_t bs = 0, bl = 0;
        for (std::size_t L = lmin; L <= lmax; ++L)
            for (std::size_t s = a; s + L < b; ++s) {
                float mx = 0.0F;
                const float c = seam(s, s + L, mx);
                if (c < best) best = c, bs = s, bl = L, report.seam_before_max_degrees = mx;
            }
        if (bl == 0U)
            return std::unexpected(detail::fail(error_code::invalid_argument,
                "the clip has no steady part long enough for the loop"));
        for (std::size_t f = bs; f < bs + bl; ++f) frames.push_back(f);
        report.start = bs;
    }
    const std::size_t L = frames.size();
    motion out;
    out.rig = in.rig;
    out.frames_per_second = fps;
    out.name = in.name;
    out.root_translations.reserve(L + 1U);
    out.local_rotations.reserve((L + 1U) * J);
    for (const auto f : frames) {
        out.root_translations.push_back(R[f]);
        for (std::size_t j = 0; j < J; ++j) out.local_rotations.push_back(Q(f, j));
    }
    const auto OQ = [&](std::size_t i, std::size_t j) -> quat & { return out.local_rotations[i * J + j]; };
    if (kind == clip_cut::loop) {
        // Fade the last K frames toward the frames just before the start, so the wrap is smooth.
        const std::size_t K = std::clamp<std::size_t>(L / 6U, 3U, 8U);
        const std::size_t s = frames.front();
        for (std::size_t i = 0; i < K && K <= s; ++i) {
            const std::size_t o = L - K + i, pre = s - K + i;
            const float t = static_cast<float>(i + 1U) / static_cast<float>(K + 1U);
            for (std::size_t j = 0; j < J; ++j) OQ(o, j) = slerp(OQ(o, j), Q(pre, j), t);
            out.root_translations[o].y += (R[pre].y - out.root_translations[o].y) * t;
        }
        // Remove the ground travel: a ramp so the root comes back to where it started.
        const std::size_t e = std::min(s + L, F - 1U);
        const float dx = R[e].x - R[s].x, dz = R[e].z - R[s].z;
        const float x0 = out.root_translations[0].x, z0 = out.root_translations[0].z;
        for (std::size_t i = 0; i < L; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(L);
            out.root_translations[i].x -= x0 + dx * t;
            out.root_translations[i].z -= z0 + dz * t;
        }
        report.travel_removed_metres_per_second = std::hypot(dx, dz) / (static_cast<float>(L) / fps);
        for (std::size_t j = 0; j < J; ++j)
            report.seam_after_max_degrees = std::max(report.seam_after_max_degrees, angle_degrees(OQ(L - 1U, j), OQ(0, j)));
        // Frame 0 again at the end: the clip's length becomes L / fps and the wrap takes one frame.
        out.root_translations.push_back(out.root_translations.front());
        for (std::size_t j = 0; j < J; ++j) out.local_rotations.push_back(out.local_rotations[j]);
    } else {
        // Keep the one-shot on the spot where it starts.
        const float x0 = out.root_translations[0].x, z0 = out.root_translations[0].z;
        for (auto & p : out.root_translations) p.x -= x0, p.z -= z0;
    }
    for (std::size_t i = 0; i + 1U < L; ++i)
        for (std::size_t j = 0; j < J; ++j)
            report.largest_step_degrees = std::max(report.largest_step_degrees, angle_degrees(OQ(i, j), OQ(i + 1U, j)));
    out.frames = out.root_translations.size();
    report.frames = L;
    return std::pair{std::move(out), report};
}

result<std::pair<std::vector<std::byte>, humanoid_match_report>> add_soma30_motion_glb_memory(
    std::span<const std::byte> rigged,
    const motion & soma30,
    const humanoid_match_options & match_options,
    const retarget_options & options) {
    if (identify_rig(soma30.rig) != rig_kind::soma30)
        return std::unexpected(detail::fail(error_code::invalid_argument, "motion is not a SOMA30 clip"));
    auto asset = load_skinned_glb_memory(rigged);
    if (!asset) return std::unexpected(asset.error());
    auto mapping = match_generated_humanoid(asset->binding.rig, match_options);
    if (!mapping) return std::unexpected(mapping.error());
    auto animation = retarget_soma30_motion_to_generated(soma30, asset->binding.rig, *mapping, options);
    if (!animation) return std::unexpected(animation.error());
    animation->name = soma30.name;
    auto bytes = add_animation_glb_memory(rigged, *animation);
    if (!bytes) return std::unexpected(bytes.error());
    return std::pair{std::move(*bytes), mapping->report};
}

} // namespace skintokens
