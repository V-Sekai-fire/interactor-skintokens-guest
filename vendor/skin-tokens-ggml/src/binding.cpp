#include "binding.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <queue>
#include <limits>
#include <unordered_map>

namespace skintokens::detail {
namespace {

float squared_distance(vec3 left, vec3 right) {
    const float x = left.x - right.x;
    const float y = left.y - right.y;
    const float z = left.z - right.z;
    return x*x + y*y + z*z;
}

float distance(vec3 left, vec3 right) {
    return std::sqrt(squared_distance(left, right));
}

struct grid_cell {
    std::int64_t x, y, z;
    bool operator==(const grid_cell &) const = default;
};

struct grid_cell_hash {
    std::size_t operator()(const grid_cell & value) const noexcept {
        // Copied from Open3D 0.19 utility::hash_eigen. VoxelGrid stores its
        // cells in an unordered_map and get_voxels() exposes that iteration
        // order; upstream then feeds the resulting order to SciPy cKDTree.
        std::size_t result = 0U;
        result ^= std::hash<std::int64_t>{}(value.x) + 0x9e3779b9U + (result << 6U) + (result >> 2U);
        result ^= std::hash<std::int64_t>{}(value.y) + 0x9e3779b9U + (result << 6U) + (result >> 2U);
        result ^= std::hash<std::int64_t>{}(value.z) + 0x9e3779b9U + (result << 6U) + (result >> 2U);
        return result;
    }
};

struct dvec3 { double x, y, z; };

// Minimal k=1 port of SciPy 1.15.3 cKDTree's default balanced/compact tree.
// SkinTokens uses cKDTree at the final voxel_skin stage, and its first-hit
// behaviour for coincident seam vertices affects which connected component a
// joint seeds. Keep this local implementation in lock-step with that pinned
// reference instead of silently choosing the lowest vertex index.
class scipy_point_tree {
    struct node {
        std::size_t begin = 0U;
        std::size_t end = 0U;
        std::int32_t split_dimension = -1;
        double split = 0.0;
        std::size_t less = 0U;
        std::size_t greater = 0U;
    };
    struct pending_item {
        double priority;
        std::size_t node_index;
        std::array<double, 3> side_distances;
    };

    std::vector<dvec3> points_;
    std::vector<std::size_t> order_;
    std::vector<node> nodes_;
    std::array<double, 3> minimum_{};
    std::array<double, 3> maximum_{};

    static double coordinate(const dvec3 & value, std::size_t dimension) {
        return dimension == 0U ? value.x : dimension == 1U ? value.y : value.z;
    }

    std::size_t build(std::size_t begin, std::size_t end) {
        const auto result = nodes_.size();
        nodes_.push_back({begin, end});
        if (end - begin <= 16U) return result;

        std::array<double, 3> low{}, high{};
        for (std::size_t dimension = 0U; dimension < 3U; ++dimension)
            low[dimension] = high[dimension] = coordinate(points_[order_[begin]], dimension);
        for (std::size_t item = begin + 1U; item < end; ++item)
            for (std::size_t dimension = 0U; dimension < 3U; ++dimension) {
                const auto value = coordinate(points_[order_[item]], dimension);
                low[dimension] = std::min(low[dimension], value);
                high[dimension] = std::max(high[dimension], value);
            }
        std::size_t dimension = 0U;
        double spread = 0.0;
        for (std::size_t candidate = 0U; candidate < 3U; ++candidate)
            if (high[candidate] - low[candidate] > spread) {
                dimension = candidate;
                spread = high[candidate] - low[candidate];
            }
        if (high[dimension] == low[dimension]) return result;

        const auto compare = [&](std::size_t left, std::size_t right) {
            return coordinate(points_[left], dimension) < coordinate(points_[right], dimension);
        };
        auto first = order_.begin() + static_cast<std::ptrdiff_t>(begin);
        auto middle = first + static_cast<std::ptrdiff_t>((end - begin) / 2U);
        auto last = order_.begin() + static_cast<std::ptrdiff_t>(end);
        std::nth_element(first, middle, last, compare);
        double split = coordinate(points_[*middle], dimension);
        auto partition = std::partition(first, middle, [&](std::size_t index) {
            return coordinate(points_[index], dimension) < split;
        });
        auto pivot = static_cast<std::size_t>(partition - order_.begin());
        if (pivot == begin) {
            const auto smallest = *std::min_element(first, last, compare);
            split = std::nextafter(coordinate(points_[smallest], dimension),
                                   std::numeric_limits<double>::infinity());
            partition = std::partition(first, last, [&](std::size_t index) {
                return coordinate(points_[index], dimension) < split;
            });
            pivot = static_cast<std::size_t>(partition - order_.begin());
        } else if (pivot == end) {
            const auto largest = *std::max_element(first, last, compare);
            split = coordinate(points_[largest], dimension);
            partition = std::partition(first, last, [&](std::size_t index) {
                return coordinate(points_[index], dimension) < split;
            });
            pivot = static_cast<std::size_t>(partition - order_.begin());
        }
        if (pivot == begin || pivot == end) return result;

        const auto less = build(begin, pivot);
        const auto greater = build(pivot, end);
        nodes_[result].split_dimension = static_cast<std::int32_t>(dimension);
        nodes_[result].split = split;
        nodes_[result].less = less;
        nodes_[result].greater = greater;
        return result;
    }

    // SciPy's heap compares priorities only and leaves equal-priority entries
    // in insertion order. Reproduce that rather than adding an index tie-break.
    static void heap_push(std::vector<pending_item> & heap, pending_item item) {
        heap.push_back(item);
        std::size_t index = heap.size() - 1U;
        while (index > 0U) {
            const auto parent = (index - 1U) / 2U;
            if (!(heap[index].priority < heap[parent].priority)) break;
            std::swap(heap[index], heap[parent]);
            index = parent;
        }
    }

    static pending_item heap_pop(std::vector<pending_item> & heap) {
        auto output = heap.front();
        heap.front() = heap.back();
        heap.pop_back();
        std::size_t index = 0U;
        while (true) {
            const auto left = index * 2U + 1U;
            const auto right = left + 1U;
            if (left >= heap.size()) break;
            auto selected = left;
            if (right < heap.size() && heap[left].priority > heap[right].priority)
                selected = right;
            if (!(heap[index].priority > heap[selected].priority)) break;
            std::swap(heap[index], heap[selected]);
            index = selected;
        }
        return output;
    }

public:
    explicit scipy_point_tree(std::vector<dvec3> points)
        : points_(std::move(points)), order_(points_.size()) {
        if (points_.empty()) return;
        std::iota(order_.begin(), order_.end(), 0U);
        for (std::size_t dimension = 0U; dimension < 3U; ++dimension)
            minimum_[dimension] = maximum_[dimension] = coordinate(points_.front(), dimension);
        for (const auto & point : points_)
            for (std::size_t dimension = 0U; dimension < 3U; ++dimension) {
                minimum_[dimension] = std::min(minimum_[dimension], coordinate(point, dimension));
                maximum_[dimension] = std::max(maximum_[dimension], coordinate(point, dimension));
            }
        nodes_.reserve(points_.size() / 8U);
        build(0U, points_.size());
    }

    std::uint32_t nearest(dvec3 query) const {
        if (points_.empty()) return 0U;
        std::array<double, 3> side{};
        double initial_distance = 0.0;
        for (std::size_t dimension = 0U; dimension < 3U; ++dimension) {
            const auto value = coordinate(query, dimension);
            const auto delta = value < minimum_[dimension] ? minimum_[dimension] - value :
                               value > maximum_[dimension] ? value - maximum_[dimension] : 0.0;
            side[dimension] = delta * delta;
            initial_distance += side[dimension];
        }
        pending_item current{initial_distance, 0U, side};
        std::vector<pending_item> pending;
        pending.reserve(64U);
        double best_distance = std::numeric_limits<double>::infinity();
        std::size_t best = points_.size();
        while (true) {
            const auto & current_node = nodes_[current.node_index];
            if (current_node.split_dimension < 0) {
                for (std::size_t item = current_node.begin; item < current_node.end; ++item) {
                    const auto index = order_[item];
                    const auto dx = points_[index].x - query.x;
                    const auto dy = points_[index].y - query.y;
                    const auto dz = points_[index].z - query.z;
                    const auto candidate = dx*dx + dy*dy + dz*dz;
                    if (candidate < best_distance) {
                        best_distance = candidate;
                        best = index;
                    }
                }
                if (pending.empty()) break;
                current = heap_pop(pending);
                continue;
            }
            if (current.priority > best_distance) break;
            const auto dimension = static_cast<std::size_t>(current_node.split_dimension);
            auto far = current;
            const auto query_coordinate = coordinate(query, dimension);
            const auto delta = query_coordinate < current_node.split ?
                current_node.split - query_coordinate : query_coordinate - current_node.split;
            if (query_coordinate < current_node.split) {
                current.node_index = current_node.less;
                far.node_index = current_node.greater;
            } else {
                current.node_index = current_node.greater;
                far.node_index = current_node.less;
            }
            const auto new_side = delta * delta;
            far.priority += new_side - far.side_distances[dimension];
            far.side_distances[dimension] = new_side;
            if (far.priority <= best_distance) heap_push(pending, far);
        }
        return static_cast<std::uint32_t>(best);
    }

    std::vector<std::pair<float, std::uint32_t>> nearest(dvec3 query, std::size_t count) const {
        if (points_.empty() || count == 0U) return {};
        count = std::min(count, points_.size());
        std::array<double, 3> side{};
        double initial_distance = 0.0;
        for (std::size_t dimension = 0U; dimension < 3U; ++dimension) {
            const auto value = coordinate(query, dimension);
            const auto delta = value < minimum_[dimension] ? minimum_[dimension] - value :
                               value > maximum_[dimension] ? value - maximum_[dimension] : 0.0;
            side[dimension] = delta * delta;
            initial_distance += side[dimension];
        }
        pending_item current{initial_distance, 0U, side};
        std::vector<pending_item> pending;
        pending.reserve(64U);
        using neighbor = std::pair<double, std::uint32_t>;
        std::priority_queue<neighbor> neighbors;
        double distance_upper_bound = std::numeric_limits<double>::infinity();
        while (true) {
            const auto & current_node = nodes_[current.node_index];
            if (current_node.split_dimension < 0) {
                for (std::size_t item = current_node.begin; item < current_node.end; ++item) {
                    const auto index = order_[item];
                    const auto dx = points_[index].x - query.x;
                    const auto dy = points_[index].y - query.y;
                    const auto dz = points_[index].z - query.z;
                    const auto candidate = dx*dx + dy*dy + dz*dz;
                    if (candidate < distance_upper_bound) {
                        if (neighbors.size() == count) neighbors.pop();
                        neighbors.emplace(candidate, static_cast<std::uint32_t>(index));
                        if (neighbors.size() == count) distance_upper_bound = neighbors.top().first;
                    }
                }
                if (pending.empty()) break;
                current = heap_pop(pending);
                continue;
            }
            if (current.priority > distance_upper_bound) break;
            const auto dimension = static_cast<std::size_t>(current_node.split_dimension);
            auto far = current;
            const auto query_coordinate = coordinate(query, dimension);
            const auto delta = query_coordinate < current_node.split ?
                current_node.split - query_coordinate : query_coordinate - current_node.split;
            if (query_coordinate < current_node.split) {
                current.node_index = current_node.less;
                far.node_index = current_node.greater;
            } else {
                current.node_index = current_node.greater;
                far.node_index = current_node.less;
            }
            const auto new_side = delta * delta;
            far.priority += new_side - far.side_distances[dimension];
            far.side_distances[dimension] = new_side;
            if (far.priority <= distance_upper_bound) heap_push(pending, far);
        }
        std::vector<std::pair<float, std::uint32_t>> output(neighbors.size());
        for (std::size_t index = output.size(); index > 0U; --index) {
            output[index - 1U] = {static_cast<float>(neighbors.top().first), neighbors.top().second};
            neighbors.pop();
        }
        return output;
    }
};

dvec3 subtract(dvec3 a, dvec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
dvec3 cross(dvec3 a, dvec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
double dot(dvec3 a, dvec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }

// Separating-axis triangle/AABB test used by Open3D's mesh voxelizer.
bool triangle_box(dvec3 center, dvec3 half, dvec3 a, dvec3 b, dvec3 c) {
    a = subtract(a, center); b = subtract(b, center); c = subtract(c, center);
    const dvec3 edges[]{subtract(b, a), subtract(c, b), subtract(a, c)};
    const dvec3 box_axes[]{{1,0,0}, {0,1,0}, {0,0,1}};
    const auto overlap = [&](dvec3 axis) {
        const double magnitude = dot(axis, axis);
        if (magnitude <= 1.0e-30) return true;
        const double pa=dot(a,axis), pb=dot(b,axis), pc=dot(c,axis);
        const double radius=half.x*std::abs(axis.x)+half.y*std::abs(axis.y)+half.z*std::abs(axis.z);
        return std::min({pa,pb,pc}) <= radius && std::max({pa,pb,pc}) >= -radius;
    };
    for (const auto axis : box_axes) if (!overlap(axis)) return false;
    if (!overlap(cross(edges[0], edges[1]))) return false;
    for (const auto edge : edges)
        for (const auto axis : box_axes)
            if (!overlap(cross(edge, axis))) return false;
    return true;
}

std::vector<grid_cell> voxelize_like_open3d(
    std::span<const vec3> vertices, std::span<const triangle> faces, double voxel_size) {
    dvec3 minimum{vertices.front().x, vertices.front().y, vertices.front().z};
    for (const auto value : vertices) {
        minimum.x=std::min(minimum.x,static_cast<double>(value.x));
        minimum.y=std::min(minimum.y,static_cast<double>(value.y));
        minimum.z=std::min(minimum.z,static_cast<double>(value.z));
    }
    minimum.x-=voxel_size*.5; minimum.y-=voxel_size*.5; minimum.z-=voxel_size*.5;
    const dvec3 half{voxel_size*.5,voxel_size*.5,voxel_size*.5};
    std::unordered_map<grid_cell, bool, grid_cell_hash> occupied;
    for (const auto face : faces) {
        if (face[0]>=vertices.size() || face[1]>=vertices.size() || face[2]>=vertices.size()) continue;
        const dvec3 value[]{
            {vertices[face[0]].x,vertices[face[0]].y,vertices[face[0]].z},
            {vertices[face[1]].x,vertices[face[1]].y,vertices[face[1]].z},
            {vertices[face[2]].x,vertices[face[2]].y,vertices[face[2]].z}};
        const dvec3 low{std::min({value[0].x,value[1].x,value[2].x}),
                        std::min({value[0].y,value[1].y,value[2].y}),
                        std::min({value[0].z,value[1].z,value[2].z})};
        const dvec3 high{std::max({value[0].x,value[1].x,value[2].x}),
                         std::max({value[0].y,value[1].y,value[2].y}),
                         std::max({value[0].z,value[1].z,value[2].z})};
        const std::int64_t ix=static_cast<std::int64_t>(std::floor((low.x-minimum.x)/voxel_size));
        const std::int64_t iy=static_cast<std::int64_t>(std::floor((low.y-minimum.y)/voxel_size));
        const std::int64_t iz=static_cast<std::int64_t>(std::floor((low.z-minimum.z)/voxel_size));
        const std::int64_t nx=static_cast<std::int64_t>(std::round((high.x-low.x)/voxel_size))+2;
        const std::int64_t ny=static_cast<std::int64_t>(std::round((high.y-low.y)/voxel_size))+2;
        const std::int64_t nz=static_cast<std::int64_t>(std::round((high.z-low.z)/voxel_size))+2;
        for (std::int64_t x=ix; x<ix+nx; ++x)
            for (std::int64_t y=iy; y<iy+ny; ++y)
                for (std::int64_t z=iz; z<iz+nz; ++z) {
                    const dvec3 center{minimum.x+(static_cast<double>(x)+.5)*voxel_size,
                                       minimum.y+(static_cast<double>(y)+.5)*voxel_size,
                                       minimum.z+(static_cast<double>(z)+.5)*voxel_size};
                    if (triangle_box(center, half, value[0], value[1], value[2]))
                        occupied[{x,y,z}]=true;
                }
    }
    std::vector<grid_cell> result;
    result.reserve(occupied.size());
    for (const auto & [value, unused] : occupied) result.push_back(value);
    return result;
}

// Reproduce the upstream demo's complete voxel_skin graph, including its use
// of Open3D integer grid indices as graph coordinates.
struct surface_result {
    std::vector<float> weights;
    std::vector<std::int32_t> voxel_coordinates;
    std::vector<std::uint32_t> seeds;
    std::vector<float> distances;
    std::size_t graph_edges = 0U;
    float maximum_distance = 0.0F;
    std::vector<std::uint32_t> graph_sources;
    std::vector<std::uint32_t> graph_targets;
    std::vector<float> graph_weights;
};

surface_result surface_weights(
    std::span<const vec3> vertices,
    std::span<const triangle> faces,
    std::span<const vec3> joints,
    std::span<const precise_vec3> precise_vertices = {},
    std::span<const precise_vec3> precise_joints = {}) {
    // Match Asset.voxel() exactly. Upstream computes the largest coordinate
    // spread within any one vertex, rather than the mesh AABB's largest axis.
    double maximum_spread = 0.0;
    for (const auto value : vertices)
        maximum_spread=std::max(maximum_spread, static_cast<double>(
            std::max({value.x,value.y,value.z})-std::min({value.x,value.y,value.z})));
    const double voxel_size=maximum_spread/196.0;
    const auto grid=voxelize_like_open3d(vertices,faces,voxel_size);
    const std::size_t mesh_count=vertices.size(), total_count=mesh_count+grid.size();
    const auto position = [&](std::size_t index) {
        if (index<mesh_count) return vertices[index];
        const auto value=grid[index-mesh_count];
        return vec3{static_cast<float>(value.x),static_cast<float>(value.y),static_cast<float>(value.z)};
    };
    std::vector<dvec3> combined;
    combined.reserve(total_count);
    for (const auto value : vertices) combined.push_back({value.x, value.y, value.z});
    for (const auto value : grid)
        combined.push_back({static_cast<double>(value.x), static_cast<double>(value.y),
                            static_cast<double>(value.z)});
    const scipy_point_tree combined_tree{std::move(combined)};
    using edge = std::pair<std::uint32_t, float>;
    std::unordered_map<std::uint64_t, float> directed_edges;
    directed_edges.reserve(faces.size() * 3U);
    const auto connect_weighted = [&](std::uint32_t left, std::uint32_t right, float weight) {
        if (left >= total_count || right >= total_count || left == right) return;
        if (weight < 0.0F || !std::isfinite(weight)) return;
        const auto key = (static_cast<std::uint64_t>(left) << 32U) | right;
        // scipy.sparse.csr_matrix sums duplicate (row, column) entries before
        // shortest_path(..., directed=False) exposes each directed CSR entry
        // from both endpoints.
        directed_edges[key] += weight;
    };
    const auto connect = [&](std::uint32_t left, std::uint32_t right) {
        connect_weighted(left, right, distance(position(left), position(right)));
    };
    for (const auto & face : faces) {
        connect(face[0], face[1]);
        connect(face[1], face[2]);
        connect(face[2], face[0]);
    }

    // voxel_skin also links each vertex to its three nearest neighbours when
    // they are within 1e-5. This reconnects coincident vertices introduced at
    // GLB seams; omitting it splits an otherwise continuous character into
    // components and produces apparently random remote influences.
    constexpr float link_distance = 1.0e-5F;
    std::vector<dvec3> vertex_points;
    vertex_points.reserve(vertices.size());
    for (const auto value : vertices) vertex_points.push_back({value.x, value.y, value.z});
    const scipy_point_tree vertex_tree{std::move(vertex_points)};
    const auto precise_position = [&](std::size_t index) {
        if (precise_vertices.size() == vertices.size()) return precise_vertices[index];
        const auto value=vertices[index];
        return precise_vec3{value.x,value.y,value.z};
    };
    const auto close_cell = [](precise_vec3 value) {
        return grid_cell{static_cast<std::int64_t>(std::floor(value.x / link_distance)),
                         static_cast<std::int64_t>(std::floor(value.y / link_distance)),
                         static_cast<std::int64_t>(std::floor(value.z / link_distance))};
    };
    std::unordered_map<grid_cell, std::vector<std::uint32_t>, grid_cell_hash> close_buckets;
    close_buckets.reserve(vertices.size());
    for (std::size_t vertex = 0; vertex < vertices.size(); ++vertex)
        close_buckets[close_cell(precise_position(vertex))].push_back(static_cast<std::uint32_t>(vertex));
    for (std::size_t vertex = 0; vertex < vertices.size(); ++vertex) {
        std::vector<std::pair<double, std::uint32_t>> nearest;
        const auto point=precise_position(vertex);
        const auto home = close_cell(point);
        for (std::int64_t x=-1; x<=1; ++x)
            for (std::int64_t y=-1; y<=1; ++y)
                for (std::int64_t z=-1; z<=1; ++z) {
                    const auto found=close_buckets.find({home.x+x,home.y+y,home.z+z});
                    if (found==close_buckets.end()) continue;
                    for (const auto candidate : found->second) {
                        const auto other=precise_position(candidate);
                        const double dx=point.x-other.x,dy=point.y-other.y,dz=point.z-other.z;
                        const double squared=dx*dx+dy*dy+dz*dz;
                        if (squared < static_cast<double>(link_distance)*link_distance)
                            nearest.emplace_back(squared,candidate);
                    }
                }
        std::sort(nearest.begin(),nearest.end());
        if (nearest.size()>4U) nearest.resize(4U);
        // cKDTree.query(vertices, 4) discards the first returned neighbour,
        // even when coincident seam duplicates mean that it is not `vertex`.
        for (std::size_t index = 1U; index < nearest.size(); ++index) {
            const float candidate = static_cast<float>(std::sqrt(nearest[index].first));
            if (candidate > 0.0F && candidate < link_distance)
                connect_weighted(static_cast<std::uint32_t>(vertex), nearest[index].second, candidate);
        }
    }

    // With normalized inputs voxel_size*1.74 is below one, so distinct integer
    // grid coordinates never pass upstream's grid-to-grid distance mask.
    // Grid-to-mesh links can only exist for the handful of integer positions
    // lying inside the normalized mesh bounds; evaluate those exactly.
    const float grid_range=static_cast<float>(voxel_size*1.74);
    for (std::size_t grid_index=0; grid_index<grid.size(); ++grid_index) {
        const vec3 grid_position{static_cast<float>(grid[grid_index].x),
                                 static_cast<float>(grid[grid_index].y),
                                 static_cast<float>(grid[grid_index].z)};
        const auto nearest = vertex_tree.nearest(
            {grid_position.x, grid_position.y, grid_position.z}, 27U);
        for (const auto & [squared, vertex] : nearest) {
            const float candidate = std::sqrt(squared);
            if (candidate > 0.0F && candidate < grid_range)
                connect(static_cast<std::uint32_t>(mesh_count + grid_index), vertex);
        }
    }

    std::vector<std::vector<edge>> adjacency(total_count);
    for (const auto [key, weight] : directed_edges) {
        const auto left = static_cast<std::uint32_t>(key >> 32U);
        const auto right = static_cast<std::uint32_t>(key & 0xffffffffU);
        adjacency[left].emplace_back(right, weight);
        adjacency[right].emplace_back(left, weight);
    }

    const std::size_t joint_count = joints.size();
    const auto precise_joint_position = [&](std::size_t index) {
        if (precise_joints.size() == joints.size()) return precise_joints[index];
        const auto value=joints[index];
        return precise_vec3{value.x,value.y,value.z};
    };
    std::vector<float> distances(total_count * joint_count,
                                 std::numeric_limits<float>::infinity());
    using queued = std::pair<float, std::uint32_t>;
    std::vector<std::uint32_t> seeds(joint_count);
    for (std::size_t joint = 0; joint < joint_count; ++joint) {
        const auto seed = combined_tree.nearest(
            {joints[joint].x, joints[joint].y, joints[joint].z});
        auto * row = distances.data() + joint * total_count;
        seeds[joint]=seed;
        row[seed] = 0.0F;
        std::priority_queue<queued, std::vector<queued>, std::greater<>> pending;
        pending.emplace(0.0F, seed);
        while (!pending.empty()) {
            const auto [current_distance, vertex] = pending.top();
            pending.pop();
            if (current_distance != row[vertex]) continue;
            for (const auto [next, weight] : adjacency[vertex]) {
                const float candidate = current_distance + weight;
                if (candidate >= row[next]) continue;
                row[next] = candidate;
                pending.emplace(candidate, next);
            }
        }
    }

    // Upstream fills the three nearest joint distances for components that no
    // graph seed can reach before finding max_dis. The fallback distances must
    // therefore participate in the value used to replace all remaining infs.
    for (std::size_t vertex = 0; vertex < vertices.size(); ++vertex) {
        bool unreachable = true;
        for (std::size_t joint = 0; joint < joint_count; ++joint)
            unreachable = unreachable && !std::isfinite(distances[joint * total_count + vertex]);
        if (unreachable) {
            const auto point=precise_position(vertex);
            std::vector<std::pair<double,std::uint32_t>> nearest;
            nearest.reserve(joint_count);
            for (std::size_t joint=0; joint<joint_count; ++joint) {
                const auto other=precise_joint_position(joint);
                const double x=point.x-other.x,y=point.y-other.y,z=point.z-other.z;
                nearest.emplace_back(x*x+y*y+z*z,static_cast<std::uint32_t>(joint));
            }
            std::partial_sort(nearest.begin(),nearest.begin()+3,nearest.end(),
                [](const auto & left,const auto & right) {
                    return left.first!=right.first ? left.first<right.first : left.second>right.second;
                });
            for (std::size_t index=0; index<3U; ++index)
                distances[static_cast<std::size_t>(nearest[index].second)*total_count+vertex]=
                    static_cast<float>(std::sqrt(nearest[index].first));
        }
    }
    float maximum = 0.0F;
    for (std::size_t joint=0; joint<joint_count; ++joint)
        for (std::size_t vertex=0; vertex<mesh_count; ++vertex) {
            const float value=distances[joint*total_count+vertex];
            if (std::isfinite(value)) maximum=std::max(maximum,value);
        }
    maximum = std::max(maximum, 1.0e-6F);
    for (std::size_t joint=0; joint<joint_count; ++joint)
        for (std::size_t vertex=0; vertex<mesh_count; ++vertex) {
            auto & value=distances[joint*total_count+vertex];
            if (!std::isfinite(value)) value=maximum;
            value=std::max(value,1.0e-6F);
        }

    std::vector<float> output(vertices.size() * joint_count);
    for (std::size_t vertex = 0; vertex < vertices.size(); ++vertex) {
        float sum = 0.0F;
        for (std::size_t joint = 0; joint < joint_count; ++joint) {
            float value = distances[joint * total_count + vertex];
            const float denominator = 0.5F * value + 0.5F * value * value;
            const float weight = 1.0F / (denominator * denominator);
            output[vertex * joint_count + joint] = weight;
            sum += weight;
        }
        if (sum > 0.0F)
            for (std::size_t joint = 0; joint < joint_count; ++joint)
                output[vertex * joint_count + joint] /= sum;
    }
    surface_result result;
    result.weights=std::move(output);
    result.seeds=std::move(seeds);
    result.graph_edges=directed_edges.size();
    result.graph_sources.reserve(directed_edges.size());
    result.graph_targets.reserve(directed_edges.size());
    result.graph_weights.reserve(directed_edges.size());
    for (const auto [key, weight] : directed_edges) {
        result.graph_sources.push_back(static_cast<std::uint32_t>(key >> 32U));
        result.graph_targets.push_back(static_cast<std::uint32_t>(key & 0xffffffffU));
        result.graph_weights.push_back(weight);
    }
    result.maximum_distance=maximum;
    result.distances.resize(joint_count * mesh_count);
    for (std::size_t joint=0; joint<joint_count; ++joint)
        std::copy_n(distances.data()+joint*total_count, mesh_count,
                    result.distances.data()+joint*mesh_count);
    result.voxel_coordinates.reserve(grid.size()*3U);
    for (const auto value:grid) {
        result.voxel_coordinates.push_back(static_cast<std::int32_t>(value.x));
        result.voxel_coordinates.push_back(static_cast<std::int32_t>(value.y));
        result.voxel_coordinates.push_back(static_cast<std::int32_t>(value.z));
    }
    return result;
}

skin integrate(
    const skeleton & target,
    std::span<const vec3> normalized_vertices,
    std::span<const vec3> sampled_points,
    std::span<const std::vector<float>> dense_joint_weights,
    std::span<const float> locality,
    binding_trace * trace) {
    skin output;
    output.rig = target;
    output.learned = true;
    output.joints.resize(normalized_vertices.size());
    output.weights.resize(normalized_vertices.size());
    if (trace != nullptr) {
        trace->neighbor_indices.resize(normalized_vertices.size() * 8U);
        trace->interpolation_weights.resize(normalized_vertices.size() * 8U);
        trace->surface_weights.assign(locality.begin(), locality.end());
        trace->final_dense_weights.resize(normalized_vertices.size() * dense_joint_weights.size());
    }

    std::vector<dvec3> sampled_double;
    sampled_double.reserve(sampled_points.size());
    for (const auto value : sampled_points) sampled_double.push_back({value.x, value.y, value.z});
    const scipy_point_tree tree{std::move(sampled_double)};
    struct binding_candidate { float learned_weight; std::uint16_t joint; };
    for (std::size_t vertex = 0; vertex < normalized_vertices.size(); ++vertex) {
        const auto point = normalized_vertices[vertex];
        const auto neighbors = tree.nearest({point.x, point.y, point.z}, 8U);
        std::array<float, 8> interpolation{};
        float interpolation_sum = 0.0F;
        for (std::size_t index = 0; index < neighbors.size(); ++index) {
            interpolation[index] = 1.0F / (std::sqrt(neighbors[index].first) + 1e-8F);
            interpolation_sum += interpolation[index];
            if (trace != nullptr) {
                trace->neighbor_indices[vertex * 8U + index] = neighbors[index].second;
                trace->interpolation_weights[vertex * 8U + index] = interpolation[index];
            }
        }

        std::vector<binding_candidate> ranked;
        ranked.reserve(dense_joint_weights.size());
        float dense_sum = 0.0F;
        for (std::size_t joint = 0; joint < dense_joint_weights.size(); ++joint) {
            float value = 0.0F;
            for (std::size_t index = 0; index < neighbors.size(); ++index)
                value += interpolation[index] * dense_joint_weights[joint][neighbors[index].second];
            value /= interpolation_sum;
            if (!locality.empty()) value *= locality[vertex * dense_joint_weights.size() + joint];
            ranked.push_back({value, static_cast<std::uint16_t>(joint)});
            dense_sum += value;
        }
        // The demo calls Asset.normalize_skin() after multiplying by
        // voxel_skin and before Blender chooses four influences. This row
        // normalization does not change the ranking, but retaining it here
        // makes the final dense trace match the actual upstream endpoint.
        if (!locality.empty() && dense_sum > 1.0e-30F) {
            for (auto & value : ranked) value.learned_weight /= dense_sum;
        }
        if (trace != nullptr) {
            for (const auto value : ranked)
                trace->final_dense_weights[vertex * dense_joint_weights.size() + value.joint] =
                    value.learned_weight;
        }
        const auto keep = std::min<std::size_t>(4U, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(keep), ranked.end(),
            [](const auto & left, const auto & right) {
                if (left.learned_weight != right.learned_weight)
                    return left.learned_weight > right.learned_weight;
                return left.joint < right.joint;
            });
        float sum = 0.0F;
        for (std::size_t slot = 0; slot < 4U; ++slot) {
            const auto value = slot < keep ? ranked[slot] : binding_candidate{0.0F, 0U};
            output.weights[vertex][slot] = value.learned_weight;
            output.joints[vertex][slot] = value.joint;
            sum += value.learned_weight;
        }
        if (sum <= 1e-12F) {
            output.weights[vertex][0] = 1.0F;
            sum = 1.0F;
        }
        for (auto & value : output.weights[vertex]) value /= sum;
    }
    return output;
}

// ---- Robust weight transfer -------------------------------------------------

// Same order as integrate(): greatest learned weight first, lower joint on ties.
void keep_four(const float * row, std::size_t joint_count,
               std::array<std::uint16_t, 4> & joints, std::array<float, 4> & weights) {
    std::array<std::pair<float, std::uint16_t>, 4> best{};
    for (auto & entry : best) entry = {-1.0F, 0U};
    for (std::size_t joint = 0; joint < joint_count; ++joint) {
        const std::pair<float, std::uint16_t> candidate{row[joint], static_cast<std::uint16_t>(joint)};
        const auto better = [](const auto & left, const auto & right) {
            return left.first != right.first ? left.first > right.first : left.second < right.second;
        };
        if (!better(candidate, best.back())) continue;
        best.back() = candidate;
        std::sort(best.begin(), best.end(), better);
    }
    float sum = 0.0F;
    for (std::size_t slot = 0; slot < 4U; ++slot) {
        const bool used = slot < joint_count && best[slot].first > 0.0F;
        joints[slot] = used ? best[slot].second : 0U;
        weights[slot] = used ? best[slot].first : 0.0F;
        sum += weights[slot];
    }
    if (sum <= 1.0e-12F) {
        joints = {best[0].second, 0U, 0U, 0U};
        weights = {1.0F, 0.0F, 0.0F, 0.0F};
        return;
    }
    for (auto & value : weights) value /= sum;
}

} // namespace

skin integrate_robust_binding(
    const skeleton & target,
    std::span<const vec3> normalized_vertices,
    std::span<const vec3> vertex_normals,
    std::span<const triangle> faces,
    std::span<const vec3> sampled_points,
    std::span<const vec3> sampled_normals,
    std::span<const std::uint32_t> sampled_faces,
    std::span<const std::vector<float>> dense_joint_weights,
    const robust_transfer_options & options) {
    const std::size_t n = normalized_vertices.size();
    const std::size_t joint_count = dense_joint_weights.size();
    skin output;
    output.rig = target;
    output.learned = true;
    output.transfer.robust = true;
    output.joints.resize(n);
    output.weights.resize(n);
    if (n == 0U || joint_count == 0U || sampled_points.empty()) return output;

    // 1. Weld coincident vertices (seams) within 1e-6 of the box diagonal.
    vec3 low = normalized_vertices.front(), high = low;
    for (const auto p : normalized_vertices) {
        low = {std::min(low.x, p.x), std::min(low.y, p.y), std::min(low.z, p.z)};
        high = {std::max(high.x, p.x), std::max(high.y, p.y), std::max(high.z, p.z)};
    }
    const double diagonal = std::max(static_cast<double>(distance(low, high)), 1.0e-12);
    const double weld = diagonal * 1.0e-6;
    const auto cell_of = [&](vec3 p) {
        return grid_cell{static_cast<std::int64_t>(std::floor(p.x / weld)),
                         static_cast<std::int64_t>(std::floor(p.y / weld)),
                         static_cast<std::int64_t>(std::floor(p.z / weld))};
    };
    std::unordered_map<grid_cell, std::vector<std::uint32_t>, grid_cell_hash> cells;
    cells.reserve(n);
    std::vector<std::uint32_t> welded_of(n);
    std::vector<vec3> position;
    std::vector<dvec3> normal_sum;
    for (std::size_t vertex = 0; vertex < n; ++vertex) {
        const auto p = normalized_vertices[vertex];
        const auto home = cell_of(p);
        std::uint32_t found = std::numeric_limits<std::uint32_t>::max();
        for (std::int64_t x = -1; x <= 1 && found == std::numeric_limits<std::uint32_t>::max(); ++x)
            for (std::int64_t y = -1; y <= 1 && found == std::numeric_limits<std::uint32_t>::max(); ++y)
                for (std::int64_t z = -1; z <= 1 && found == std::numeric_limits<std::uint32_t>::max(); ++z) {
                    const auto it = cells.find({home.x + x, home.y + y, home.z + z});
                    if (it == cells.end()) continue;
                    for (const auto candidate : it->second)
                        if (static_cast<double>(distance(position[candidate], p)) <= weld) { found = candidate; break; }
                }
        if (found == std::numeric_limits<std::uint32_t>::max()) {
            found = static_cast<std::uint32_t>(position.size());
            position.push_back(p);
            normal_sum.push_back({0.0, 0.0, 0.0});
            cells[home].push_back(found);
        }
        welded_of[vertex] = found;
        if (vertex < vertex_normals.size()) {
            const auto normal = vertex_normals[vertex];
            normal_sum[found].x += normal.x; normal_sum[found].y += normal.y; normal_sum[found].z += normal.z;
        }
    }
    const std::size_t count = position.size();
    output.transfer.welded = count;
    std::vector<std::array<std::uint32_t, 3>> welded_faces;
    welded_faces.reserve(faces.size());
    std::vector<std::int32_t> welded_face_of(faces.size(), -1);
    std::vector<std::vector<std::uint32_t>> faces_at(count);
    for (std::size_t index = 0; index < faces.size(); ++index) {
        const auto & face = faces[index];
        if (face[0] >= n || face[1] >= n || face[2] >= n) continue;
        const std::array<std::uint32_t, 3> value{welded_of[face[0]], welded_of[face[1]], welded_of[face[2]]};
        if (value[0] == value[1] || value[1] == value[2] || value[2] == value[0]) continue;
        welded_face_of[index] = static_cast<std::int32_t>(welded_faces.size());
        for (const auto corner : value) faces_at[corner].push_back(static_cast<std::uint32_t>(welded_faces.size()));
        welded_faces.push_back(value);
    }
    // One-ring adjacency of the welded mesh.
    std::vector<std::vector<std::uint32_t>> ring(count);
    for (const auto & face : welded_faces)
        for (std::size_t k = 0; k < 3U; ++k) {
            ring[face[k]].push_back(face[(k + 1U) % 3U]);
            ring[face[k]].push_back(face[(k + 2U) % 3U]);
        }
    for (auto & list : ring) {
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
    }
    std::vector<dvec3> normal(count);
    for (std::size_t vertex = 0; vertex < count; ++vertex) {
        const auto value = normal_sum[vertex];
        const double magnitude = std::sqrt(dot(value, value));
        normal[vertex] = magnitude > 1.0e-12 ? dvec3{value.x / magnitude, value.y / magnitude, value.z / magnitude}
                                             : dvec3{0.0, 0.0, 0.0};
    }
    // A sample is on the vertex's own surface when the surface path to it is
    // not much longer than the straight line: geodesic <= ratio * euclidean +
    // half an edge. Another finger, the other leg or the far side of a thin
    // part are close in space but far along the surface. The geodesic is
    // Dijkstra over welded edges, then the last step to the sample's face.
    std::vector<double> edge_mean(count, 0.0);
    for (std::size_t vertex = 0; vertex < count; ++vertex) {
        for (const auto other : ring[vertex]) edge_mean[vertex] += static_cast<double>(distance(position[vertex], position[other]));
        if (!ring[vertex].empty()) edge_mean[vertex] /= static_cast<double>(ring[vertex].size());
    }
    std::vector<double> geodesic(count, std::numeric_limits<double>::infinity());
    std::vector<std::uint32_t> visited;
    const auto walk = [&](std::uint32_t centre, double radius) {
        for (const auto v : visited) geodesic[v] = std::numeric_limits<double>::infinity();
        visited.assign(1U, centre);
        geodesic[centre] = 0.0;
        using item = std::pair<double, std::uint32_t>;
        std::priority_queue<item, std::vector<item>, std::greater<>> pending;
        pending.emplace(0.0, centre);
        while (!pending.empty()) {
            const auto [d, at] = pending.top();
            pending.pop();
            if (d != geodesic[at]) continue;
            for (const auto next : ring[at]) {
                const double candidate = d + static_cast<double>(distance(position[at], position[next]));
                if (candidate > radius || candidate >= geodesic[next]) continue;
                if (!std::isfinite(geodesic[next])) visited.push_back(next);
                geodesic[next] = candidate;
                pending.emplace(candidate, next);
            }
        }
    };
    const auto on_surface = [&](std::uint32_t centre, std::uint32_t sample, double straight) {
        if (sample >= sampled_faces.size() || sampled_faces[sample] >= welded_face_of.size()) return false;
        const auto f = welded_face_of[sampled_faces[sample]];
        if (f < 0) return false;
        const auto p = sampled_points[sample];
        double along = std::numeric_limits<double>::infinity();
        for (const auto corner : welded_faces[static_cast<std::size_t>(f)])
            if (std::isfinite(geodesic[corner]))
                along = std::min(along, geodesic[corner] + static_cast<double>(distance(position[corner], p)));
        return along <= static_cast<double>(options.geodesic_ratio) * straight + 0.5 * edge_mean[centre];
    };

    // 2. Match. `raw` keeps the upstream eight-neighbour interpolation as the
    // starting point of the solve and the fallback for parts with no match.
    std::vector<dvec3> cloud;
    cloud.reserve(sampled_points.size());
    for (const auto value : sampled_points) cloud.push_back({value.x, value.y, value.z});
    const scipy_point_tree tree{std::move(cloud)};
    const double reach = static_cast<double>(options.distance_fraction) * diagonal;
    const double facing = std::cos(static_cast<double>(options.max_normal_degrees) * 3.14159265358979323846 / 180.0);
    std::vector<float> raw(count * joint_count, 0.0F), robust(count * joint_count, 0.0F);
    std::vector<std::vector<std::pair<float, std::uint32_t>>> found(count);
    for (std::size_t vertex = 0; vertex < count; ++vertex) {
        const auto p = position[vertex];
        found[vertex] = tree.nearest({p.x, p.y, p.z}, std::max(options.candidates, options.interpolated));
    }
    // Facing is judged against the faces around the vertex, which share the
    // winding the samples' face normals come from. An averaged vertex normal
    // turns 45 degrees away from both sides of a hard edge and would reject
    // every sample on a low-poly or hard-surface model.
    std::vector<std::vector<dvec3>> incident(count);
    for (const auto & f : welded_faces) {
        const auto a = position[f[0]], b = position[f[1]], c = position[f[2]];
        const dvec3 value = cross({static_cast<double>(b.x - a.x), static_cast<double>(b.y - a.y), static_cast<double>(b.z - a.z)},
                                  {static_cast<double>(c.x - a.x), static_cast<double>(c.y - a.y), static_cast<double>(c.z - a.z)});
        const double magnitude = std::sqrt(dot(value, value));
        if (!(magnitude > 0.0)) continue;
        const dvec3 unit{value.x / magnitude, value.y / magnitude, value.z / magnitude};
        for (const auto corner : f) incident[corner].push_back(unit);
    }
    const auto faces_toward = [&](std::size_t vertex, vec3 s, double limit) {
        if (incident[vertex].empty())
            return normal[vertex].x * s.x + normal[vertex].y * s.y + normal[vertex].z * s.z >= limit;
        for (const auto & f : incident[vertex])
            if (f.x * s.x + f.y * s.y + f.z * s.z >= limit) return true;
        return false;
    };
    std::vector<char> known(count, 0);
    const auto interpolate = [&](std::span<const std::pair<float, std::uint32_t>> picks, float * row) {
        double total = 0.0;
        std::array<double, 64> inverse{};
        const std::size_t used = std::min<std::size_t>(picks.size(), inverse.size());
        for (std::size_t index = 0; index < used; ++index) {
            inverse[index] = 1.0 / (std::sqrt(static_cast<double>(picks[index].first)) + 1.0e-8);
            total += inverse[index];
        }
        for (std::size_t joint = 0; joint < joint_count; ++joint) {
            double value = 0.0;
            for (std::size_t index = 0; index < used; ++index)
                value += inverse[index] * static_cast<double>(dense_joint_weights[joint][picks[index].second]);
            row[joint] = static_cast<float>(value / total);
        }
    };
    for (std::size_t vertex = 0; vertex < count; ++vertex) {
        const auto & candidates = found[vertex];
        interpolate(std::span{candidates.data(), std::min(options.interpolated, candidates.size())},
                    raw.data() + vertex * joint_count);
        double farthest = 0.0;
        for (const auto & entry : candidates)
            if (static_cast<double>(entry.first) <= reach * reach) farthest = std::sqrt(static_cast<double>(entry.first));
        walk(static_cast<std::uint32_t>(vertex),
             static_cast<double>(options.geodesic_ratio) * farthest + 0.5 * edge_mean[vertex]);
        std::vector<std::pair<float, std::uint32_t>> accepted;
        for (const auto & entry : candidates) {
            if (accepted.size() >= options.interpolated) break;
            if (static_cast<double>(entry.first) > reach * reach) break;
            if (entry.second >= sampled_normals.size() ||
                !on_surface(static_cast<std::uint32_t>(vertex), entry.second, std::sqrt(static_cast<double>(entry.first)))) continue;
            if (!faces_toward(vertex, sampled_normals[entry.second], facing)) continue;
            accepted.push_back(entry);
        }
        if (accepted.empty()) continue;
        interpolate(accepted, robust.data() + vertex * joint_count);
        known[vertex] = 1;
        ++output.transfer.matched;
    }

    // 3. Outliers: a matched vertex unlike its matched neighbours.
    std::vector<std::uint32_t> outliers;
    std::vector<double> mean(joint_count);
    for (std::size_t vertex = 0; vertex < count; ++vertex) {
        if (!known[vertex]) continue;
        std::size_t neighbours = 0U;
        std::fill(mean.begin(), mean.end(), 0.0);
        for (const auto other : ring[vertex]) {
            if (!known[other]) continue;
            ++neighbours;
            for (std::size_t joint = 0; joint < joint_count; ++joint)
                mean[joint] += robust[other * joint_count + joint];
        }
        if (neighbours < 2U) continue;
        double difference = 0.0;
        for (std::size_t joint = 0; joint < joint_count; ++joint)
            difference += std::abs(mean[joint] / static_cast<double>(neighbours) - robust[vertex * joint_count + joint]);
        if (difference > static_cast<double>(options.outlier_l1)) outliers.push_back(static_cast<std::uint32_t>(vertex));
    }
    for (const auto vertex : outliers) known[vertex] = 0;
    output.transfer.outliers = outliers.size();

    // 4. Mollified cotan Laplacian (Sharp and Crane 2020, section 4.5) and
    // lumped mass on the welded mesh.
    std::vector<std::array<double, 3>> lengths(welded_faces.size());
    double length_sum = 0.0;
    for (std::size_t face = 0; face < welded_faces.size(); ++face) {
        const auto & f = welded_faces[face];
        for (std::size_t k = 0; k < 3U; ++k) {
            lengths[face][k] = static_cast<double>(distance(position[f[(k + 1U) % 3U]], position[f[(k + 2U) % 3U]]));
            length_sum += lengths[face][k];
        }
    }
    const double epsilon = welded_faces.empty() ? 0.0 :
        1.0e-5 * length_sum / static_cast<double>(welded_faces.size() * 3U);
    double delta = 0.0;
    for (const auto & l : lengths)
        for (std::size_t k = 0; k < 3U; ++k)
            delta = std::max(delta, epsilon - (l[(k + 1U) % 3U] + l[(k + 2U) % 3U] - l[k]));
    std::vector<double> mass(count, 0.0);
    std::unordered_map<std::uint64_t, double> cotangent;
    cotangent.reserve(welded_faces.size() * 3U);
    for (std::size_t face = 0; face < welded_faces.size(); ++face) {
        const auto & f = welded_faces[face];
        const double a = lengths[face][0] + delta, b = lengths[face][1] + delta, c = lengths[face][2] + delta;
        const double product = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c);
        const double area = 0.25 * std::sqrt(std::max(product, 0.0));
        if (!(area > 0.0)) continue;
        const std::array<double, 3> l{a, b, c};
        for (std::size_t k = 0; k < 3U; ++k) {
            // Angle at corner k faces edge k (between the other two corners).
            const double opposite = l[k], left = l[(k + 1U) % 3U], right = l[(k + 2U) % 3U];
            const double cot = (left * left + right * right - opposite * opposite) / (4.0 * area);
            auto i = f[(k + 1U) % 3U], j = f[(k + 2U) % 3U];
            if (i > j) std::swap(i, j);
            cotangent[(static_cast<std::uint64_t>(i) << 32U) | j] += 0.5 * cot;
            mass[f[k]] += area / 3.0;
        }
    }
    std::vector<std::vector<std::pair<std::uint32_t, double>>> laplacian(count);
    std::vector<double> diagonal_entry(count, 0.0);
    for (const auto & [key, weight] : cotangent) {
        const auto i = static_cast<std::uint32_t>(key >> 32U);
        const auto j = static_cast<std::uint32_t>(key & 0xffffffffU);
        laplacian[i].emplace_back(j, weight);
        laplacian[j].emplace_back(i, weight);
        diagonal_entry[i] -= weight;
        diagonal_entry[j] -= weight;
    }

    // Unknowns with no known one-ring neighbour anywhere on their unknown
    // island cannot be inpainted: keep plain interpolation there.
    std::vector<std::int32_t> slot(count, -1);
    {
        std::vector<char> seen(count, 0);
        std::vector<std::uint32_t> island, stack;
        for (std::size_t start = 0; start < count; ++start) {
            if (known[start] || seen[start]) continue;
            island.clear();
            stack.assign(1U, static_cast<std::uint32_t>(start));
            seen[start] = 1;
            bool anchored = false;
            while (!stack.empty()) {
                const auto vertex = stack.back();
                stack.pop_back();
                island.push_back(vertex);
                for (const auto & [other, weight] : laplacian[vertex]) {
                    (void) weight;
                    if (known[other]) { anchored = true; continue; }
                    if (!seen[other]) { seen[other] = 1; stack.push_back(other); }
                }
            }
            bool solvable = anchored;
            for (const auto vertex : island) solvable = solvable && mass[vertex] > 0.0;
            if (!solvable) {
                for (const auto vertex : island) {
                    std::copy_n(raw.data() + vertex * joint_count, joint_count, robust.data() + vertex * joint_count);
                    known[vertex] = 1;
                }
                output.transfer.fallback += island.size();
            }
        }
    }
    std::vector<std::uint32_t> unknown;
    for (std::size_t vertex = 0; vertex < count; ++vertex)
        if (!known[vertex]) {
            slot[vertex] = static_cast<std::int32_t>(unknown.size());
            unknown.push_back(static_cast<std::uint32_t>(vertex));
        }
    output.transfer.inpainted = unknown.size();

    if (!unknown.empty()) {
        // Rows of Q = -L + L M^-1 L for the unknowns, split into the unknown
        // block A and the known block B.
        const std::size_t u = unknown.size();
        std::vector<std::size_t> row_start(u + 1U, 0U);
        std::vector<std::uint32_t> a_column;
        std::vector<double> a_value;
        std::vector<std::vector<std::pair<std::uint32_t, double>>> b_rows(u);
        std::vector<double> a_diagonal(u, 0.0);
        std::vector<std::pair<std::uint32_t, double>> row;
        for (std::size_t r = 0; r < u; ++r) {
            const auto i = unknown[r];
            row.clear();
            const auto visit = [&](std::uint32_t j, double l_ij) {
                row.emplace_back(j, -l_ij);
                const double inverse_mass = mass[j] > 0.0 ? 1.0 / mass[j] : 0.0;
                row.emplace_back(j, l_ij * diagonal_entry[j] * inverse_mass);
                for (const auto & [k, l_jk] : laplacian[j]) row.emplace_back(k, l_ij * l_jk * inverse_mass);
            };
            visit(i, diagonal_entry[i]);
            for (const auto & [j, l_ij] : laplacian[i]) visit(j, l_ij);
            std::sort(row.begin(), row.end(), [](const auto & x, const auto & y) { return x.first < y.first; });
            for (std::size_t index = 0; index < row.size();) {
                const auto column = row[index].first;
                double value = 0.0;
                while (index < row.size() && row[index].first == column) value += row[index++].second;
                if (value == 0.0) continue;
                if (slot[column] >= 0) {
                    a_column.push_back(static_cast<std::uint32_t>(slot[column]));
                    a_value.push_back(value);
                    if (column == i) a_diagonal[r] = value;
                } else {
                    b_rows[r].emplace_back(column, value);
                }
            }
            row_start[r + 1U] = a_column.size();
        }
        const auto multiply = [&](const std::vector<double> & x, std::vector<double> & y) {
            for (std::size_t r = 0; r < u; ++r) {
                double value = 0.0;
                for (std::size_t e = row_start[r]; e < row_start[r + 1U]; ++e) value += a_value[e] * x[a_column[e]];
                y[r] = value;
            }
        };
        std::vector<double> b(u), x(u), residual(u), z(u), direction(u), product(u);
        for (std::size_t joint = 0; joint < joint_count; ++joint) {
            double b_norm = 0.0;
            for (std::size_t r = 0; r < u; ++r) {
                double value = 0.0;
                for (const auto & [column, q] : b_rows[r])
                    value -= q * static_cast<double>(robust[column * joint_count + joint]);
                b[r] = value;
                b_norm += value * value;
                x[r] = static_cast<double>(raw[unknown[r] * joint_count + joint]);
            }
            b_norm = std::sqrt(b_norm);
            if (b_norm <= 1.0e-12) {
                // No known neighbour carries this joint: it stays zero here.
                for (std::size_t r = 0; r < u; ++r) robust[unknown[r] * joint_count + joint] = 0.0F;
                continue;
            }
            ++output.transfer.solved_joints;
            // Jacobi-preconditioned conjugate gradients, warm-started from the
            // plain interpolation.
            multiply(x, product);
            double rz = 0.0;
            for (std::size_t r = 0; r < u; ++r) {
                residual[r] = b[r] - product[r];
                z[r] = a_diagonal[r] > 0.0 ? residual[r] / a_diagonal[r] : residual[r];
                direction[r] = z[r];
                rz += residual[r] * z[r];
            }
            std::size_t iteration = 0U;
            double relative = 0.0;
            for (; iteration < options.max_iterations; ++iteration) {
                double r_norm = 0.0;
                for (const auto value : residual) r_norm += value * value;
                relative = std::sqrt(r_norm) / b_norm;
                if (relative <= options.tolerance) break;
                multiply(direction, product);
                double curvature = 0.0;
                for (std::size_t r = 0; r < u; ++r) curvature += direction[r] * product[r];
                if (!(curvature > 0.0)) break;
                const double alpha = rz / curvature;
                double next_rz = 0.0;
                for (std::size_t r = 0; r < u; ++r) {
                    x[r] += alpha * direction[r];
                    residual[r] -= alpha * product[r];
                    z[r] = a_diagonal[r] > 0.0 ? residual[r] / a_diagonal[r] : residual[r];
                    next_rz += residual[r] * z[r];
                }
                const double beta = next_rz / rz;
                rz = next_rz;
                for (std::size_t r = 0; r < u; ++r) direction[r] = z[r] + beta * direction[r];
            }
            output.transfer.iterations = std::max(output.transfer.iterations, iteration);
            output.transfer.residual = std::max(output.transfer.residual, static_cast<float>(relative));
            for (std::size_t r = 0; r < u; ++r)
                robust[unknown[r] * joint_count + joint] = static_cast<float>(x[r]);
        }
    }

    // 5. Clamp, normalise, keep four; seam twins share their welded row.
    for (std::size_t vertex = 0; vertex < count; ++vertex) {
        float * values = robust.data() + vertex * joint_count;
        double sum = 0.0;
        for (std::size_t joint = 0; joint < joint_count; ++joint) {
            values[joint] = std::isfinite(values[joint]) ? std::clamp(values[joint], 0.0F, 1.0F) : 0.0F;
            sum += values[joint];
        }
        if (sum <= 1.0e-12) {
            std::copy_n(raw.data() + vertex * joint_count, joint_count, values);
            sum = 0.0;
            for (std::size_t joint = 0; joint < joint_count; ++joint) sum += values[joint];
        }
        if (sum > 1.0e-12)
            for (std::size_t joint = 0; joint < joint_count; ++joint)
                values[joint] = static_cast<float>(values[joint] / sum);
    }

    // 5b. Smoothing. A vertex whose row differs sharply from its ring tears
    // away from it when the joints move apart; each pass moves every row
    // halfway to its ring's mean. Parts stay separate (rings never cross).
    if (options.smooth_iterations > 0U) {
        std::vector<float> next(robust.size());
        for (std::size_t pass = 0; pass < options.smooth_iterations; ++pass) {
            for (std::size_t vertex = 0; vertex < count; ++vertex) {
                const float * self = robust.data() + vertex * joint_count;
                float * out = next.data() + vertex * joint_count;
                if (ring[vertex].empty()) { std::copy_n(self, joint_count, out); continue; }
                const float k = 1.0F / static_cast<float>(ring[vertex].size());
                for (std::size_t joint = 0; joint < joint_count; ++joint) {
                    float mean = 0.0F;
                    for (const auto other : ring[vertex]) mean += robust[other * joint_count + joint];
                    out[joint] = 0.5F * self[joint] + 0.5F * mean * k;
                }
            }
            robust.swap(next);
        }
    }

    // 6. Part support. Each separate mesh part (a shoe, a glove, a hair card,
    // or the whole body when it is one piece) keeps the joints reachable in
    // the skeleton from its strongest joint through joints that carry at
    // least `support_share` of the part's weight. A right shoe whose decoded
    // weights pick up the left foot drops it: the left foot connects to the
    // right foot only through hips and legs that carry nothing on that shoe.
    // A connected body keeps everything, because its legs, hips and spine all
    // carry weight (one empty joint in a row, such as a bare root, may be
    // crossed). Parts never share an edge, so the cut cannot tear a seam.
    if (options.support_share > 0.0F && target.parents.size() == joint_count) {
        std::vector<std::vector<std::uint32_t>> bones(joint_count);
        for (std::size_t joint = 0; joint < joint_count; ++joint) {
            const auto parent = target.parents[joint];
            if (parent < 0 || static_cast<std::size_t>(parent) >= joint_count) continue;
            bones[joint].push_back(static_cast<std::uint32_t>(parent));
            bones[static_cast<std::size_t>(parent)].push_back(static_cast<std::uint32_t>(joint));
        }
        std::vector<char> seen(count, 0), keep(joint_count, 0);
        std::vector<std::uint8_t> gap(joint_count, 255U);
        std::vector<std::uint32_t> part, stack, reach_queue;
        std::vector<double> share(joint_count);
        for (std::size_t start = 0; start < count; ++start) {
            if (seen[start]) continue;
            part.clear();
            stack.assign(1U, static_cast<std::uint32_t>(start));
            seen[start] = 1;
            while (!stack.empty()) {
                const auto at = stack.back();
                stack.pop_back();
                part.push_back(at);
                for (const auto next : ring[at])
                    if (!seen[next]) { seen[next] = 1; stack.push_back(next); }
            }
            std::fill(share.begin(), share.end(), 0.0);
            double total = 0.0;
            // Shares count vertices, not area: a fingertip is a sliver of a
            // body's area but a fair share of its vertices.
            for (const auto v : part) {
                for (std::size_t joint = 0; joint < joint_count; ++joint)
                    share[joint] += robust[v * joint_count + joint];
                total += 1.0;
            }
            if (!(total > 0.0)) continue;
            const auto strongest = static_cast<std::size_t>(std::max_element(share.begin(), share.end()) - share.begin());
            // Breadth-first over the skeleton; `gap` counts the joints in a
            // row below the share. One such joint may be crossed (a root or
            // pelvis joint that carries nothing), two in a row may not.
            std::fill(keep.begin(), keep.end(), 0);
            std::fill(gap.begin(), gap.end(), 255U);
            const double floor = static_cast<double>(options.support_share) * total;
            keep[strongest] = 1;
            gap[strongest] = 0U;
            reach_queue.assign(1U, static_cast<std::uint32_t>(strongest));
            for (std::size_t head = 0; head < reach_queue.size(); ++head) {
                const auto at = reach_queue[head];
                for (const auto next : bones[at]) {
                    const std::uint8_t run = share[next] >= floor ? 0U : static_cast<std::uint8_t>(gap[at] + 1U);
                    if (run > 1U || run >= gap[next]) continue;
                    gap[next] = run;
                    keep[next] = 1;
                    reach_queue.push_back(next);
                }
            }
            bool dropped = false;
            for (std::size_t joint = 0; joint < joint_count; ++joint)
                dropped = dropped || (!keep[joint] && share[joint] > 0.0);
            if (!dropped) continue;
            ++output.transfer.trimmed_parts;
            for (const auto v : part) {
                float * values = robust.data() + v * joint_count;
                double kept = 0.0;
                for (std::size_t joint = 0; joint < joint_count; ++joint) if (keep[joint]) kept += values[joint];
                if (!(kept > 1.0e-6)) continue;
                for (std::size_t joint = 0; joint < joint_count; ++joint)
                    values[joint] = keep[joint] ? static_cast<float>(values[joint] / kept) : 0.0F;
            }
        }
    }

    for (std::size_t vertex = 0; vertex < n; ++vertex)
        keep_four(robust.data() + static_cast<std::size_t>(welded_of[vertex]) * joint_count, joint_count,
                  output.joints[vertex], output.weights[vertex]);
    return output;
}

std::function<void(const transfer_inputs &)> & transfer_capture() {
    static std::function<void(const transfer_inputs &)> value;
    return value;
}

std::vector<std::byte> serialize_transfer_inputs(const transfer_inputs & value) {
    std::vector<std::byte> out;
    const auto put = [&](const void * data, std::size_t bytes) {
        const auto * first = static_cast<const std::byte *>(data);
        out.insert(out.end(), first, first + bytes);
    };
    const auto put_u32 = [&](std::size_t v) { const auto x = static_cast<std::uint32_t>(v); put(&x, 4U); };
    put("STKT", 4U);
    put_u32(1U);
    put_u32(value.normalized_vertices.size());
    put_u32(value.faces.size());
    put_u32(value.sampled_points.size());
    put_u32(value.dense_joint_weights.size());
    put(value.normalized_vertices.data(), value.normalized_vertices.size_bytes());
    put(value.vertex_normals.data(), value.vertex_normals.size_bytes());
    put(value.faces.data(), value.faces.size_bytes());
    put(value.sampled_points.data(), value.sampled_points.size_bytes());
    put(value.sampled_normals.data(), value.sampled_normals.size_bytes());
    put(value.sampled_faces.data(), value.sampled_faces.size_bytes());
    for (const auto & column : value.dense_joint_weights) put(column.data(), column.size() * sizeof(float));
    return out;
}

bool deserialize_transfer_inputs(std::span<const std::byte> bytes, transfer_inputs_storage & out) {
    std::size_t at = 0U;
    const auto take = [&](void * data, std::size_t size) {
        if (at + size > bytes.size()) return false;
        std::memcpy(data, bytes.data() + at, size);
        at += size;
        return true;
    };
    char magic[4]{};
    std::uint32_t version = 0, n = 0, f = 0, s = 0, j = 0;
    if (!take(magic, 4U) || std::memcmp(magic, "STKT", 4U) != 0) return false;
    if (!take(&version, 4U) || version != 1U || !take(&n, 4U) || !take(&f, 4U) || !take(&s, 4U) || !take(&j, 4U)) return false;
    out.normalized_vertices.resize(n); out.vertex_normals.resize(n); out.faces.resize(f);
    out.sampled_points.resize(s); out.sampled_normals.resize(s); out.sampled_faces.resize(s);
    out.dense_joint_weights.assign(j, std::vector<float>(s));
    bool ok = take(out.normalized_vertices.data(), n * sizeof(vec3)) && take(out.vertex_normals.data(), n * sizeof(vec3)) &&
              take(out.faces.data(), f * sizeof(triangle)) && take(out.sampled_points.data(), s * sizeof(vec3)) &&
              take(out.sampled_normals.data(), s * sizeof(vec3)) && take(out.sampled_faces.data(), s * sizeof(std::uint32_t));
    for (auto & column : out.dense_joint_weights) ok = ok && take(column.data(), s * sizeof(float));
    return ok && at == bytes.size();
}

skin integrate_learned_binding(
    const skeleton & target,
    std::span<const vec3> normalized_vertices,
    std::span<const vec3> sampled_points,
    std::span<const std::vector<float>> dense_joint_weights,
    binding_trace * trace) {
    return integrate(target, normalized_vertices, sampled_points, dense_joint_weights, {}, trace);
}

skin integrate_postprocessed_binding(
    const skeleton & target,
    std::span<const vec3> normalized_vertices,
    std::span<const triangle> faces,
    std::span<const vec3> normalized_joints,
    std::span<const vec3> sampled_points,
    std::span<const std::vector<float>> dense_joint_weights,
    binding_trace * trace) {
    const auto locality = surface_weights(normalized_vertices, faces, normalized_joints);
    auto output=integrate(target, normalized_vertices, sampled_points, dense_joint_weights, locality.weights, trace);
    if (trace != nullptr) {
        trace->voxel_coordinates=locality.voxel_coordinates;
        trace->surface_seeds=locality.seeds;
        trace->surface_distances=locality.distances;
        trace->surface_graph_edges=locality.graph_edges;
        trace->maximum_surface_distance=locality.maximum_distance;
        trace->surface_graph_sources=locality.graph_sources;
        trace->surface_graph_targets=locality.graph_targets;
        trace->surface_graph_weights=locality.graph_weights;
    }
    return output;
}

skin integrate_postprocessed_binding_precise(
    const skeleton & target,
    std::span<const vec3> normalized_vertices,
    std::span<const precise_vec3> precise_vertices,
    std::span<const triangle> faces,
    std::span<const vec3> normalized_joints,
    std::span<const precise_vec3> precise_joints,
    std::span<const vec3> sampled_points,
    std::span<const std::vector<float>> dense_joint_weights,
    binding_trace * trace) {
    const auto locality = surface_weights(
        normalized_vertices, faces, normalized_joints, precise_vertices, precise_joints);
    auto output=integrate(target, normalized_vertices, sampled_points, dense_joint_weights, locality.weights, trace);
    if (trace != nullptr) {
        trace->voxel_coordinates=locality.voxel_coordinates;
        trace->surface_seeds=locality.seeds;
        trace->surface_distances=locality.distances;
        trace->surface_graph_edges=locality.graph_edges;
        trace->maximum_surface_distance=locality.maximum_distance;
        trace->surface_graph_sources=locality.graph_sources;
        trace->surface_graph_targets=locality.graph_targets;
        trace->surface_graph_weights=locality.graph_weights;
    }
    return output;
}

} // namespace skintokens::detail
