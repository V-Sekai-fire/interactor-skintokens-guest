#pragma once

#include <skintokens/skintokens.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace skintokens::detail {

struct precise_vec3 { double x, y, z; };

struct binding_trace {
    std::vector<std::uint32_t> neighbor_indices;
    std::vector<float> interpolation_weights;
    std::vector<float> surface_weights;
    // Final dense matrix after interpolation and optional voxel_skin
    // multiplication, immediately before top-four selection.
    std::vector<float> final_dense_weights;
    std::vector<std::int32_t> voxel_coordinates;
    std::vector<std::uint32_t> surface_seeds;
    std::vector<float> surface_distances; // [joint, vertex], after fallback/inf replacement
    std::size_t surface_graph_edges = 0U;
    float maximum_surface_distance = 0.0F;
    std::vector<std::uint32_t> surface_graph_sources;
    std::vector<std::uint32_t> surface_graph_targets;
    std::vector<float> surface_graph_weights;
};

// Reproduce upstream Asset.from_data() followed by its Blender export: each
// source vertex receives inverse-distance interpolation from eight sampled
// points, then the greatest four learned joint weights are retained and
// normalized. This intentionally contains no geometric bone-distance gate.
skin integrate_learned_binding(
    const skeleton & target,
    std::span<const vec3> normalized_vertices,
    std::span<const vec3> sampled_points,
    std::span<const std::vector<float>> dense_joint_weights,
    binding_trace * trace = nullptr);

// Reproduce the optional postprocess used by the upstream demo: learned
// weights remain the signal, but are multiplied by a normalized surface-
// geodesic weight for each joint before the final top-four export. The
// upstream call does not pass parents to voxel_skin, so joints (rather than
// subdivided bones) seed the mesh graph.
skin integrate_postprocessed_binding(
    const skeleton & target,
    std::span<const vec3> normalized_vertices,
    std::span<const triangle> faces,
    std::span<const vec3> normalized_joints,
    std::span<const vec3> sampled_points,
    std::span<const std::vector<float>> dense_joint_weights,
    binding_trace * trace = nullptr);

skin integrate_postprocessed_binding_precise(
    const skeleton & target,
    std::span<const vec3> normalized_vertices,
    std::span<const precise_vec3> precise_vertices,
    std::span<const triangle> faces,
    std::span<const vec3> normalized_joints,
    std::span<const precise_vec3> precise_joints,
    std::span<const vec3> sampled_points,
    std::span<const std::vector<float>> dense_joint_weights,
    binding_trace * trace = nullptr);

struct robust_transfer_options {
    // A sample further than this fraction of the bounding-box diagonal is
    // never a match (the paper's default).
    float distance_fraction = 0.05F;
    // A sample counts only when the surface path to it is at most this many
    // times the straight line (plus half an edge), and its face turns at most
    // this angle from one of the vertex's own faces (90 = reject only
    // back-facing samples; tighter angles starve curved fingertips).
    float geodesic_ratio = 2.0F;
    float max_normal_degrees = 90.0F;
    // A matched vertex whose dense weights differ from its one-ring mean by
    // more than this L1 distance is an outlier and is inpainted too.
    float outlier_l1 = 1.2F;
    // Per mesh part, keep joints reachable in the skeleton from the part's
    // strongest joint without crossing two joints in a row that hold less
    // than this share of the part's vertex weight (0 = off). Small enough
    // that a fingertip on a whole-body mesh (~0.5%) still counts.
    float support_share = 0.001F;
    // Passes moving each row halfway to its one-ring mean (step 5b).
    std::size_t smooth_iterations = 15U;
    std::size_t candidates = 16U;
    std::size_t interpolated = 8U;
    std::size_t max_iterations = 4000U;
    double tolerance = 1.0e-7;
};

// Robust Skin Weights Transfer via Weight Inpainting (Abdrashitov, Raichstat,
// Monsen, Hill; SIGGRAPH Asia 2023) applied to the SkinVAE point cloud:
//   1. weld coincident vertices so UV and normal seams share one weight row;
//   2. match each welded vertex only to samples that are close, do not face
//      away and are about as near along the surface as in space (so not on
//      another finger, the other leg or the far side of a thin part), and
//      interpolate from those;
//   3. mark matched vertices that disagree with their neighbours as outliers;
//   4. solve the unmatched and outlier vertices from the matched ones by
//      minimising tr(W^T Q W), Q = -L + L M^-1 L, with L the cotan Laplacian
//      on edge lengths mollified as in Sharp and Crane 2020, so slivers and
//      degenerate triangles cannot produce infinite cotangents;
//   5. clamp to [0, 1] and normalise;
//   6. per separate mesh part, drop joints the skeleton connects to the part's
//      strongest joint only through joints that carry no weight on it;
//   7. keep the four greatest and normalise. (Cutting joints
//      far apart in the skeleton was tried and dropped: on a connected mesh the
//      cut is a weight discontinuity that tears under motion.)
skin integrate_robust_binding(
    const skeleton & target,
    std::span<const vec3> normalized_vertices,
    std::span<const vec3> vertex_normals,
    std::span<const triangle> faces,
    std::span<const vec3> sampled_points,
    std::span<const vec3> sampled_normals,
    std::span<const std::uint32_t> sampled_faces,
    std::span<const std::vector<float>> dense_joint_weights,
    const robust_transfer_options & options = {});

// Everything the point-cloud-to-vertex transfer reads, so a transfer change
// can be replayed and measured without rerunning the model. Serialised as
// "STKT" u32 version=1, u32 vertices, faces, samples, joints, then
// normalized vertices, vertex normals (f32 x3 each), faces (u32 x3), sample
// points, sample normals (f32 x3), sample faces (u32), dense weights
// (f32, joint-major: joints x samples), all little-endian.
struct transfer_inputs {
    std::span<const vec3> normalized_vertices;
    std::span<const vec3> vertex_normals;
    std::span<const triangle> faces;
    std::span<const vec3> sampled_points;
    std::span<const vec3> sampled_normals;
    std::span<const std::uint32_t> sampled_faces;
    std::span<const std::vector<float>> dense_joint_weights;
};
std::vector<std::byte> serialize_transfer_inputs(const transfer_inputs & value);
struct transfer_inputs_storage {
    std::vector<vec3> normalized_vertices, vertex_normals, sampled_points, sampled_normals;
    std::vector<triangle> faces;
    std::vector<std::uint32_t> sampled_faces;
    std::vector<std::vector<float>> dense_joint_weights;
};
bool deserialize_transfer_inputs(std::span<const std::byte> bytes, transfer_inputs_storage & out);
// When set, decode_binding hands its transfer inputs here before the
// transfer (the guest's trace=1 option; the CLI's SKINTOKENS_DUMP_TRANSFER).
std::function<void(const transfer_inputs &)> & transfer_capture();

} // namespace skintokens::detail
