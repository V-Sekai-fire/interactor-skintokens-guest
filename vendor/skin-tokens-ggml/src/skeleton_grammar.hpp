// TokenRig skeleton token grammar: which token may come next in a skeleton sequence.
// Header-only so tests can check it without the model.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace skintokens::detail {

// The skin codes cover at most 256 joints. A skeleton that reaches it may only end: a beam that
// grew past it could never finish, and every beam doing so failed only at the token limit.
inline constexpr std::size_t max_skeleton_joints = 256U;

enum class skeleton_state {
    expect_bos,
    expect_cls_or_part_or_joint,
    expect_part_or_joint,
    expect_joint_2,
    expect_joint_3,
    expect_branch_or_part_or_joint,
    expect_joint,
};

struct skeleton_parse {
    skeleton_state state = skeleton_state::expect_bos;
    std::size_t joints = 0;
    bool ended = false;
};

inline skeleton_parse parse_skeleton(std::span<const std::int32_t> ids) {
    skeleton_parse output;
    bool branch_half = false;
    for (const auto id : ids) {
        if (output.ended) throw std::runtime_error("tokens follow the skeleton terminator");
        switch (output.state) {
        case skeleton_state::expect_bos:
            if (id != 257) throw std::runtime_error("skeleton token sequence does not start with BOS");
            output.state = skeleton_state::expect_cls_or_part_or_joint;
            break;
        case skeleton_state::expect_cls_or_part_or_joint:
            if (id < 256) output.state = skeleton_state::expect_joint_2;
            else if (id >= 263 && id <= 266) output.state = skeleton_state::expect_part_or_joint;
            else if (id >= 260 && id <= 262) output.state = skeleton_state::expect_joint;
            else throw std::runtime_error("invalid token after skeleton BOS");
            break;
        case skeleton_state::expect_part_or_joint:
            if (id == 258) output.ended = true;
            else if (id < 256) output.state = skeleton_state::expect_joint_2;
            else if (id >= 260 && id <= 262) output.state = skeleton_state::expect_part_or_joint;
            else throw std::runtime_error("invalid skeleton part/joint token");
            break;
        case skeleton_state::expect_joint_2:
            if (id >= 256) throw std::runtime_error("invalid second coordinate token");
            output.state = skeleton_state::expect_joint_3;
            break;
        case skeleton_state::expect_joint_3:
            if (id >= 256) throw std::runtime_error("invalid third coordinate token");
            if (branch_half) branch_half = false;
            else ++output.joints;
            output.state = skeleton_state::expect_branch_or_part_or_joint;
            break;
        case skeleton_state::expect_branch_or_part_or_joint:
            if (id == 258) output.ended = true;
            else if (id == 256) {
                output.state = skeleton_state::expect_joint;
                branch_half = true;
            } else if (id < 256) output.state = skeleton_state::expect_joint_2;
            else if (id >= 260 && id <= 262) output.state = skeleton_state::expect_joint;
            else throw std::runtime_error("invalid token between skeleton joints");
            break;
        case skeleton_state::expect_joint:
            if (id >= 256) throw std::runtime_error("expected skeleton coordinate token");
            output.state = skeleton_state::expect_joint_2;
            break;
        }
    }
    return output;
}

inline std::vector<std::int32_t> allowed_skeleton_tokens(const skeleton_parse &parsed) {
    std::vector<std::int32_t> output;
    const auto coordinates = [&] { for (std::int32_t id = 0; id < 256; ++id) output.push_back(id); };
    const auto parts = [&] { output.insert(output.end(), {260, 261, 262}); };
    switch (parsed.state) {
    case skeleton_state::expect_bos: output.push_back(257); break;
    case skeleton_state::expect_cls_or_part_or_joint:
        output.insert(output.end(), {263, 264, 265, 266});
        parts();
        coordinates();
        break;
    case skeleton_state::expect_part_or_joint:
        if (parsed.joints >= max_skeleton_joints) {
            output.push_back(258);
            break;
        }
        parts();
        coordinates();
        if (parsed.joints != 0U) output.push_back(258);
        break;
    case skeleton_state::expect_joint_2:
    case skeleton_state::expect_joint_3:
    case skeleton_state::expect_joint: coordinates(); break;
    case skeleton_state::expect_branch_or_part_or_joint:
        if (parsed.joints >= max_skeleton_joints) {
            output.push_back(258);
            break;
        }
        coordinates();
        parts();
        output.push_back(256);
        if (parsed.joints != 0U) output.push_back(258);
        break;
    }
    return output;
}

} // namespace skintokens::detail
