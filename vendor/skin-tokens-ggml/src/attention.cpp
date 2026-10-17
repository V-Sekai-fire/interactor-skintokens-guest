#include "internal.hpp"

#include <algorithm>
#include <cmath>

namespace skintokens::detail {

// Every tensor here stays under WebGPU's default 128 MiB storage binding. The full-width
// [width, data] K/V projection and the [data, queries, heads] score tensor of a 54,000-point
// cross attention do not, so they are built one head and one block of queries at a time.
// Each output element is the same dot product, softmax row and weighted sum as before.
namespace {
constexpr std::size_t score_budget = std::size_t{64} << 20U;
} // namespace

ggml_tensor * rows_linear(ggml_context * context, ggml_tensor * weight, ggml_tensor * bias,
                          std::int64_t first_row, std::int64_t rows, ggml_tensor * input) {
    auto * slice = ggml_view_2d(context, weight, weight->ne[0], rows, weight->nb[1],
                                static_cast<std::size_t>(first_row) * weight->nb[1]);
    auto * output = ggml_mul_mat(context, slice, input);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    if (bias != nullptr) {
        auto * part = ggml_view_1d(context, bias, rows, static_cast<std::size_t>(first_row) * bias->nb[0]);
        if (part->type != output->type) part = ggml_cast(context, part, output->type);
        output = ggml_add(context, output, ggml_repeat(context, part, output));
    }
    return output;
}

ggml_tensor * attend_by_head(ggml_context * context, ggml_tensor * q, std::int64_t heads,
                             std::int64_t head_dim, std::int64_t data_count,
                             const std::function<ggml_tensor *(std::int64_t head)> & head_kv) {
    const std::int64_t query_count = q->ne[1];
    const std::int64_t block = std::max<std::int64_t>(1,
        std::min<std::int64_t>(query_count,
            static_cast<std::int64_t>(score_budget / (static_cast<std::size_t>(data_count) * sizeof(float)))));
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    ggml_tensor * joined = nullptr;
    for (std::int64_t head = 0; head < heads; ++head) {
        auto * kv = head_kv(head); // [2 * head_dim, data]: K then V
        auto * k = ggml_cont(context, ggml_view_2d(context, kv, head_dim, data_count, kv->nb[1], 0));
        auto * v = ggml_view_2d(context, kv, head_dim, data_count, kv->nb[1],
                                static_cast<std::size_t>(head_dim) * kv->nb[0]);
        auto * v_t = ggml_cont(context, ggml_transpose(context, v));
        ggml_tensor * head_out = nullptr;
        for (std::int64_t first = 0; first < query_count; first += block) {
            const std::int64_t count = std::min(block, query_count - first);
            auto * q_part = ggml_cont(context, ggml_view_2d(context, q, head_dim, count, q->nb[1],
                static_cast<std::size_t>(head * head_dim) * q->nb[0] + static_cast<std::size_t>(first) * q->nb[1]));
            auto * scores = ggml_mul_mat(context, k, q_part);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            auto * probability = ggml_soft_max(context, ggml_scale(context, scores, scale));
            auto * out = ggml_mul_mat(context, v_t, probability);
            ggml_mul_mat_set_prec(out, GGML_PREC_F32);
            head_out = head_out == nullptr ? out : ggml_concat(context, head_out, out, 1);
        }
        joined = joined == nullptr ? head_out : ggml_concat(context, joined, head_out, 0);
    }
    return joined;
}

} // namespace skintokens::detail
