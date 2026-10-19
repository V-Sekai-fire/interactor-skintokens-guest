// GLB validation and accessor handling are adapted from animate-any-mesh.cpp's
// independently written parser. This version additionally reads Kimodo node
// animation channels and keeps the API deliberately bounded.
#include "internal.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>

namespace skintokens {
namespace {

using json = nlohmann::json;
constexpr std::uint32_t glb_magic = 0x46546C67U;
constexpr std::uint32_t json_chunk = 0x4E4F534AU;
constexpr std::uint32_t bin_chunk = 0x004E4942U;
constexpr std::size_t max_file = 512U << 20U;
constexpr std::size_t max_json = 16U << 20U;
constexpr std::size_t max_vertices = 4U << 20U;

std::uint32_t u32(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    if constexpr (std::endian::native == std::endian::big) value = std::byteswap(value);
    return value;
}

template<class T> T little(const std::byte * data) {
    T value{};
    std::memcpy(&value, data, sizeof(value));
    if constexpr (std::endian::native == std::endian::big && std::is_integral_v<T>) value = std::byteswap(value);
    return value;
}

struct document {
    json root;
    std::vector<std::byte> binary;
};

result<document> parse_document(std::span<const std::byte> bytes) {
    if (bytes.size() > max_file)
        return std::unexpected(detail::fail(error_code::limit_exceeded, "GLB exceeds 512 MiB safety limit"));
    if (bytes.size() < 20U || u32(bytes, 0) != glb_magic || u32(bytes, 4) != 2U || u32(bytes, 8) != bytes.size())
        return std::unexpected(detail::fail(error_code::invalid_format, "invalid GLB v2 header"));
    const std::size_t json_size = u32(bytes, 12);
    if (u32(bytes, 16) != json_chunk || json_size == 0 || json_size > max_json || json_size > bytes.size() - 20U)
        return std::unexpected(detail::fail(error_code::invalid_format, "invalid GLB JSON chunk"));
    document output;
    try {
        const auto * begin = reinterpret_cast<const char *>(bytes.data() + 20U);
        output.root = json::parse(begin, begin + json_size);
    } catch (const std::exception & exception) {
        return std::unexpected(detail::fail(error_code::invalid_format, std::string{"invalid glTF JSON: "} + exception.what()));
    }
    const std::size_t next = 20U + json_size;
    if (next + 8U <= bytes.size()) {
        const std::size_t binary_size = u32(bytes, next);
        if (u32(bytes, next + 4U) != bin_chunk || binary_size > bytes.size() - next - 8U)
            return std::unexpected(detail::fail(error_code::invalid_format, "invalid GLB binary chunk"));
        output.binary.assign(bytes.begin() + static_cast<std::ptrdiff_t>(next + 8U),
                             bytes.begin() + static_cast<std::ptrdiff_t>(next + 8U + binary_size));
    }
    return output;
}

result<document> read_document(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return std::unexpected(detail::fail(error_code::io, "cannot open GLB: " + path.string()));
    const auto end = stream.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) > max_file)
        return std::unexpected(detail::fail(error_code::limit_exceeded, "GLB exceeds 512 MiB safety limit"));
    std::vector<std::byte> bytes(static_cast<std::size_t>(end));
    stream.seekg(0);
    if (!bytes.empty() && !stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        return std::unexpected(detail::fail(error_code::io, "failed to read GLB"));
    return parse_document(bytes);
}

result<const json *> item(const json & root, std::string_view array, std::size_t index) {
    auto found = root.find(array);
    if (found == root.end() || !found->is_array() || index >= found->size() || !(*found)[index].is_object())
        return std::unexpected(detail::fail(error_code::invalid_format, "glTF " + std::string{array} + " index is invalid"));
    return &(*found)[index];
}

result<std::size_t> integer(const json & object, std::string_view name, std::size_t fallback, bool required = true) {
    auto found = object.find(name);
    if (found == object.end()) {
        if (!required) return fallback;
        return std::unexpected(detail::fail(error_code::invalid_format, "missing glTF integer field " + std::string{name}));
    }
    if (!found->is_number_unsigned())
        return std::unexpected(detail::fail(error_code::invalid_format, "invalid glTF integer field " + std::string{name}));
    const auto value = found->get<std::uint64_t>();
    if (value > std::numeric_limits<std::size_t>::max())
        return std::unexpected(detail::fail(error_code::limit_exceeded, "glTF integer is out of range"));
    return static_cast<std::size_t>(value);
}

std::size_t scalar_size(std::uint32_t component) {
    if (component == 5121U) return 1U;
    if (component == 5123U) return 2U;
    if (component == 5125U || component == 5126U) return 4U;
    return 0U;
}

std::size_t components(std::string_view shape) {
    if (shape == "SCALAR") return 1U;
    if (shape == "VEC2") return 2U;
    if (shape == "VEC3") return 3U;
    if (shape == "VEC4") return 4U;
    if (shape == "MAT4") return 16U;
    return 0U;
}

struct accessor {
    std::span<const std::byte> bytes;
    std::size_t count = 0;
    std::size_t stride = 0;
    std::uint32_t component = 0;
    std::string shape;
    bool normalized = false;
};

result<accessor> get_accessor_object(const document & doc, const json & object);
result<accessor> get_accessor(const document & doc, std::size_t index) {
    auto accessor_item = item(doc.root, "accessors", index);
    if (!accessor_item) return std::unexpected(accessor_item.error());
    return get_accessor_object(doc, **accessor_item);
}

result<accessor> get_accessor_object(const document & doc, const json & object) {
    if (object.contains("sparse"))
        return std::unexpected(detail::fail(error_code::invalid_format, "sparse glTF accessors are unsupported"));
    auto view_index = integer(object, "bufferView", 0U);
    auto count = integer(object, "count", 0U);
    auto component = integer(object, "componentType", 0U);
    auto offset = integer(object, "byteOffset", 0U, false);
    if (!view_index) return std::unexpected(view_index.error());
    if (!count) return std::unexpected(count.error());
    if (!component) return std::unexpected(component.error());
    if (!offset) return std::unexpected(offset.error());
    auto shape_value = object.find("type");
    if (shape_value == object.end() || !shape_value->is_string())
        return std::unexpected(detail::fail(error_code::invalid_format, "glTF accessor type is missing"));
    const std::string shape = shape_value->get<std::string>();
    const std::size_t element = scalar_size(static_cast<std::uint32_t>(*component)) * components(shape);
    if (element == 0U || *count > max_vertices * 256U)
        return std::unexpected(detail::fail(error_code::limit_exceeded, "unsupported or oversized glTF accessor"));
    auto view_item = item(doc.root, "bufferViews", *view_index);
    if (!view_item) return std::unexpected(view_item.error());
    const auto & view = **view_item;
    auto buffer = integer(view, "buffer", 0U);
    auto view_size = integer(view, "byteLength", 0U);
    auto view_offset = integer(view, "byteOffset", 0U, false);
    auto stride = integer(view, "byteStride", element, false);
    if (!buffer) return std::unexpected(buffer.error());
    if (!view_size) return std::unexpected(view_size.error());
    if (!view_offset) return std::unexpected(view_offset.error());
    if (!stride) return std::unexpected(stride.error());
    if (*buffer != 0U || *stride < element || *stride > 252U || *offset > *view_size)
        return std::unexpected(detail::fail(error_code::invalid_format, "invalid GLB buffer view"));
    std::size_t payload = 0;
    if (*count != 0U) {
        std::size_t prefix = 0;
        if (!detail::checked_mul(*count - 1U, *stride, prefix) || prefix > std::numeric_limits<std::size_t>::max() - element)
            return std::unexpected(detail::fail(error_code::limit_exceeded, "glTF accessor size overflows"));
        payload = prefix + element;
    }
    if (payload > *view_size - *offset || *view_offset > doc.binary.size() ||
        *offset > doc.binary.size() - *view_offset || payload > doc.binary.size() - *view_offset - *offset)
        return std::unexpected(detail::fail(error_code::invalid_format, "glTF accessor exceeds binary buffer"));
    const std::size_t begin = *view_offset + *offset;
    return accessor{std::span<const std::byte>{doc.binary}.subspan(begin, payload), *count, *stride,
                    static_cast<std::uint32_t>(*component), shape, object.value("normalized", false)};
}

result<float> scalar(const accessor & value, std::size_t index) {
    if (value.component != 5126U || value.shape != "SCALAR" || index >= value.count)
        return std::unexpected(detail::fail(error_code::invalid_format, "animation time accessor must be float SCALAR"));
    const float output = little<float>(value.bytes.data() + index * value.stride);
    if (!std::isfinite(output)) return std::unexpected(detail::fail(error_code::invalid_format, "accessor contains non-finite float"));
    return output;
}

result<vec3> vector3(const accessor & value, std::size_t index) {
    if (value.component != 5126U || value.shape != "VEC3" || index >= value.count)
        return std::unexpected(detail::fail(error_code::invalid_format, "accessor must be float VEC3"));
    const auto * pointer = value.bytes.data() + index * value.stride;
    vec3 output{little<float>(pointer), little<float>(pointer + 4U), little<float>(pointer + 8U)};
    if (!std::isfinite(output.x) || !std::isfinite(output.y) || !std::isfinite(output.z))
        return std::unexpected(detail::fail(error_code::invalid_format, "accessor contains non-finite vector"));
    return output;
}

result<quat> quaternion(const accessor & value, std::size_t index) {
    if (value.component != 5126U || value.shape != "VEC4" || index >= value.count)
        return std::unexpected(detail::fail(error_code::invalid_format, "rotation accessor must be float VEC4"));
    const auto * pointer = value.bytes.data() + index * value.stride;
    quat output{little<float>(pointer), little<float>(pointer + 4U), little<float>(pointer + 8U), little<float>(pointer + 12U)};
    if (!std::isfinite(output.x) || !std::isfinite(output.y) || !std::isfinite(output.z) || !std::isfinite(output.w))
        return std::unexpected(detail::fail(error_code::invalid_format, "accessor contains non-finite quaternion"));
    return output;
}

result<std::array<std::uint16_t,4>> joint_vector(const accessor & value,std::size_t index) {
    if (value.shape!="VEC4" || index>=value.count ||
        (value.component!=5121U && value.component!=5123U))
        return std::unexpected(detail::fail(error_code::invalid_format,
            "JOINTS_0 must be an unsigned byte or unsigned short VEC4"));
    const auto * pointer=value.bytes.data()+index*value.stride;
    std::array<std::uint16_t,4> output{};
    for (std::size_t component=0;component<4U;++component)
        output[component]=value.component==5121U?
            std::to_integer<std::uint8_t>(pointer[component]):little<std::uint16_t>(pointer+component*2U);
    return output;
}

result<std::array<float,4>> weight_vector(const accessor & value,std::size_t index) {
    if (value.shape!="VEC4" || index>=value.count ||
        (value.component!=5121U && value.component!=5123U && value.component!=5126U))
        return std::unexpected(detail::fail(error_code::invalid_format,
            "WEIGHTS_0 must be a supported VEC4"));
    if (value.component!=5126U && !value.normalized)
        return std::unexpected(detail::fail(error_code::invalid_format,
            "integer WEIGHTS_0 must be normalized"));
    const auto * pointer=value.bytes.data()+index*value.stride;
    std::array<float,4> output{};
    for (std::size_t component=0;component<4U;++component) {
        output[component]=value.component==5126U?little<float>(pointer+component*4U):
            value.component==5123U?static_cast<float>(little<std::uint16_t>(pointer+component*2U))/65535.0F:
            static_cast<float>(std::to_integer<std::uint8_t>(pointer[component]))/255.0F;
        if (!std::isfinite(output[component]) || output[component]<0.0F)
            return std::unexpected(detail::fail(error_code::invalid_format,
                "WEIGHTS_0 contains an invalid value"));
    }
    const float sum=output[0]+output[1]+output[2]+output[3];
    if (sum<=1.0e-12F)
        return std::unexpected(detail::fail(error_code::invalid_format,
            "WEIGHTS_0 contains an unweighted vertex"));
    for (auto & weight:output) weight/=sum;
    return output;
}

result<color4> color(const accessor & value, std::size_t index) {
    const auto count = components(value.shape);
    if ((count != 3U && count != 4U) || index >= value.count ||
        (value.component != 5121U && value.component != 5123U && value.component != 5126U))
        return std::unexpected(detail::fail(error_code::invalid_format, "COLOR_0 must be a supported VEC3 or VEC4"));
    const auto * pointer = value.bytes.data() + index * value.stride;
    const auto channel = [&](std::size_t component) {
        if (value.component == 5126U) return little<float>(pointer + component * 4U);
        if (value.component == 5123U)
            return static_cast<float>(little<std::uint16_t>(pointer + component * 2U)) / 65535.0F;
        return static_cast<float>(std::to_integer<std::uint8_t>(pointer[component])) / 255.0F;
    };
    if (value.component != 5126U && !value.normalized)
        return std::unexpected(detail::fail(error_code::invalid_format, "integer COLOR_0 accessor must be normalized"));
    color4 output{channel(0U), channel(1U), channel(2U), count == 4U ? channel(3U) : 1.0F};
    if (!std::isfinite(output.r) || !std::isfinite(output.g) || !std::isfinite(output.b) || !std::isfinite(output.a))
        return std::unexpected(detail::fail(error_code::invalid_format, "COLOR_0 contains a non-finite value"));
    return output;
}

result<std::uint32_t> index_value(const accessor & value, std::size_t index) {
    if (value.shape != "SCALAR" || index >= value.count)
        return std::unexpected(detail::fail(error_code::invalid_format, "index accessor must be SCALAR"));
    const auto * pointer = value.bytes.data() + index * value.stride;
    if (value.component == 5121U) return std::to_integer<std::uint8_t>(*pointer);
    if (value.component == 5123U) return little<std::uint16_t>(pointer);
    if (value.component == 5125U) return little<std::uint32_t>(pointer);
    return std::unexpected(detail::fail(error_code::invalid_format, "unsupported index component type"));
}

vec3 add(vec3 left, vec3 right) { return {left.x + right.x, left.y + right.y, left.z + right.z}; }
vec3 sub(vec3 left, vec3 right) { return {left.x - right.x, left.y - right.y, left.z - right.z}; }
vec3 cross(vec3 left, vec3 right) {
    return {left.y * right.z - left.z * right.y, left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x};
}

using matrix4 = std::array<float, 16>;

quat q_mul(quat a, quat b) {
    return {a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y, a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
            a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w, a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z};
}
quat q_conj(quat a) { return {-a.x, -a.y, -a.z, a.w}; }
quat q_norm(quat a) {
    const float length = std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w);
    return length > 1e-12F ? quat{a.x/length, a.y/length, a.z/length, a.w/length} : quat{};
}
vec3 q_rotate(quat q, vec3 v) {
    const quat r = q_mul(q_mul(q, {v.x, v.y, v.z, 0.0F}), q_conj(q));
    return {r.x, r.y, r.z};
}
// A joint's rest in its parent's frame, given every joint's world rest rotation.
quat rest_local_rotation(const skeleton & rig, const std::vector<quat> & frames, std::size_t joint) {
    const std::int32_t parent = rig.parents[joint];
    return parent < 0 ? frames[joint] : q_norm(q_mul(q_conj(frames[static_cast<std::size_t>(parent)]), frames[joint]));
}
vec3 rest_local_translation(const skeleton & rig, const std::vector<quat> & frames, std::size_t joint) {
    const std::int32_t parent = rig.parents[joint];
    if (parent < 0) return rig.rest_positions[joint];
    const auto p = static_cast<std::size_t>(parent);
    return q_rotate(q_conj(frames[p]), sub(rig.rest_positions[joint], rig.rest_positions[p]));
}
// A clip's local rotation (world-aligned rest frames) in the joint's own rest frames.
quat framed_rotation(const skeleton & rig, const std::vector<quat> & frames, std::size_t joint, quat local) {
    const std::int32_t parent = rig.parents[joint];
    const quat parent_frame = parent < 0 ? quat{} : frames[static_cast<std::size_t>(parent)];
    return q_norm(q_mul(q_mul(q_conj(parent_frame), local), frames[joint]));
}
std::array<float, 16> framed_inverse_bind(vec3 position, quat frame) {
    const quat inverse = q_conj(frame);
    const vec3 x = q_rotate(inverse, {1, 0, 0}), y = q_rotate(inverse, {0, 1, 0}), z = q_rotate(inverse, {0, 0, 1});
    const vec3 t = q_rotate(inverse, {-position.x, -position.y, -position.z});
    return {x.x, x.y, x.z, 0, y.x, y.y, y.z, 0, z.x, z.y, z.z, 0, t.x, t.y, t.z, 1};
}
// Rotation part of a column-major matrix, with any scale divided out.
quat matrix_rotation(const std::array<float, 16> & m) {
    vec3 c0{m[0], m[1], m[2]}, c1{m[4], m[5], m[6]}, c2{m[8], m[9], m[10]};
    const auto len = [](vec3 v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); };
    const float s0 = len(c0), s1 = len(c1), s2 = len(c2);
    if (s0 < 1e-12F || s1 < 1e-12F || s2 < 1e-12F) return {};
    const float a = c0.x/s0, b = c1.x/s1, c = c2.x/s2, d = c0.y/s0, e = c1.y/s1, f = c2.y/s2,
                g = c0.z/s0, h = c1.z/s1, i = c2.z/s2;
    quat q;
    const float trace = a + e + i;
    if (trace > 0.0F) {
        const float t = std::sqrt(trace + 1.0F) * 2.0F;
        q = {(h - f)/t, (c - g)/t, (d - b)/t, 0.25F*t};
    } else if (a > e && a > i) {
        const float t = std::sqrt(1.0F + a - e - i) * 2.0F;
        q = {0.25F*t, (b + d)/t, (c + g)/t, (h - f)/t};
    } else if (e > i) {
        const float t = std::sqrt(1.0F + e - a - i) * 2.0F;
        q = {(b + d)/t, 0.25F*t, (f + h)/t, (c - g)/t};
    } else {
        const float t = std::sqrt(1.0F + i - a - e) * 2.0F;
        q = {(c + g)/t, (f + h)/t, 0.25F*t, (d - b)/t};
    }
    return q_norm(q);
}

matrix4 identity_matrix() {
    return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
}

matrix4 multiply(const matrix4 & left, const matrix4 & right) {
    matrix4 output{};
    for (std::size_t column = 0; column < 4U; ++column)
        for (std::size_t row = 0; row < 4U; ++row)
            for (std::size_t inner = 0; inner < 4U; ++inner)
                output[column * 4U + row] += left[inner * 4U + row] * right[column * 4U + inner];
    return output;
}

result<std::vector<float>> numeric_array(const json & object, std::string_view name, std::size_t count) {
    auto found = object.find(name);
    if (found == object.end()) return std::vector<float>{};
    if (!found->is_array() || found->size() != count)
        return std::unexpected(detail::fail(error_code::invalid_format, "invalid glTF node " + std::string{name}));
    std::vector<float> output(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (!(*found)[i].is_number())
            return std::unexpected(detail::fail(error_code::invalid_format, "non-numeric glTF node " + std::string{name}));
        output[i] = (*found)[i].get<float>();
        if (!std::isfinite(output[i]))
            return std::unexpected(detail::fail(error_code::invalid_format, "non-finite glTF node " + std::string{name}));
    }
    return output;
}

result<matrix4> node_matrix(const json & node) {
    auto explicit_matrix = numeric_array(node, "matrix", 16U);
    if (!explicit_matrix) return std::unexpected(explicit_matrix.error());
    if (!explicit_matrix->empty()) {
        matrix4 output{};
        std::copy(explicit_matrix->begin(), explicit_matrix->end(), output.begin());
        return output;
    }
    auto translation = numeric_array(node, "translation", 3U);
    auto rotation = numeric_array(node, "rotation", 4U);
    auto scale = numeric_array(node, "scale", 3U);
    if (!translation) return std::unexpected(translation.error());
    if (!rotation) return std::unexpected(rotation.error());
    if (!scale) return std::unexpected(scale.error());
    const float x = rotation->empty() ? 0.0F : (*rotation)[0];
    const float y = rotation->empty() ? 0.0F : (*rotation)[1];
    const float z = rotation->empty() ? 0.0F : (*rotation)[2];
    const float w = rotation->empty() ? 1.0F : (*rotation)[3];
    const float sx = scale->empty() ? 1.0F : (*scale)[0];
    const float sy = scale->empty() ? 1.0F : (*scale)[1];
    const float sz = scale->empty() ? 1.0F : (*scale)[2];
    const float tx = translation->empty() ? 0.0F : (*translation)[0];
    const float ty = translation->empty() ? 0.0F : (*translation)[1];
    const float tz = translation->empty() ? 0.0F : (*translation)[2];
    const float length = std::sqrt(x*x + y*y + z*z + w*w);
    if (length <= 1e-12F)
        return std::unexpected(detail::fail(error_code::invalid_format, "glTF node rotation is zero"));
    const float qx = x / length, qy = y / length, qz = z / length, qw = w / length;
    return matrix4{
        (1.0F - 2.0F*(qy*qy + qz*qz))*sx, 2.0F*(qx*qy + qz*qw)*sx, 2.0F*(qx*qz - qy*qw)*sx, 0,
        2.0F*(qx*qy - qz*qw)*sy, (1.0F - 2.0F*(qx*qx + qz*qz))*sy, 2.0F*(qy*qz + qx*qw)*sy, 0,
        2.0F*(qx*qz + qy*qw)*sz, 2.0F*(qy*qz - qx*qw)*sz, (1.0F - 2.0F*(qx*qx + qy*qy))*sz, 0,
        tx, ty, tz, 1
    };
}

vec3 transform_point(const matrix4 & matrix, vec3 value) {
    return {matrix[0]*value.x + matrix[4]*value.y + matrix[8]*value.z + matrix[12],
            matrix[1]*value.x + matrix[5]*value.y + matrix[9]*value.z + matrix[13],
            matrix[2]*value.x + matrix[6]*value.y + matrix[10]*value.z + matrix[14]};
}

float determinant3(const matrix4 & matrix) {
    return matrix[0]*(matrix[5]*matrix[10] - matrix[9]*matrix[6]) -
           matrix[4]*(matrix[1]*matrix[10] - matrix[9]*matrix[2]) +
           matrix[8]*(matrix[1]*matrix[6] - matrix[5]*matrix[2]);
}

template<class T>
void append_value(std::vector<std::byte> & bytes, const T & value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto old = bytes.size();
    bytes.resize(old + sizeof(T));
    std::memcpy(bytes.data() + old, &value, sizeof(T));
}

void align4(std::vector<std::byte> & bytes) {
    while (bytes.size() % 4U != 0U) bytes.push_back(std::byte{});
}

struct glb_builder {
    json root = {
        {"asset", {{"version", "2.0"}, {"generator", "skin-tokens.cpp"}}},
        {"buffers", json::array()}, {"bufferViews", json::array()},
        {"accessors", json::array()}, {"nodes", json::array()},
    };
    std::vector<std::byte> binary;

    template<class T>
    std::size_t accessor_for(std::span<const T> values, std::uint32_t component,
                             std::string_view type, std::optional<std::uint32_t> target = std::nullopt,
                             json extra = {}) {
        align4(binary);
        const std::size_t offset = binary.size();
        for (const T & value : values) append_value(binary, value);
        json view{{"buffer", 0U}, {"byteOffset", offset}, {"byteLength", values.size_bytes()}};
        if (target) view["target"] = *target;
        const std::size_t view_index = root["bufferViews"].size();
        root["bufferViews"].push_back(std::move(view));
        json accessor_value{{"bufferView", view_index}, {"componentType", component},
                            {"count", values.size()}, {"type", type}};
        if (!extra.is_null()) accessor_value.update(extra);
        const std::size_t index = root["accessors"].size();
        root["accessors"].push_back(std::move(accessor_value));
        return index;
    }
};

result<void> validate_export(const mesh & geometry, const skin & binding, const motion & animation) {
    if (geometry.vertices.empty() || geometry.faces.empty() || geometry.normals.size() != geometry.vertices.size())
        return std::unexpected(detail::fail(error_code::invalid_argument, "export mesh arrays are incomplete"));
    if (!geometry.colors.empty() && geometry.colors.size() != geometry.vertices.size())
        return std::unexpected(detail::fail(error_code::invalid_argument, "mesh colours must match vertex count"));
    for (const auto & value : geometry.colors) {
        if (!std::isfinite(value.r) || !std::isfinite(value.g) || !std::isfinite(value.b) || !std::isfinite(value.a))
            return std::unexpected(detail::fail(error_code::invalid_argument, "mesh contains a non-finite colour"));
    }
    if (binding.joints.size() != geometry.vertices.size() || binding.weights.size() != geometry.vertices.size())
        return std::unexpected(detail::fail(error_code::invalid_argument, "skin influence count does not match mesh vertices"));
    if (binding.rig.names.empty() || binding.rig.names.size() > 256U ||
        binding.rig.parents.size() != binding.rig.names.size() ||
        binding.rig.rest_positions.size() != binding.rig.names.size())
        return std::unexpected(detail::fail(error_code::invalid_argument, "skin skeleton arrays are invalid"));
    if (animation.frames == 0U || animation.rig.names != binding.rig.names ||
        animation.rig.parents != binding.rig.parents ||
        animation.root_translations.size() != animation.frames ||
        animation.local_rotations.size() != animation.frames * binding.rig.names.size())
        return std::unexpected(detail::fail(error_code::invalid_argument, "animation does not match the skin skeleton"));
    for (std::size_t vertex = 0; vertex < binding.joints.size(); ++vertex) {
        float sum = 0.0F;
        for (std::size_t slot = 0; slot < 4U; ++slot) {
            if (binding.joints[vertex][slot] >= binding.rig.names.size() ||
                !std::isfinite(binding.weights[vertex][slot]) || binding.weights[vertex][slot] < 0.0F)
                return std::unexpected(detail::fail(error_code::invalid_argument, "skin contains an invalid influence"));
            sum += binding.weights[vertex][slot];
        }
        if (std::abs(sum - 1.0F) > 1e-3F)
            return std::unexpected(detail::fail(error_code::invalid_argument, "skin weights are not normalized"));
    }
    return {};
}

} // namespace

namespace {
result<mesh> load_glb_document(result<document> doc);
} // namespace

result<mesh> load_glb_file(const std::filesystem::path & path) {
    return load_glb_document(read_document(path));
}

result<mesh> load_glb_memory(std::span<const std::byte> bytes) {
    return load_glb_document(parse_document(bytes));
}

namespace {
// Every mesh placed by the selected scene, in the order the loader reads them, with its world
// transform and the node that places it. The writer that keeps the source GLB walks the same list,
// so vertex i of the loaded mesh is vertex i of this walk.
constexpr std::size_t no_node = std::numeric_limits<std::size_t>::max();
struct mesh_instance { std::size_t mesh_index; matrix4 transform; std::size_t node; };

bool loaded_primitive(const json & primitive) {
    return primitive.value("mode", 4U) == 4U && primitive.contains("attributes") && primitive.contains("indices") &&
        primitive["attributes"].contains("POSITION") && primitive["attributes"]["POSITION"].is_number_unsigned();
}

result<std::vector<mesh_instance>> collect_mesh_instances(const document & doc, const json & meshes) {
    std::vector<mesh_instance> instances;
    auto nodes = doc.root.find("nodes");
    auto scenes = doc.root.find("scenes");
    if (nodes != doc.root.end() && nodes->is_array() && scenes != doc.root.end() && scenes->is_array() && !scenes->empty()) {
        std::size_t scene_index = 0U;
        if (auto selected = doc.root.find("scene"); selected != doc.root.end()) {
            if (!selected->is_number_unsigned())
                return std::unexpected(detail::fail(error_code::invalid_format, "glTF scene index is invalid"));
            scene_index = selected->get<std::size_t>();
        }
        if (scene_index >= scenes->size() || !(*scenes)[scene_index].is_object())
            return std::unexpected(detail::fail(error_code::invalid_format, "glTF scene index is out of range"));
        const auto & roots = (*scenes)[scene_index].value("nodes", json::array());
        if (!roots.is_array())
            return std::unexpected(detail::fail(error_code::invalid_format, "glTF scene roots are invalid"));
        std::vector<std::uint8_t> visiting(nodes->size(), 0U);
        std::vector<std::optional<matrix4>> node_world(nodes->size());
        std::function<result<void>(std::size_t, const matrix4 &)> visit =
            [&](std::size_t index, const matrix4 & parent) -> result<void> {
                if (index >= nodes->size() || !(*nodes)[index].is_object())
                    return std::unexpected(detail::fail(error_code::invalid_format, "glTF node index is invalid"));
                if (visiting[index] != 0U)
                    return std::unexpected(detail::fail(error_code::invalid_format, "glTF scene graph contains a cycle or repeated node"));
                visiting[index] = 1U;
                const auto & node = (*nodes)[index];
                auto local = node_matrix(node);
                if (!local) return std::unexpected(local.error());
                const matrix4 world = multiply(parent, *local);
                node_world[index] = world;
                if (auto mesh_index = node.find("mesh"); mesh_index != node.end()) {
                    if (!mesh_index->is_number_unsigned() || mesh_index->get<std::size_t>() >= meshes.size())
                        return std::unexpected(detail::fail(error_code::invalid_format, "glTF node mesh index is invalid"));
                    if (instances.size() >= 4096U)
                        return std::unexpected(detail::fail(error_code::limit_exceeded, "too many glTF mesh instances"));
                    instances.push_back({mesh_index->get<std::size_t>(), world, index});
                }
                if (auto children = node.find("children"); children != node.end()) {
                    if (!children->is_array())
                        return std::unexpected(detail::fail(error_code::invalid_format, "glTF node children are invalid"));
                    for (const auto & child : *children) {
                        if (!child.is_number_unsigned())
                            return std::unexpected(detail::fail(error_code::invalid_format, "glTF child index is invalid"));
                        auto nested = visit(child.get<std::size_t>(), world);
                        if (!nested) return nested;
                    }
                }
                visiting[index] = 0U;
                return {};
            };
        for (const auto & root : roots) {
            if (!root.is_number_unsigned())
                return std::unexpected(detail::fail(error_code::invalid_format, "glTF scene root is invalid"));
            auto visited = visit(root.get<std::size_t>(), identity_matrix());
            if (!visited) return std::unexpected(visited.error());
        }
        // glTF ignores a skinned mesh's node transform: its joints place it. At rest each vertex
        // goes through joint world x inverse bind; use the matrix of the joint carrying the most
        // weight (for a mesh bound in its rest pose every weighted joint gives the same one).
        for (auto & instance : instances) {
            const auto & node = (*nodes)[instance.node];
            auto skin_index = node.find("skin");
            if (skin_index == node.end()) continue;
            if (!skin_index->is_number_unsigned())
                return std::unexpected(detail::fail(error_code::invalid_format, "glTF node skin index is invalid"));
            auto skin_item = item(doc.root, "skins", skin_index->get<std::size_t>());
            if (!skin_item) return std::unexpected(skin_item.error());
            const auto & joints = (**skin_item).value("joints", json::array());
            if (!joints.is_array() || joints.empty()) continue;
            std::vector<double> load(joints.size(), 0.0);
            for (const auto & primitive : meshes[instance.mesh_index].value("primitives", json::array())) {
                if (!loaded_primitive(primitive)) continue;
                const auto & attributes = primitive["attributes"];
                if (!attributes.contains("JOINTS_0") || !attributes.contains("WEIGHTS_0")) continue;
                auto joint_values = get_accessor(doc, attributes["JOINTS_0"].get<std::size_t>());
                auto weight_values = get_accessor(doc, attributes["WEIGHTS_0"].get<std::size_t>());
                if (!joint_values) return std::unexpected(joint_values.error());
                if (!weight_values) return std::unexpected(weight_values.error());
                for (std::size_t v = 0; v < joint_values->count && v < weight_values->count; ++v) {
                    auto j = joint_vector(*joint_values, v);
                    auto w = weight_vector(*weight_values, v);
                    if (!j) return std::unexpected(j.error());
                    if (!w) continue;  // an unweighted vertex adds nothing
                    for (std::size_t c = 0; c < 4U; ++c)
                        if ((*w)[c] > 0.0F && (*j)[c] < load.size()) load[(*j)[c]] += (*w)[c];
                }
            }
            const auto heaviest = static_cast<std::size_t>(std::max_element(load.begin(), load.end()) - load.begin());
            if (load[heaviest] <= 0.0 || !joints[heaviest].is_number_unsigned()) continue;
            const auto joint_node = joints[heaviest].get<std::size_t>();
            if (joint_node >= node_world.size() || !node_world[joint_node]) continue;
            matrix4 bind = identity_matrix();
            if (auto binds = (**skin_item).find("inverseBindMatrices"); binds != (**skin_item).end()) {
                if (!binds->is_number_unsigned())
                    return std::unexpected(detail::fail(error_code::invalid_format, "glTF inverse binds index is invalid"));
                auto read = get_accessor(doc, binds->get<std::size_t>());
                if (!read) return std::unexpected(read.error());
                if (read->component != 5126U || read->shape != "MAT4" || read->count != joints.size())
                    return std::unexpected(detail::fail(error_code::invalid_format, "inverse binds must be float MAT4, one per joint"));
                for (std::size_t k = 0; k < 16U; ++k)
                    bind[k] = little<float>(read->bytes.data() + heaviest * read->stride + k * 4U);
            }
            instance.transform = multiply(*node_world[joint_node], bind);
        }
    } else {
        for (std::size_t index = 0; index < meshes.size(); ++index)
            instances.push_back({index, identity_matrix(), no_node});
    }

    return instances;
}

result<mesh> load_glb_document(result<document> doc) {
    if (!doc) return std::unexpected(doc.error());
    auto meshes = doc->root.find("meshes");
    if (meshes == doc->root.end() || !meshes->is_array())
        return std::unexpected(detail::fail(error_code::invalid_format, "GLB contains no meshes"));

    auto collected = collect_mesh_instances(*doc, *meshes);
    if (!collected) return std::unexpected(collected.error());
    const auto & instances = *collected;
    mesh output;
    bool has_colors = false;
    double metallic_sum = 0.0;
    double roughness_sum = 0.0;
    std::size_t material_vertices = 0U;
    for (const auto & instance : instances) {
        const auto & mesh_object = (*meshes)[instance.mesh_index];
        if (!mesh_object.is_object())
            return std::unexpected(detail::fail(error_code::invalid_format, "glTF mesh is invalid"));
        auto primitives = mesh_object.find("primitives");
        if (primitives == mesh_object.end() || !primitives->is_array()) continue;
        for (const auto & primitive : *primitives) {
            if (!loaded_primitive(primitive)) continue;
            const auto & attributes = primitive["attributes"];
            auto positions = get_accessor(*doc, attributes["POSITION"].get<std::size_t>());
            auto indices = get_accessor(*doc, primitive["indices"].get<std::size_t>());
            if (!positions) return std::unexpected(positions.error());
            if (!indices) return std::unexpected(indices.error());
            if (indices->count % 3U != 0U || output.vertices.size() + positions->count > max_vertices)
                return std::unexpected(detail::fail(error_code::limit_exceeded, "mesh primitive exceeds supported size"));
            std::optional<accessor> colors;
            if (auto color_index = attributes.find("COLOR_0"); color_index != attributes.end()) {
                if (!color_index->is_number_unsigned())
                    return std::unexpected(detail::fail(error_code::invalid_format, "COLOR_0 accessor index is invalid"));
                auto value = get_accessor(*doc, color_index->get<std::size_t>());
                if (!value) return std::unexpected(value.error());
                if (value->count != positions->count)
                    return std::unexpected(detail::fail(error_code::invalid_format, "COLOR_0 count differs from POSITION"));
                colors = *value;
                has_colors = true;
            }
            color4 factor;
            float metallic = 0.0F, roughness = 0.72F;
            if (auto material_index = primitive.find("material"); material_index != primitive.end()) {
                if (!material_index->is_number_unsigned())
                    return std::unexpected(detail::fail(error_code::invalid_format, "glTF material index is invalid"));
                auto material = item(doc->root, "materials", material_index->get<std::size_t>());
                if (!material) return std::unexpected(material.error());
                output.double_sided = output.double_sided || (**material).value("doubleSided", false);
                if (auto pbr = (**material).find("pbrMetallicRoughness"); pbr != (**material).end() && pbr->is_object()) {
                    metallic = pbr->value("metallicFactor", 1.0F);
                    roughness = pbr->value("roughnessFactor", 1.0F);
                    auto base = numeric_array(*pbr, "baseColorFactor", 4U);
                    if (!base) return std::unexpected(base.error());
                    if (!base->empty()) {
                        factor = {(*base)[0], (*base)[1], (*base)[2], (*base)[3]};
                        has_colors = has_colors || factor.r != 1.0F || factor.g != 1.0F ||
                            factor.b != 1.0F || factor.a != 1.0F;
                    }
                }
            }
            if (!std::isfinite(metallic) || !std::isfinite(roughness))
                return std::unexpected(detail::fail(error_code::invalid_format, "glTF material contains a non-finite factor"));
            metallic_sum += static_cast<double>(metallic) * static_cast<double>(positions->count);
            roughness_sum += static_cast<double>(roughness) * static_cast<double>(positions->count);
            material_vertices += positions->count;
            const auto base = static_cast<std::uint32_t>(output.vertices.size());
            for (std::size_t i = 0; i < positions->count; ++i) {
                auto value = vector3(*positions, i);
                if (!value) return std::unexpected(value.error());
                output.vertices.push_back(transform_point(instance.transform, *value));
                color4 shade = factor;
                if (colors) {
                    auto vertex_color = color(*colors, i);
                    if (!vertex_color) return std::unexpected(vertex_color.error());
                    shade = {vertex_color->r * factor.r, vertex_color->g * factor.g,
                             vertex_color->b * factor.b, vertex_color->a * factor.a};
                }
                output.colors.push_back(shade);
            }
            const bool mirrored = determinant3(instance.transform) < 0.0F;
            for (std::size_t i = 0; i < indices->count; i += 3U) {
                triangle face{};
                for (std::size_t j = 0; j < 3U; ++j) {
                    auto value = index_value(*indices, i + j);
                    if (!value) return std::unexpected(value.error());
                    if (*value >= positions->count)
                        return std::unexpected(detail::fail(error_code::invalid_format, "triangle index exceeds POSITION accessor"));
                    face[j] = base + *value;
                }
                if (mirrored) std::swap(face[1], face[2]);
                output.faces.push_back(face);
            }
        }
    }
    if (output.vertices.empty() || output.faces.empty())
        return std::unexpected(detail::fail(error_code::invalid_format, "GLB has no indexed triangle geometry"));
    if (!has_colors) output.colors.clear();
    if (material_vertices != 0U) {
        output.metallic_factor = static_cast<float>(metallic_sum / static_cast<double>(material_vertices));
        output.roughness_factor = static_cast<float>(roughness_sum / static_cast<double>(material_vertices));
    }
    output.normals.assign(output.vertices.size(), {});
    for (const auto & face : output.faces) {
        const vec3 normal = cross(sub(output.vertices[face[1]], output.vertices[face[0]]),
                                  sub(output.vertices[face[2]], output.vertices[face[0]]));
        for (auto vertex : face) output.normals[vertex] = add(output.normals[vertex], normal);
    }
    for (auto & normal : output.normals) {
        const float length = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (length > 1e-12F) { normal.x /= length; normal.y /= length; normal.z /= length; }
        else normal = {0.0F, 1.0F, 0.0F};
    }
    return output;
}
} // namespace

result<mesh> load_trellis_mesh_file(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return std::unexpected(detail::fail(error_code::io, "cannot open T2MESH: " + path.string()));
    const auto end = stream.tellg();
    if (end < 16 || static_cast<std::uint64_t>(end) > max_file)
        return std::unexpected(detail::fail(error_code::limit_exceeded, "T2MESH size is outside supported limits"));
    const std::size_t file_size = static_cast<std::size_t>(end);
    stream.seekg(0);
    std::array<char, 8> magic{};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    const std::string_view signature{magic.data(), magic.size()};
    const std::size_t pbr_width = signature == "T2MESH01" ? 0U :
        signature == "T2MESH02" ? 5U : signature == "T2MESH03" ? 6U : std::numeric_limits<std::size_t>::max();
    if (pbr_width == std::numeric_limits<std::size_t>::max())
        return std::unexpected(detail::fail(error_code::invalid_format, "unsupported T2MESH version"));
    std::array<std::byte, 8> counts{};
    stream.read(reinterpret_cast<char *>(counts.data()), static_cast<std::streamsize>(counts.size()));
    if (!stream) return std::unexpected(detail::fail(error_code::invalid_format, "truncated T2MESH header"));
    const std::size_t vertex_count = u32(counts, 0U);
    const std::size_t triangle_count = u32(counts, 4U);
    if (vertex_count == 0U || vertex_count > max_vertices || triangle_count == 0U || triangle_count > max_vertices * 2U)
        return std::unexpected(detail::fail(error_code::limit_exceeded, "T2MESH geometry exceeds supported limits"));
    std::size_t per_vertex = 0U, vertex_bytes = 0U, triangle_bytes = 0U;
    if (!detail::checked_mul(6U + pbr_width, sizeof(float), per_vertex) ||
        !detail::checked_mul(vertex_count, per_vertex, vertex_bytes) ||
        !detail::checked_mul(triangle_count, sizeof(std::int32_t) * 3U, triangle_bytes) ||
        vertex_bytes > std::numeric_limits<std::size_t>::max() - 16U ||
        vertex_bytes > file_size - 16U ||
        triangle_bytes != file_size - 16U - vertex_bytes)
        return std::unexpected(detail::fail(error_code::invalid_format, "T2MESH byte length does not match its header"));

    const auto read_float = [&]() -> result<float> {
        std::array<std::byte, 4> bytes{};
        stream.read(reinterpret_cast<char *>(bytes.data()), 4);
        if (!stream) return std::unexpected(detail::fail(error_code::invalid_format, "truncated T2MESH float stream"));
        const float value = little<float>(bytes.data());
        if (!std::isfinite(value))
            return std::unexpected(detail::fail(error_code::invalid_format, "T2MESH contains a non-finite float"));
        return value;
    };
    mesh output;
    output.vertices.resize(vertex_count);
    output.normals.resize(vertex_count);
    for (auto & value : output.vertices) {
        auto x = read_float(), y = read_float(), z = read_float();
        if (!x) return std::unexpected(x.error());
        if (!y) return std::unexpected(y.error());
        if (!z) return std::unexpected(z.error());
        value = {*x, *z, -*y}; // Trellis Z-up -> glTF/SkinTokens Y-up.
    }
    for (auto & value : output.normals) {
        auto x = read_float(), y = read_float(), z = read_float();
        if (!x) return std::unexpected(x.error());
        if (!y) return std::unexpected(y.error());
        if (!z) return std::unexpected(z.error());
        value = {*x, *z, -*y};
        const float length = std::sqrt(value.x*value.x + value.y*value.y + value.z*value.z);
        if (length > 1e-12F) { value.x /= length; value.y /= length; value.z /= length; }
    }
    if (pbr_width != 0U) {
        const auto linear = [](float value) {
            value = std::clamp(value, 0.0F, 1.0F);
            return value <= 0.04045F ? value / 12.92F : std::pow((value + 0.055F) / 1.055F, 2.4F);
        };
        output.colors.resize(vertex_count);
        double metallic = 0.0, roughness = 0.0, weight = 0.0;
        for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
            std::array<float, 6> pbr{0.7F, 0.7F, 0.7F, 0.0F, 0.6F, 1.0F};
            for (std::size_t channel = 0; channel < pbr_width; ++channel) {
                auto value = read_float();
                if (!value) return std::unexpected(value.error());
                pbr[channel] = *value;
            }
            const float alpha = std::clamp(pbr[5], 0.0F, 1.0F);
            output.colors[vertex] = {linear(pbr[0]), linear(pbr[1]), linear(pbr[2]), alpha};
            const double visible = std::max(0.001, static_cast<double>(alpha));
            metallic += std::clamp(pbr[3], 0.0F, 1.0F) * visible;
            roughness += std::clamp(pbr[4], 0.0F, 1.0F) * visible;
            weight += visible;
        }
        output.metallic_factor = static_cast<float>(metallic / weight);
        output.roughness_factor = static_cast<float>(roughness / weight);
        output.double_sided = true;
    }
    output.faces.resize(triangle_count);
    for (auto & face : output.faces) {
        for (auto & index : face) {
            std::array<std::byte, 4> bytes{};
            stream.read(reinterpret_cast<char *>(bytes.data()), 4);
            if (!stream) return std::unexpected(detail::fail(error_code::invalid_format, "truncated T2MESH index stream"));
            const std::int32_t signed_index = little<std::int32_t>(bytes.data());
            if (signed_index < 0 || static_cast<std::size_t>(signed_index) >= vertex_count)
                return std::unexpected(detail::fail(error_code::invalid_format, "T2MESH triangle index is out of range"));
            index = static_cast<std::uint32_t>(signed_index);
        }
    }
    return output;
}

namespace {

std::string canonical_joint_name(std::string_view name) {
    const auto colon = name.rfind(':');
    if (colon != std::string_view::npos) name.remove_prefix(colon + 1U);
    std::string output{name};
    std::transform(output.begin(), output.end(), output.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return output;
}

result<motion> load_skeleton_document(const document & doc, bool require_animation) {
    auto nodes = doc.root.find("nodes");
    auto animations = doc.root.find("animations");
    // The skeleton itself is bounded below (256 joints); a rigged character may hold many more
    // nodes (its meshes and the source's old bones).
    if (nodes == doc.root.end() || !nodes->is_array() || nodes->empty() || nodes->size() > 65536U)
        return std::unexpected(detail::fail(error_code::invalid_format, "GLB has no bounded skeleton nodes"));
    if (require_animation &&
        (animations == doc.root.end() || !animations->is_array() || animations->empty()))
        return std::unexpected(detail::fail(error_code::invalid_format, "GLB has no bounded skeleton animation"));
    const auto declared_skins = doc.root.find("skins");
    const auto declared_meshes = doc.root.find("meshes");
    const bool has_declared_skin = declared_skins != doc.root.end() && declared_skins->is_array() && !declared_skins->empty();
    const bool has_declared_animation = animations != doc.root.end() && animations->is_array() && !animations->empty();
    const bool has_declared_mesh = declared_meshes != doc.root.end() && declared_meshes->is_array() && !declared_meshes->empty();
    if (!has_declared_skin && !has_declared_animation && (has_declared_mesh || nodes->size() < 2U))
        return std::unexpected(detail::fail(error_code::invalid_format, "GLB contains no identifiable skeleton"));
    motion output;
    std::vector<std::size_t> joint_nodes;
    const auto skins = doc.root.find("skins");
    if (skins != doc.root.end() && skins->is_array() && !skins->empty()) {
        if (!(*skins)[0].is_object() || !(*skins)[0].contains("joints") ||
            !(*skins)[0]["joints"].is_array())
            return std::unexpected(detail::fail(error_code::invalid_format, "glTF skin has no joint array"));
        for (const auto & value : (*skins)[0]["joints"]) {
            if (!value.is_number_unsigned() || value.get<std::size_t>() >= nodes->size())
                return std::unexpected(detail::fail(error_code::invalid_format, "skin joint node is invalid"));
            joint_nodes.push_back(value.get<std::size_t>());
        }
    } else {
        joint_nodes.resize(nodes->size());
        std::iota(joint_nodes.begin(), joint_nodes.end(), 0U);
    }
    if (joint_nodes.empty() || joint_nodes.size() > 256U)
        return std::unexpected(detail::fail(error_code::limit_exceeded, "GLB has no bounded skin joints"));
    std::vector<std::int32_t> node_to_joint(nodes->size(), -1);
    for (std::size_t joint = 0; joint < joint_nodes.size(); ++joint) {
        if (node_to_joint[joint_nodes[joint]] >= 0)
            return std::unexpected(detail::fail(error_code::invalid_format, "skin contains duplicate joints"));
        node_to_joint[joint_nodes[joint]] = static_cast<std::int32_t>(joint);
    }
    const std::size_t joint_count = joint_nodes.size();
    output.rig.names.resize(joint_count);
    output.rig.parents.assign(joint_count, -1);
    output.rig.rest_positions.resize(joint_count);
    std::vector<std::int32_t> node_parents(nodes->size(), -1);
    for (std::size_t i = 0; i < nodes->size(); ++i) {
        const auto & node = (*nodes)[i];
        auto children = node.find("children");
        if (children != node.end() && children->is_array()) {
            for (const auto & child_value : *children) {
                if (!child_value.is_number_unsigned() || child_value.get<std::size_t>() >= nodes->size())
                    return std::unexpected(detail::fail(error_code::invalid_format, "skeleton child index is invalid"));
                const auto child = child_value.get<std::size_t>();
                if (node_parents[child] >= 0)
                    return std::unexpected(detail::fail(error_code::invalid_format, "skeleton node has multiple parents"));
                node_parents[child] = static_cast<std::int32_t>(i);
            }
        }
    }
    // glTF does not require skin.joints to be topologically ordered, while
    // TokenRig and the exported flat hierarchy require parents before
    // children. Preserve sibling order but normalize arbitrary valid skins.
    std::vector<std::size_t> ordered_joints;
    ordered_joints.reserve(joint_nodes.size());
    std::vector<bool> added(nodes->size(), false);
    while (ordered_joints.size() != joint_nodes.size()) {
        const auto before = ordered_joints.size();
        for (const auto node_index : joint_nodes) {
            if (added[node_index]) continue;
            std::int32_t parent_node = node_parents[node_index];
            while (parent_node >= 0 && node_to_joint[static_cast<std::size_t>(parent_node)] < 0)
                parent_node = node_parents[static_cast<std::size_t>(parent_node)];
            if (parent_node < 0 || added[static_cast<std::size_t>(parent_node)]) {
                ordered_joints.push_back(node_index);
                added[node_index] = true;
            }
        }
        if (ordered_joints.size() == before)
            return std::unexpected(detail::fail(error_code::invalid_format, "skin joint hierarchy contains a cycle"));
    }
    joint_nodes = std::move(ordered_joints);
    std::fill(node_to_joint.begin(), node_to_joint.end(), -1);
    for (std::size_t joint = 0; joint < joint_nodes.size(); ++joint)
        node_to_joint[joint_nodes[joint]] = static_cast<std::int32_t>(joint);
    std::vector<matrix4> node_transforms(nodes->size());
    std::vector<std::uint8_t> transform_state(nodes->size());
    std::function<result<matrix4>(std::size_t)> global_transform = [&](std::size_t node) -> result<matrix4> {
        if (transform_state[node] == 2U) return node_transforms[node];
        if (transform_state[node] == 1U)
            return std::unexpected(detail::fail(error_code::invalid_format, "skeleton hierarchy contains a cycle"));
        transform_state[node] = 1U;
        auto local = node_matrix((*nodes)[node]);
        if (!local) return std::unexpected(local.error());
        if (node_parents[node] < 0) node_transforms[node] = *local;
        else {
            auto parent = global_transform(static_cast<std::size_t>(node_parents[node]));
            if (!parent) return std::unexpected(parent.error());
            node_transforms[node] = multiply(*parent, *local);
        }
        transform_state[node] = 2U;
        return node_transforms[node];
    };
    for (std::size_t joint = 0; joint < joint_count; ++joint) {
        const std::size_t node_index = joint_nodes[joint];
        output.rig.names[joint] = (*nodes)[node_index].value("name", "joint_" + std::to_string(joint));
        std::int32_t parent_node = node_parents[node_index];
        while (parent_node >= 0 && node_to_joint[static_cast<std::size_t>(parent_node)] < 0)
            parent_node = node_parents[static_cast<std::size_t>(parent_node)];
        output.rig.parents[joint] = parent_node < 0 ? -1 : node_to_joint[static_cast<std::size_t>(parent_node)];
        auto transform = global_transform(node_index);
        if (!transform) return std::unexpected(transform.error());
        output.rig.rest_positions[joint] = {(*transform)[12], (*transform)[13], (*transform)[14]};
    }
    output.frames = 1U;
    output.frames_per_second = 30.0F;
    output.root_translations.assign(1U, output.rig.rest_positions.front());
    output.local_rotations.assign(joint_count, {});
    if (animations == doc.root.end() || !animations->is_array() || animations->empty()) return output;
    const auto & animation = (*animations)[0];
    if (auto name = animation.find("name"); name != animation.end() && name->is_string())
        output.name = name->get<std::string>();
    auto samplers = animation.find("samplers");
    auto channels = animation.find("channels");
    if (samplers == animation.end() || !samplers->is_array() || channels == animation.end() || !channels->is_array())
        return std::unexpected(detail::fail(error_code::invalid_format, "animation has no samplers/channels"));
    struct channel_data { std::size_t node; std::string path; accessor times; accessor values; };
    std::vector<channel_data> parsed;
    std::size_t frames = 0;
    float duration = 0.0F;
    for (const auto & channel : *channels) {
        auto sampler_index = integer(channel, "sampler", 0U);
        if (!sampler_index || *sampler_index >= samplers->size() || !channel.contains("target")) continue;
        const auto & sampler = (*samplers)[*sampler_index];
        const auto & target = channel["target"];
        if (!sampler.contains("input") || !sampler.contains("output") || !target.contains("node") || !target.contains("path")) continue;
        if (!sampler["input"].is_number_unsigned() || !sampler["output"].is_number_unsigned() ||
            !target["node"].is_number_unsigned() || !target["path"].is_string()) continue;
        auto times = get_accessor(doc, sampler["input"].get<std::size_t>());
        auto values = get_accessor(doc, sampler["output"].get<std::size_t>());
        if (!times) return std::unexpected(times.error());
        if (!values) return std::unexpected(values.error());
        if (times->count != values->count || target["node"].get<std::size_t>() >= nodes->size()) continue;
        const auto joint = node_to_joint[target["node"].get<std::size_t>()];
        if (joint < 0) continue;
        frames = std::max(frames, times->count);
        if (times->count) {
            auto last = scalar(*times, times->count - 1U);
            if (!last) return std::unexpected(last.error());
            duration = std::max(duration, *last);
        }
        parsed.push_back({static_cast<std::size_t>(joint), target["path"].get<std::string>(), *times, *values});
    }
    if (frames > 100000U)
        return std::unexpected(detail::fail(error_code::limit_exceeded, "animation exceeds 100000 keyframes"));
    if (frames == 0U) {
        if (require_animation)
            return std::unexpected(detail::fail(error_code::invalid_format, "animation has no bounded keyframes"));
        return output;
    }
    std::vector<quat> own_rest(joint_count), parent_rest(joint_count);
    for (std::size_t joint = 0; joint < joint_count; ++joint) {
        const std::size_t node_index = joint_nodes[joint];
        auto transform = global_transform(node_index);
        if (!transform) return std::unexpected(transform.error());
        own_rest[joint] = matrix_rotation(*transform);
        if (node_parents[node_index] >= 0) {
            auto parent = global_transform(static_cast<std::size_t>(node_parents[node_index]));
            if (!parent) return std::unexpected(parent.error());
            parent_rest[joint] = matrix_rotation(*parent);
        }
    }
    output.frames = frames;
    output.frames_per_second = duration > 0.0F ? static_cast<float>(frames - 1U) / duration : 30.0F;
    output.root_translations.assign(frames, output.rig.rest_positions[0]);
    output.local_rotations.assign(frames * joint_count, {});
    for (const auto & value : parsed) {
        for (std::size_t frame = 0; frame < std::min(frames, value.values.count); ++frame) {
            if (value.path == "translation" && output.rig.parents[value.node] < 0) {
                auto translation = vector3(value.values, frame);
                if (!translation) return std::unexpected(translation.error());
                output.root_translations[frame] = *translation;
            } else if (value.path == "rotation") {
                auto rotation = quaternion(value.values, frame);
                if (!rotation) return std::unexpected(rotation.error());
                // Stored relative to world-aligned rest frames: a joint with a turned rest reads
                // identity at rest, as the rest of the library expects.
                output.local_rotations[frame * joint_count + value.node] =
                    q_norm(q_mul(q_mul(parent_rest[value.node], *rotation), q_conj(own_rest[value.node])));
            }
        }
    }
    return output;
}

} // namespace

std::vector<quat> joint_rest_frames(const skeleton & rig) { return joint_rest_frames(rig, {}); }

std::vector<quat> joint_rest_frames(const skeleton & rig, const std::vector<std::optional<vec3>> & aims) {
    // Each joint's up (+Y) runs along its bone, toward the child that best continues the
    // parent's direction; the side axes turn as little as possible from the parent's.
    const std::size_t count = rig.names.size();
    std::vector<quat> frames(count);
    const auto length = [](vec3 v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); };
    const auto dot = [](vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; };
    for (std::size_t joint = 0; joint < count && joint < rig.parents.size() && joint < rig.rest_positions.size(); ++joint) {
        const std::int32_t parent = rig.parents[joint];
        const quat base = parent >= 0 && static_cast<std::size_t>(parent) < joint ? frames[static_cast<std::size_t>(parent)] : quat{};
        const vec3 up = q_rotate(base, {0.0F, 1.0F, 0.0F});
        vec3 best{};
        float best_score = -2.0F;
        for (std::size_t child = joint + 1U; child < count; ++child) {
            if (rig.parents[child] != static_cast<std::int32_t>(joint)) continue;
            const vec3 bone = sub(rig.rest_positions[child], rig.rest_positions[joint]);
            const float bone_length = length(bone);
            if (bone_length < 1e-6F) continue;
            const vec3 direction{bone.x/bone_length, bone.y/bone_length, bone.z/bone_length};
            const float score = dot(direction, up);
            if (score > best_score) { best_score = score; best = direction; }
        }
        if (joint < aims.size() && aims[joint]) {
            const vec3 bone = sub(*aims[joint], rig.rest_positions[joint]);
            const float bone_length = length(bone);
            if (bone_length > 1e-6F) { best = {bone.x/bone_length, bone.y/bone_length, bone.z/bone_length}; best_score = 0.0F; }
        }
        if (best_score < -1.5F) { frames[joint] = base; continue; }
        quat turn;
        const float cosine = std::clamp(dot(up, best), -1.0F, 1.0F);
        if (cosine < -0.999999F) {
            // Straight down from straight up: half a turn about the parent's forward axis.
            const vec3 forward = q_rotate(base, {0.0F, 0.0F, 1.0F});
            turn = {forward.x, forward.y, forward.z, 0.0F};
        } else {
            const vec3 axis = cross(up, best);
            turn = q_norm({axis.x, axis.y, axis.z, 1.0F + cosine});
        }
        frames[joint] = q_norm(q_mul(turn, base));
    }
    return frames;
}

rig_kind identify_rig(const skeleton & value) {
    const auto has = [&](std::string_view wanted) {
        const auto canonical = canonical_joint_name(wanted);
        return std::any_of(value.names.begin(), value.names.end(), [&](const std::string & name) {
            return canonical_joint_name(name) == canonical;
        });
    };
    const auto linked = [&](std::string_view child_name, std::string_view parent_name) {
        std::size_t child = value.names.size(), parent = value.names.size();
        const auto child_key = canonical_joint_name(child_name), parent_key = canonical_joint_name(parent_name);
        for (std::size_t index = 0; index < value.names.size(); ++index) {
            const auto name = canonical_joint_name(value.names[index]);
            if (name == child_key) child = index;
            if (name == parent_key) parent = index;
        }
        return child < value.parents.size() && parent < value.names.size() &&
               value.parents[child] == static_cast<std::int32_t>(parent);
    };
    if (value.names.size() == 30U && has("Hips") && has("Chest") && has("LeftHandMiddleEnd") &&
        has("RightHandMiddleEnd") && has("LeftToeBase") && has("RightToeBase") &&
        linked("Spine1", "Hips") && linked("LeftForeArm", "LeftArm") &&
        linked("RightForeArm", "RightArm") && linked("LeftShin", "LeftLeg") &&
        linked("RightShin", "RightLeg")) return rig_kind::soma30;
    if (value.names.size() == 52U && has("Hips") && has("Spine") && has("LeftHandThumb1") &&
        has("RightHandPinky3") && has("LeftUpLeg") && has("RightToeBase") &&
        linked("Spine", "Hips") && linked("LeftForeArm", "LeftArm") &&
        linked("RightForeArm", "RightArm") && linked("LeftLeg", "LeftUpLeg") &&
        linked("RightLeg", "RightUpLeg")) return rig_kind::mixamo52;
    return rig_kind::unknown;
}

result<motion> load_skeleton_glb_file(const std::filesystem::path & path) {
    auto doc = read_document(path);
    if (!doc) return std::unexpected(doc.error());
    return load_skeleton_document(*doc, false);
}

result<motion> load_kimodo_glb_file(const std::filesystem::path & path) {
    auto doc = read_document(path);
    if (!doc) return std::unexpected(doc.error());
    return load_skeleton_document(*doc, true);
}

namespace {
result<skinned_asset> load_skinned_document(result<document> doc, result<mesh> geometry);
}

result<skinned_asset> load_skinned_glb_file(const std::filesystem::path & path) {
    return load_skinned_document(read_document(path), load_glb_file(path));
}

result<skinned_asset> load_skinned_glb_memory(std::span<const std::byte> bytes) {
    return load_skinned_document(parse_document(bytes), load_glb_memory(bytes));
}

namespace {
result<skinned_asset> load_skinned_document(result<document> doc, result<mesh> geometry) {
    if (!doc) return std::unexpected(doc.error());
    if (!geometry) return std::unexpected(geometry.error());
    auto animation = load_skeleton_document(*doc, false);
    if (!animation) return std::unexpected(animation.error());

    const auto skins = doc->root.find("skins");
    const auto nodes = doc->root.find("nodes");
    const auto meshes = doc->root.find("meshes");
    if (skins == doc->root.end() || !skins->is_array() || skins->size() != 1U ||
        nodes == doc->root.end() || !nodes->is_array() ||
        meshes == doc->root.end() || !meshes->is_array())
        return std::unexpected(detail::fail(error_code::invalid_format,
            "rigged GLB must contain exactly one skin and a bounded mesh"));
    const auto & skin_object = (*skins)[0];
    if (!skin_object.is_object() || !skin_object.contains("joints") ||
        !skin_object["joints"].is_array())
        return std::unexpected(detail::fail(error_code::invalid_format,
            "rigged GLB skin has no joint array"));

    // JOINTS_0 indexes the original skin.joints array. The skeleton loader
    // normalizes arbitrary glTF joint order so parents precede children; build
    // the same order here and explicitly remap every vertex influence.
    std::vector<std::size_t> original_nodes;
    std::vector<std::int32_t> node_to_original(nodes->size(), -1);
    for (const auto & value : skin_object["joints"]) {
        if (!value.is_number_unsigned() || value.get<std::size_t>() >= nodes->size())
            return std::unexpected(detail::fail(error_code::invalid_format,
                "rigged GLB contains an invalid skin joint"));
        const auto node = value.get<std::size_t>();
        if (node_to_original[node] >= 0)
            return std::unexpected(detail::fail(error_code::invalid_format,
                "rigged GLB contains duplicate skin joints"));
        node_to_original[node] = static_cast<std::int32_t>(original_nodes.size());
        original_nodes.push_back(node);
    }
    if (original_nodes.size() != animation->rig.names.size())
        return std::unexpected(detail::fail(error_code::invalid_format,
            "skin and loaded skeleton joint counts differ"));
    std::vector<std::int32_t> node_parents(nodes->size(), -1);
    for (std::size_t parent = 0; parent < nodes->size(); ++parent) {
        const auto children = (*nodes)[parent].find("children");
        if (children == (*nodes)[parent].end()) continue;
        if (!children->is_array())
            return std::unexpected(detail::fail(error_code::invalid_format,
                "rigged GLB node children are invalid"));
        for (const auto & child : *children) {
            if (!child.is_number_unsigned() || child.get<std::size_t>() >= nodes->size() ||
                node_parents[child.get<std::size_t>()] >= 0)
                return std::unexpected(detail::fail(error_code::invalid_format,
                    "rigged GLB joint hierarchy is invalid"));
            node_parents[child.get<std::size_t>()] = static_cast<std::int32_t>(parent);
        }
    }
    std::vector<std::size_t> ordered_nodes;
    std::vector<bool> added(nodes->size(), false);
    while (ordered_nodes.size() != original_nodes.size()) {
        const auto before = ordered_nodes.size();
        for (const auto node : original_nodes) {
            if (added[node]) continue;
            auto parent = node_parents[node];
            while (parent >= 0 && node_to_original[static_cast<std::size_t>(parent)] < 0)
                parent = node_parents[static_cast<std::size_t>(parent)];
            if (parent < 0 || added[static_cast<std::size_t>(parent)]) {
                ordered_nodes.push_back(node);
                added[node] = true;
            }
        }
        if (ordered_nodes.size() == before)
            return std::unexpected(detail::fail(error_code::invalid_format,
                "rigged GLB skin hierarchy contains a cycle"));
    }
    std::vector<std::uint16_t> original_to_ordered(original_nodes.size());
    for (std::size_t ordered = 0; ordered < ordered_nodes.size(); ++ordered)
        original_to_ordered[static_cast<std::size_t>(node_to_original[ordered_nodes[ordered]])] =
            static_cast<std::uint16_t>(ordered);

    // Every skinned primitive, in the order the mesh loader reads them, so influence i belongs to
    // vertex i of the loaded mesh (one primitive per material, one mesh per placed copy).
    struct skinned_part { accessor joints, weights; };
    std::vector<skinned_part> parts;
    std::size_t influence_count = 0;
    auto instances = collect_mesh_instances(*doc, *meshes);
    if (!instances) return std::unexpected(instances.error());
    for (const auto & instance : *instances) {
        if (instance.node == no_node) continue;
        const auto & node = (*nodes)[instance.node];
        for (const auto & candidate : (*meshes)[instance.mesh_index].value("primitives", json::array())) {
            if (!loaded_primitive(candidate)) continue;
            if (!node.contains("skin") || !node["skin"].is_number_unsigned() || node["skin"].get<std::size_t>() != 0U)
                return std::unexpected(detail::fail(error_code::invalid_format,
                    "rigged GLB places a mesh without the skin"));
            const auto & attributes = candidate["attributes"];
            for (const auto * name : {"JOINTS_0", "WEIGHTS_0"})
                if (!attributes.contains(name) || !attributes[name].is_number_unsigned())
                    return std::unexpected(detail::fail(error_code::invalid_format,
                        std::string{"rigged GLB primitive has no "} + name));
            auto positions = get_accessor(*doc, attributes["POSITION"].get<std::size_t>());
            auto joints = get_accessor(*doc, attributes["JOINTS_0"].get<std::size_t>());
            auto weights = get_accessor(*doc, attributes["WEIGHTS_0"].get<std::size_t>());
            if (!positions) return std::unexpected(positions.error());
            if (!joints) return std::unexpected(joints.error());
            if (!weights) return std::unexpected(weights.error());
            if (joints->count != positions->count || weights->count != positions->count)
                return std::unexpected(detail::fail(error_code::invalid_format,
                    "skinned attribute counts differ from mesh vertex count"));
            influence_count += positions->count;
            parts.push_back({*joints, *weights});
        }
    }
    if (parts.empty())
        return std::unexpected(detail::fail(error_code::invalid_format,
            "rigged GLB has no JOINTS_0/WEIGHTS_0 triangle primitive"));
    if (influence_count != geometry->vertices.size())
        return std::unexpected(detail::fail(error_code::invalid_format,
            "skinned attribute counts differ from mesh vertex count"));

    skinned_asset output;
    output.geometry = std::move(*geometry);
    output.animation = std::move(*animation);
    output.binding.rig = output.animation.rig;
    output.binding.joints.resize(influence_count);
    output.binding.weights.resize(influence_count);
    output.binding.learned = skin_object.value("name", std::string{}).find("learned") != std::string::npos;
    std::size_t vertex = 0;
    for (const auto & part : parts) {
        for (std::size_t i = 0; i < part.joints.count; ++i, ++vertex) {
            auto joint = joint_vector(part.joints, i);
            auto weight = weight_vector(part.weights, i);
            if (!joint) return std::unexpected(joint.error());
            if (!weight) return std::unexpected(weight.error());
            for (auto & index : *joint) {
                if (index >= original_to_ordered.size())
                    return std::unexpected(detail::fail(error_code::invalid_format,
                        "JOINTS_0 influence exceeds skin joint count"));
                index = original_to_ordered[index];
            }
            output.binding.joints[vertex] = *joint;
            output.binding.weights[vertex] = *weight;
        }
    }
    return output;
}
} // namespace

result<glb_info> inspect_glb_file(const std::filesystem::path & path) {
    auto doc = read_document(path);
    if (!doc) return std::unexpected(doc.error());
    glb_info output;
    const auto meshes = doc->root.find("meshes");
    const auto skins = doc->root.find("skins");
    output.has_mesh = meshes != doc->root.end() && meshes->is_array() && !meshes->empty();
    output.has_skin = skins != doc->root.end() && skins->is_array() && !skins->empty();
    auto skeleton = load_skeleton_document(*doc, false);
    if (skeleton) {
        output.has_skeleton = true;
        output.joint_count = skeleton->rig.names.size();
        output.frame_count = skeleton->frames;
        output.frames_per_second = skeleton->frames_per_second;
        output.has_animation = skeleton->frames > 1U;
        output.rig = identify_rig(skeleton->rig);
    } else if (output.has_skin) {
        return std::unexpected(skeleton.error());
    }
    return output;
}

namespace {
// Writes the builder's JSON and binary chunk as one GLB.
result<std::vector<std::byte>> pack_glb(glb_builder & builder) {
    std::string json_text = builder.root.dump();
    while (json_text.size() % 4U != 0U) json_text.push_back(' ');
    align4(builder.binary);
    const std::uint64_t total64 = 12ULL + 8ULL + json_text.size() + 8ULL + builder.binary.size();
    if (total64 > std::numeric_limits<std::uint32_t>::max())
        return std::unexpected(detail::fail(error_code::limit_exceeded, "exported GLB exceeds 4 GiB"));
    std::vector<std::byte> output;
    output.reserve(static_cast<std::size_t>(total64));
    const auto write = [&](const void * data, std::size_t bytes) {
        const auto * begin = static_cast<const std::byte *>(data);
        output.insert(output.end(), begin, begin + bytes);
    };
    const auto write_u32 = [&](std::uint32_t value) {
        if constexpr (std::endian::native == std::endian::big) value = std::byteswap(value);
        write(&value, sizeof(value));
    };
    write_u32(glb_magic); write_u32(2U); write_u32(static_cast<std::uint32_t>(total64));
    write_u32(static_cast<std::uint32_t>(json_text.size())); write_u32(json_chunk);
    write(json_text.data(), json_text.size());
    write_u32(static_cast<std::uint32_t>(builder.binary.size())); write_u32(bin_chunk);
    write(builder.binary.data(), builder.binary.size());
    return output;
}

// A one-frame clip is the rest pose, not a motion: give it Godot's rest-pose clip name.
std::string animation_name(const motion & animation) {
    if (!animation.name.empty()) return animation.name;
    return animation.frames <= 1U ? "RESET" : "Motion";
}
} // namespace

result<std::vector<std::byte>> save_skinned_animation_glb_memory(
    const mesh & geometry, const skin & binding, const motion & animation) {
    auto valid = validate_export(geometry, binding, animation);
    if (!valid) return std::unexpected(valid.error());

    glb_builder builder;
    const std::size_t joint_count = binding.rig.names.size();
    const auto position_minmax = [&] {
        vec3 low = geometry.vertices.front();
        vec3 high = geometry.vertices.front();
        for (const auto & value : geometry.vertices) {
            low.x = std::min(low.x, value.x); low.y = std::min(low.y, value.y); low.z = std::min(low.z, value.z);
            high.x = std::max(high.x, value.x); high.y = std::max(high.y, value.y); high.z = std::max(high.z, value.z);
        }
        return json{{"min", {low.x, low.y, low.z}}, {"max", {high.x, high.y, high.z}}};
    }();
    const auto positions = builder.accessor_for<vec3>(geometry.vertices, 5126U, "VEC3", 34962U, position_minmax);
    const auto normals = builder.accessor_for<vec3>(geometry.normals, 5126U, "VEC3", 34962U);
    std::optional<std::size_t> colors;
    if (!geometry.colors.empty())
        colors = builder.accessor_for<color4>(geometry.colors, 5126U, "VEC4", 34962U);
    const auto joints = builder.accessor_for<std::array<std::uint16_t, 4>>(binding.joints, 5123U, "VEC4", 34962U);
    const auto weights = builder.accessor_for<std::array<float, 4>>(binding.weights, 5126U, "VEC4", 34962U);
    std::vector<std::uint32_t> indices;
    indices.reserve(geometry.faces.size() * 3U);
    for (const auto & face : geometry.faces) indices.insert(indices.end(), face.begin(), face.end());
    const auto index_accessor = builder.accessor_for<std::uint32_t>(indices, 5125U, "SCALAR", 34963U);

    const std::vector<quat> frames = joint_rest_frames(binding.rig);
    std::vector<std::array<float, 16>> inverse_bind(joint_count);
    for (std::size_t joint = 0; joint < joint_count; ++joint)
        inverse_bind[joint] = framed_inverse_bind(binding.rig.rest_positions[joint], frames[joint]);
    const auto inverse_accessor = builder.accessor_for<std::array<float, 16>>(inverse_bind, 5126U, "MAT4");

    std::vector<float> times(animation.frames);
    const float fps = animation.frames_per_second > 0.0F ? animation.frames_per_second : 30.0F;
    for (std::size_t frame = 0; frame < animation.frames; ++frame) times[frame] = static_cast<float>(frame) / fps;
    const auto time_accessor = builder.accessor_for<float>(times, 5126U, "SCALAR", std::nullopt,
        json{{"min", {times.front()}}, {"max", {times.back()}}});

    for (std::size_t joint = 0; joint < joint_count; ++joint) {
        const vec3 local = rest_local_translation(binding.rig, frames, joint);
        const quat turn = rest_local_rotation(binding.rig, frames, joint);
        json node{{"name", binding.rig.names[joint]}, {"translation", {local.x, local.y, local.z}},
                  {"rotation", {turn.x, turn.y, turn.z, turn.w}}};
        json children = json::array();
        for (std::size_t child = 0; child < joint_count; ++child)
            if (binding.rig.parents[child] == static_cast<std::int32_t>(joint)) children.push_back(child);
        if (!children.empty()) node["children"] = std::move(children);
        builder.root["nodes"].push_back(std::move(node));
    }
    const std::size_t mesh_node = builder.root["nodes"].size();
    builder.root["nodes"].push_back({{"name", "SkinnedMesh"}, {"mesh", 0U}, {"skin", 0U}});
    const bool transparent = std::any_of(geometry.colors.begin(), geometry.colors.end(),
        [](const color4 & value) { return value.a < 0.999F; });
    json material{{"name", "SkinTokens material"},
        {"pbrMetallicRoughness", {{"baseColorFactor", colors ? json::array({1.0, 1.0, 1.0, 1.0}) :
                                                        json::array({0.55, 0.68, 0.78, 1.0})},
                                  {"metallicFactor", std::clamp(geometry.metallic_factor, 0.0F, 1.0F)},
                                  {"roughnessFactor", std::clamp(geometry.roughness_factor, 0.0F, 1.0F)}}},
        {"doubleSided", geometry.double_sided}};
    if (transparent) material["alphaMode"] = "BLEND";
    builder.root["materials"] = json::array({std::move(material)});
    json attributes{{"POSITION", positions}, {"NORMAL", normals},
                    {"JOINTS_0", joints}, {"WEIGHTS_0", weights}};
    if (colors) attributes["COLOR_0"] = *colors;
    builder.root["meshes"] = json::array({{{"name", "Imported mesh"}, {"primitives", json::array({{
        {"attributes", std::move(attributes)},
        {"indices", index_accessor}, {"material", 0U}, {"mode", 4U}
    }})}}});
    json skin_object{{"name", binding.learned ? "SkinTokens learned binding" : "Geometric baseline binding"},
                     {"inverseBindMatrices", inverse_accessor}, {"skeleton", 0U}, {"joints", json::array()}};
    for (std::size_t joint = 0; joint < joint_count; ++joint) skin_object["joints"].push_back(joint);
    builder.root["skins"] = json::array({std::move(skin_object)});

    json samplers = json::array();
    json channels = json::array();
    std::vector<vec3> root_values = animation.root_translations;
    const auto root_accessor = builder.accessor_for<vec3>(root_values, 5126U, "VEC3");
    samplers.push_back({{"input", time_accessor}, {"output", root_accessor}, {"interpolation", "LINEAR"}});
    channels.push_back({{"sampler", 0U}, {"target", {{"node", 0U}, {"path", "translation"}}}});
    for (std::size_t joint = 0; joint < joint_count; ++joint) {
        std::vector<quat> values(animation.frames);
        for (std::size_t frame = 0; frame < animation.frames; ++frame)
            values[frame] = framed_rotation(binding.rig, frames, joint, animation.local_rotations[frame * joint_count + joint]);
        const auto output_accessor = builder.accessor_for<quat>(values, 5126U, "VEC4");
        const auto sampler = samplers.size();
        samplers.push_back({{"input", time_accessor}, {"output", output_accessor}, {"interpolation", "LINEAR"}});
        channels.push_back({{"sampler", sampler}, {"target", {{"node", joint}, {"path", "rotation"}}}});
    }
    builder.root["animations"] = json::array({{{"name", animation_name(animation)},
        {"samplers", std::move(samplers)}, {"channels", std::move(channels)}}});
    builder.root["scenes"] = json::array({{{"nodes", {0U, mesh_node}}}});
    builder.root["scene"] = 0U;
    builder.root["buffers"].push_back({{"byteLength", builder.binary.size()}});
    return pack_glb(builder);
}

namespace {
vec3 linear(const matrix4 & m, vec3 v) {
    return {m[0]*v.x + m[4]*v.y + m[8]*v.z, m[1]*v.x + m[5]*v.y + m[9]*v.z, m[2]*v.x + m[6]*v.y + m[10]*v.z};
}
vec3 unit(vec3 v, vec3 fallback) {
    const float length = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    return length > 1e-12F ? vec3{v.x/length, v.y/length, v.z/length} : fallback;
}
bool is_identity(const matrix4 & m) { return m == identity_matrix(); }
json vec3_bounds(std::span<const vec3> values) {
    vec3 low{0.0F, 0.0F, 0.0F}, high{0.0F, 0.0F, 0.0F};
    if (!values.empty()) low = high = values.front();
    for (const auto & v : values) {
        low = {std::min(low.x, v.x), std::min(low.y, v.y), std::min(low.z, v.z)};
        high = {std::max(high.x, v.x), std::max(high.y, v.y), std::max(high.z, v.z)};
    }
    return {{"min", json::array({low.x, low.y, low.z})}, {"max", json::array({high.x, high.y, high.z})}};
}

// The rows of the transform's cofactor matrix: cofactor * n is the normal transform up to scale.
struct normal_transform {
    vec3 c0, c1, c2;
    float sign;
    vec3 operator()(vec3 n) const {
        return {sign*(c0.x*n.x + c1.x*n.y + c2.x*n.z), sign*(c0.y*n.x + c1.y*n.y + c2.y*n.z),
                sign*(c0.z*n.x + c1.z*n.y + c2.z*n.z)};
    }
};
normal_transform normal_matrix(const matrix4 & m) {
    const vec3 a0{m[0], m[1], m[2]}, a1{m[4], m[5], m[6]}, a2{m[8], m[9], m[10]};
    return {cross(a1, a2), cross(a2, a0), cross(a0, a1), determinant3(m) < 0.0F ? -1.0F : 1.0F};
}

// A float VEC3 accessor as dense values; sparse ones (blend-shape deltas) are expanded.
result<std::vector<vec3>> read_vec3(const document & doc, const json & index) {
    if (!index.is_number_unsigned())
        return std::unexpected(detail::fail(error_code::invalid_format, "glTF accessor index is invalid"));
    auto object = item(doc.root, "accessors", index.get<std::size_t>());
    if (!object) return std::unexpected(object.error());
    if ((**object).contains("sparse")) {
        const json & dense = **object;
        const json & sparse = dense["sparse"];
        const std::size_t count = dense.value("count", 0U);
        if (dense.value("componentType", 0U) != 5126U || dense.value("type", std::string{}) != "VEC3" ||
            count > max_vertices || !sparse.contains("indices") || !sparse.contains("values"))
            return std::unexpected(detail::fail(error_code::invalid_format, "sparse accessor must be float VEC3"));
        std::vector<vec3> output(count, vec3{0.0F, 0.0F, 0.0F});
        if (dense.contains("bufferView")) {
            json base = dense;
            base.erase("sparse");
            auto values = get_accessor_object(doc, base);
            if (!values) return std::unexpected(values.error());
            for (std::size_t i = 0; i < count; ++i) {
                auto value = vector3(*values, i);
                if (!value) return std::unexpected(value.error());
                output[i] = *value;
            }
        }
        const std::size_t changed = sparse.value("count", 0U);
        const json & ix = sparse["indices"];
        const json & vx = sparse["values"];
        json index_object{{"bufferView", ix.value("bufferView", 0U)}, {"byteOffset", ix.value("byteOffset", 0U)},
                          {"componentType", ix.value("componentType", 0U)}, {"count", changed}, {"type", "SCALAR"}};
        json value_object{{"bufferView", vx.value("bufferView", 0U)}, {"byteOffset", vx.value("byteOffset", 0U)},
                          {"componentType", 5126U}, {"count", changed}, {"type", "VEC3"}};
        auto indices = get_accessor_object(doc, index_object);
        auto values = get_accessor_object(doc, value_object);
        if (!indices) return std::unexpected(indices.error());
        if (!values) return std::unexpected(values.error());
        for (std::size_t i = 0; i < changed; ++i) {
            auto at = index_value(*indices, i);
            auto value = vector3(*values, i);
            if (!at) return std::unexpected(at.error());
            if (!value) return std::unexpected(value.error());
            if (*at >= count) return std::unexpected(detail::fail(error_code::invalid_format, "sparse index out of range"));
            output[*at] = *value;
        }
        return output;
    }
    if (!(**object).contains("bufferView")) {  // no data and no sparse part: all zeros
        if ((**object).value("componentType", 0U) != 5126U || (**object).value("type", std::string{}) != "VEC3" ||
            (**object).value("count", 0U) > max_vertices)
            return std::unexpected(detail::fail(error_code::invalid_format, "accessor must be float VEC3"));
        return std::vector<vec3>((**object).value("count", 0U), vec3{0.0F, 0.0F, 0.0F});
    }
    auto values = get_accessor(doc, index.get<std::size_t>());
    if (!values) return std::unexpected(values.error());
    std::vector<vec3> output(values->count);
    for (std::size_t i = 0; i < values->count; ++i) {
        auto value = vector3(*values, i);
        if (!value) return std::unexpected(value.error());
        output[i] = *value;
    }
    return output;
}
} // namespace

namespace {
// Drops the bytes nothing points at any more (a mesh replaced in place leaves its old vertex data
// behind). Indices stay valid: an accessor nobody uses loses its data (glTF reads that as zeros),
// a buffer view nobody uses shrinks to 4 bytes, and the binary chunk is packed again.
void drop_unused_bytes(glb_builder & builder) {
    json & root = builder.root;
    const std::size_t accessor_count = root["accessors"].size();
    const std::size_t view_count = root["bufferViews"].size();
    std::vector<std::uint8_t> used_accessor(accessor_count, 0U), used_view(view_count, 0U);
    const auto mark = [&](const json & value) {
        if (value.is_number_unsigned() && value.get<std::size_t>() < accessor_count) used_accessor[value.get<std::size_t>()] = 1U;
    };
    for (const auto & mesh_value : root["meshes"])
        for (const auto & primitive : mesh_value.value("primitives", json::array())) {
            for (const auto & [key, value] : primitive.value("attributes", json::object()).items()) mark(value);
            if (primitive.contains("indices")) mark(primitive["indices"]);
            for (const auto & target : primitive.value("targets", json::array()))
                for (const auto & [key, value] : target.items()) mark(value);
        }
    for (const auto & skin_value : root.value("skins", json::array()))
        if (skin_value.contains("inverseBindMatrices")) mark(skin_value["inverseBindMatrices"]);
    for (const auto & clip : root.value("animations", json::array()))
        for (const auto & sampler : clip.value("samplers", json::array())) {
            if (sampler.contains("input")) mark(sampler["input"]);
            if (sampler.contains("output")) mark(sampler["output"]);
        }
    for (const auto & node : root["nodes"])  // EXT_mesh_gpu_instancing
        if (auto ext = node.find("extensions"); ext != node.end() && ext->contains("EXT_mesh_gpu_instancing"))
            for (const auto & [key, value] : (*ext)["EXT_mesh_gpu_instancing"].value("attributes", json::object()).items()) mark(value);
    const auto mark_view = [&](const json & value) {
        if (value.is_number_unsigned() && value.get<std::size_t>() < view_count) used_view[value.get<std::size_t>()] = 1U;
    };
    for (std::size_t i = 0; i < accessor_count; ++i) {
        json & accessor_value = root["accessors"][i];
        if (!used_accessor[i]) {
            accessor_value = {{"componentType", 5126U}, {"count", 1U}, {"type", "SCALAR"}};
            continue;
        }
        if (accessor_value.contains("bufferView")) mark_view(accessor_value["bufferView"]);
        if (accessor_value.contains("sparse")) {
            mark_view(accessor_value["sparse"]["indices"].value("bufferView", json{}));
            mark_view(accessor_value["sparse"]["values"].value("bufferView", json{}));
        }
    }
    // Any other "bufferView" (images, compression extensions) is kept.
    std::function<void(const json &)> walk = [&](const json & value) {
        if (value.is_object()) {
            for (const auto & [key, child] : value.items()) {
                if (key == "bufferView") mark_view(child);
                else walk(child);
            }
        } else if (value.is_array()) {
            for (const auto & child : value) walk(child);
        }
    };
    for (const auto & [key, value] : root.items())
        if (key != "accessors" && key != "bufferViews") walk(value);
    std::vector<std::byte> packed;
    packed.resize(4U);  // the shared stand-in for unused views
    for (std::size_t i = 0; i < view_count; ++i) {
        json & view = root["bufferViews"][i];
        if (view.value("buffer", 0U) != 0U) continue;
        if (!used_view[i]) {
            view = {{"buffer", 0U}, {"byteOffset", 0U}, {"byteLength", 4U}};
            continue;
        }
        const std::size_t offset = view.value("byteOffset", 0U), length = view.value("byteLength", 0U);
        if (offset > builder.binary.size() || length > builder.binary.size() - offset) continue;
        while (packed.size() % 4U != 0U) packed.push_back(std::byte{});
        view["byteOffset"] = packed.size();
        packed.insert(packed.end(), builder.binary.begin() + static_cast<std::ptrdiff_t>(offset),
                      builder.binary.begin() + static_cast<std::ptrdiff_t>(offset + length));
    }
    builder.binary = std::move(packed);
}

// Calls visit on each node index an extension holds: "node" anywhere in an extension (VRM 1.0
// humanoid, spring bones, VRM 0.x humanoid), "source" in VRMC_node_constraint and VRM 0.x
// secondaryAnimation "bones". Other "source" keys name images (EXT_texture_webp), not nodes.
void for_each_extension_node(json & value, std::string_view extension, const std::function<void(json &)> & visit) {
    if (value.is_object()) {
        for (auto & [key, child] : value.items()) {
            if (key == "extensions" && child.is_object()) {
                for (auto & [name, body] : child.items()) for_each_extension_node(body, name, visit);
                continue;
            }
            if (!extension.empty() && child.is_number_unsigned() &&
                (key == "node" || (key == "source" && extension == "VRMC_node_constraint"))) {
                visit(child);
                continue;
            }
            if (!extension.empty() && key == "bones" && extension == "VRM" && child.is_array()) {
                for (auto & bone : child) if (bone.is_number_unsigned()) visit(bone);
                continue;
            }
            for_each_extension_node(child, extension, visit);
        }
    } else if (value.is_array()) {
        for (auto & child : value) for_each_extension_node(child, extension, visit);
    }
}

// Removes each replaced node with nothing kept below it, and renumbers every node reference.
// A replaced node stays when it holds a mesh, camera, skin or extension, when an extension names
// it, or when a kept node is below it. Returns the new index of each old node (removed: npos).
std::vector<std::size_t> drop_replaced_nodes(json & root, const std::vector<std::uint8_t> & replaced) {
    json & nodes = root["nodes"];
    const std::size_t count = nodes.size();
    constexpr std::size_t gone = std::numeric_limits<std::size_t>::max();
    std::vector<std::uint8_t> named(count, 0U);
    for_each_extension_node(root, {}, [&](json & index) {
        if (index.get<std::size_t>() < count) named[index.get<std::size_t>()] = 1U;
    });
    std::vector<std::uint8_t> keep(count, 2U);  // 2: not decided yet
    std::function<bool(std::size_t)> kept = [&](std::size_t n) -> bool {
        if (keep[n] != 2U) return keep[n] != 0U;
        keep[n] = 1U;  // a cycle (invalid glTF) keeps its nodes
        const json & node = nodes[n];
        bool hold = n >= replaced.size() || replaced[n] == 0U || named[n] != 0U || node.contains("mesh") ||
                    node.contains("camera") || node.contains("skin") || node.contains("extensions");
        for (const auto & child : node.value("children", json::array()))
            if (child.is_number_unsigned() && child.get<std::size_t>() < count && kept(child.get<std::size_t>())) hold = true;
        keep[n] = hold ? 1U : 0U;
        return hold;
    };
    std::vector<std::size_t> index_of(count, gone);
    std::size_t next = 0;
    for (std::size_t n = 0; n < count; ++n)
        if (kept(n)) index_of[n] = next++;
    if (next == count) return index_of;
    const auto renumber = [&](json & list) {
        json out = json::array();
        for (const auto & value : list)
            if (value.is_number_unsigned() && value.get<std::size_t>() < count && index_of[value.get<std::size_t>()] != gone)
                out.push_back(index_of[value.get<std::size_t>()]);
        list = std::move(out);
    };
    json out = json::array();
    for (std::size_t n = 0; n < count; ++n) {
        if (index_of[n] == gone) continue;
        json node = std::move(nodes[n]);
        if (node.contains("children")) {
            renumber(node["children"]);
            if (node["children"].empty()) node.erase("children");
        }
        out.push_back(std::move(node));
    }
    nodes = std::move(out);
    for (auto & scene : root["scenes"]) if (scene.contains("nodes")) renumber(scene["nodes"]);
    if (root.contains("skins"))
        for (auto & skin_value : root["skins"]) {
            if (skin_value.contains("joints")) renumber(skin_value["joints"]);
            if (skin_value.contains("skeleton") && skin_value["skeleton"].is_number_unsigned())
                skin_value["skeleton"] = index_of[skin_value["skeleton"].get<std::size_t>()];
        }
    if (root.contains("animations"))
        for (auto & clip : root["animations"])
            for (auto & channel : clip["channels"])
                if (channel["target"].contains("node"))
                    channel["target"]["node"] = index_of[channel["target"]["node"].get<std::size_t>()];
    for_each_extension_node(root, {}, [&](json & index) {
        if (index.get<std::size_t>() < count) index = index_of[index.get<std::size_t>()];
    });
    return index_of;
}
} // namespace

result<std::vector<std::byte>> save_rigged_source_glb_memory(
    std::span<const std::byte> source, const skin & binding, const motion & animation) {
    auto geometry = load_glb_memory(source);
    if (!geometry) return std::unexpected(geometry.error());
    auto valid = validate_export(*geometry, binding, animation);
    if (!valid) return std::unexpected(valid.error());
    auto doc = parse_document(source);
    if (!doc) return std::unexpected(doc.error());

    glb_builder builder;
    builder.root = doc->root;
    builder.binary = doc->binary;
    json & root = builder.root;
    if (!root.contains("buffers") || !root["buffers"].is_array() || root["buffers"].empty())
        root["buffers"] = json::array({json::object()});
    if (root["buffers"][0].contains("uri"))
        return std::unexpected(detail::fail(error_code::invalid_format, "GLB buffer 0 is not the binary chunk"));
    for (const char * name : {"bufferViews", "accessors", "nodes", "meshes"})
        if (!root.contains(name)) root[name] = json::array();
    std::size_t scene_index = root.value("scene", 0U);
    if (!root.contains("scenes") || !root["scenes"].is_array() || scene_index >= root["scenes"].size())
        return std::unexpected(detail::fail(error_code::invalid_format, "GLB has no scene to rig"));

    const json source_meshes = doc->root["meshes"];
    auto collected = collect_mesh_instances(*doc, source_meshes);
    if (!collected) return std::unexpected(collected.error());
    std::vector<std::size_t> uses(source_meshes.size(), 0U);
    for (const auto & instance : *collected) ++uses[instance.mesh_index];

    // The source's skins and clips drive its old skeleton, which no longer moves these vertices.
    root.erase("skins");
    root.erase("animations");
    for (auto & node : root["nodes"]) node.erase("skin");

    const std::size_t joint_count = binding.rig.names.size();
    const std::size_t joint_base = root["nodes"].size();
    json pending = json::array();
    std::vector<std::uint8_t> node_done(root["nodes"].size(), 0U);
    std::uint32_t base = 0U;
    for (const auto & instance : *collected) {
        const json & source_mesh = source_meshes[instance.mesh_index];
        const matrix4 & world = instance.transform;
        // Only the 3x3 part turns normals, tangents and blend-shape deltas; a pure move leaves them.
        matrix4 linear_part = world;
        linear_part[12] = linear_part[13] = linear_part[14] = 0.0F;
        const bool turned = !is_identity(linear_part);
        const bool mirrored = determinant3(world) < 0.0F;
        const auto normals = normal_matrix(world);
        json skinned_primitives = json::array();
        json kept_primitives = json::array();
        for (const auto & primitive : source_mesh.value("primitives", json::array())) {
            if (!loaded_primitive(primitive)) { kept_primitives.push_back(primitive); continue; }
            json out = primitive;
            json & attributes = out["attributes"];
            auto count = get_accessor(*doc, attributes["POSITION"].get<std::size_t>());
            if (!count) return std::unexpected(count.error());
            const std::size_t n = count->count;
            if (base + n > geometry->vertices.size())
                return std::unexpected(detail::fail(error_code::invalid_format, "source walk does not match the loaded mesh"));
            const std::span<const vec3> positions{geometry->vertices.data() + base, n};
            json bounds{{"min", json::array({positions[0].x, positions[0].y, positions[0].z})},
                        {"max", json::array({positions[0].x, positions[0].y, positions[0].z})}};
            for (const auto & v : positions)
                for (std::size_t axis = 0; axis < 3U; ++axis) {
                    const float value = axis == 0U ? v.x : axis == 1U ? v.y : v.z;
                    bounds["min"][axis] = std::min(bounds["min"][axis].get<float>(), value);
                    bounds["max"][axis] = std::max(bounds["max"][axis].get<float>(), value);
                }
            attributes["POSITION"] = builder.accessor_for<vec3>(positions, 5126U, "VEC3", 34962U, bounds);
            if (turned && attributes.contains("NORMAL")) {
                auto values = read_vec3(*doc, attributes["NORMAL"]);
                if (!values) return std::unexpected(values.error());
                for (auto & v : *values) v = unit(normals(v), {0.0F, 1.0F, 0.0F});
                attributes["NORMAL"] = builder.accessor_for<vec3>(*values, 5126U, "VEC3", 34962U);
            }
            if (turned && attributes.contains("TANGENT")) {
                auto tangent = get_accessor(*doc, attributes["TANGENT"].get<std::size_t>());
                if (!tangent) return std::unexpected(tangent.error());
                std::vector<quat> values(tangent->count);
                for (std::size_t i = 0; i < tangent->count; ++i) {
                    auto t = quaternion(*tangent, i);
                    if (!t) return std::unexpected(t.error());
                    const vec3 xyz = unit(linear(world, {t->x, t->y, t->z}), {1.0F, 0.0F, 0.0F});
                    values[i] = {xyz.x, xyz.y, xyz.z, mirrored ? -t->w : t->w};
                }
                attributes["TANGENT"] = builder.accessor_for<quat>(values, 5126U, "VEC4", 34962U);
            }
            for (const char * name : {"JOINTS_1", "WEIGHTS_1", "JOINTS_2", "WEIGHTS_2", "JOINTS_3", "WEIGHTS_3"})
                attributes.erase(name);
            attributes["JOINTS_0"] = builder.accessor_for<std::array<std::uint16_t, 4>>(
                std::span<const std::array<std::uint16_t, 4>>{binding.joints.data() + base, n}, 5123U, "VEC4", 34962U);
            attributes["WEIGHTS_0"] = builder.accessor_for<std::array<float, 4>>(
                std::span<const std::array<float, 4>>{binding.weights.data() + base, n}, 5126U, "VEC4", 34962U);
            if (turned && out.contains("targets")) {
                const float det = std::max(std::abs(determinant3(world)), 1e-20F);
                for (auto & target : out["targets"]) {
                    for (const char * name : {"POSITION", "NORMAL", "TANGENT"}) {
                        if (!target.contains(name) || !target[name].is_number_unsigned()) continue;
                        const json source_accessor = root["accessors"][target[name].get<std::size_t>()];
                        if (!source_accessor.contains("bufferView") && !source_accessor.contains("sparse"))
                            continue;  // all zeros stay zeros
                        // Deltas keep their length: normals go through the inverse transpose.
                        const bool normal = std::string_view{name} == "NORMAL";
                        const auto move = [&](vec3 v) {
                            if (!normal) return linear(world, v);
                            v = normals(v);
                            return vec3{v.x/det, v.y/det, v.z/det};
                        };
                        if (source_accessor.contains("sparse") && !source_accessor.contains("bufferView")) {
                            // Keep it sparse: same indices, moved values.
                            const json & sparse = source_accessor["sparse"];
                            const std::size_t changed = sparse.value("count", 0U);
                            json value_object{{"bufferView", sparse["values"].value("bufferView", 0U)},
                                              {"byteOffset", sparse["values"].value("byteOffset", 0U)},
                                              {"componentType", 5126U}, {"count", changed}, {"type", "VEC3"}};
                            auto values = get_accessor_object(*doc, value_object);
                            if (!values) return std::unexpected(values.error());
                            std::vector<vec3> moved_values(changed);
                            for (std::size_t i = 0; i < changed; ++i) {
                                auto value = vector3(*values, i);
                                if (!value) return std::unexpected(value.error());
                                moved_values[i] = move(*value);
                            }
                            const auto packed = builder.accessor_for<vec3>(moved_values, 5126U, "VEC3");
                            json accessor_value = source_accessor;
                            // The dense values are the moved deltas plus zeros where nothing moves.
                            std::vector<vec3> with_zero = moved_values;
                            with_zero.push_back({0.0F, 0.0F, 0.0F});
                            accessor_value.update(vec3_bounds(with_zero));
                            accessor_value["sparse"]["values"] = {{"bufferView", root["accessors"][packed]["bufferView"]}};
                            target[name] = root["accessors"].size();
                            root["accessors"].push_back(std::move(accessor_value));
                            continue;
                        }
                        auto values = read_vec3(*doc, target[name]);
                        if (!values) return std::unexpected(values.error());
                        for (auto & v : *values) v = move(v);
                        target[name] = builder.accessor_for<vec3>(*values, 5126U, "VEC3", 34962U,
                            std::string_view{name} == "POSITION" ? vec3_bounds(*values) : json{});
                    }
                }
            }
            if (mirrored) {
                auto indices = get_accessor(*doc, primitive["indices"].get<std::size_t>());
                if (!indices) return std::unexpected(indices.error());
                std::vector<std::uint32_t> flipped(indices->count);
                for (std::size_t i = 0; i < indices->count; ++i) {
                    auto value = index_value(*indices, i);
                    if (!value) return std::unexpected(value.error());
                    flipped[i] = *value;
                }
                for (std::size_t i = 0; i + 2U < flipped.size(); i += 3U) std::swap(flipped[i + 1U], flipped[i + 2U]);
                out["indices"] = builder.accessor_for<std::uint32_t>(flipped, 5125U, "SCALAR", 34963U);
            }
            skinned_primitives.push_back(std::move(out));
            base += static_cast<std::uint32_t>(n);
        }
        if (skinned_primitives.empty()) continue;  // nothing the rig read: the node stays as it was
        json skinned_mesh = source_mesh;
        skinned_mesh["primitives"] = std::move(skinned_primitives);
        std::size_t mesh_slot = instance.mesh_index;
        if (uses[instance.mesh_index] == 1U && kept_primitives.empty()) {
            root["meshes"][mesh_slot] = std::move(skinned_mesh);
        } else {
            mesh_slot = root["meshes"].size();
            root["meshes"].push_back(std::move(skinned_mesh));
        }
        std::string name = source_mesh.value("name", std::string{"mesh"});
        if (instance.node != no_node) {
            json & node = root["nodes"][instance.node];
            name = node.value("name", name);
            if (node_done[instance.node] == 0U) {
                node_done[instance.node] = 1U;
                if (kept_primitives.empty()) {
                    node.erase("mesh");
                    node.erase("weights");
                } else {
                    json rest = source_mesh;
                    rest["primitives"] = std::move(kept_primitives);
                    node["mesh"] = root["meshes"].size();
                    root["meshes"].push_back(std::move(rest));
                }
            }
        }
        json skinned_node{{"name", name}, {"mesh", mesh_slot}, {"skin", 0U}};
        if (instance.node != no_node && doc->root["nodes"][instance.node].contains("weights"))
            skinned_node["weights"] = doc->root["nodes"][instance.node]["weights"];
        pending.push_back(std::move(skinned_node));
    }
    if (base != geometry->vertices.size())
        return std::unexpected(detail::fail(error_code::invalid_format, "source walk does not match the loaded mesh"));

    const std::vector<quat> frames = joint_rest_frames(binding.rig);
    for (std::size_t joint = 0; joint < joint_count; ++joint) {
        const vec3 local = rest_local_translation(binding.rig, frames, joint);
        const quat turn = rest_local_rotation(binding.rig, frames, joint);
        json node{{"name", binding.rig.names[joint]}, {"translation", {local.x, local.y, local.z}},
                  {"rotation", {turn.x, turn.y, turn.z, turn.w}}};
        json children = json::array();
        for (std::size_t child = 0; child < joint_count; ++child)
            if (binding.rig.parents[child] == static_cast<std::int32_t>(joint)) children.push_back(joint_base + child);
        if (!children.empty()) node["children"] = std::move(children);
        root["nodes"].push_back(std::move(node));
    }
    std::vector<std::size_t> placed;
    for (auto & node : pending) { placed.push_back(root["nodes"].size()); root["nodes"].push_back(std::move(node)); }

    std::vector<std::array<float, 16>> inverse_bind(joint_count);
    for (std::size_t joint = 0; joint < joint_count; ++joint)
        inverse_bind[joint] = framed_inverse_bind(binding.rig.rest_positions[joint], frames[joint]);
    json skin_object{{"name", binding.learned ? "SkinTokens learned binding" : "Geometric baseline binding"},
                     {"inverseBindMatrices", builder.accessor_for<std::array<float, 16>>(inverse_bind, 5126U, "MAT4")},
                     {"skeleton", joint_base}, {"joints", json::array()}};
    for (std::size_t joint = 0; joint < joint_count; ++joint) skin_object["joints"].push_back(joint_base + joint);
    root["skins"] = json::array({std::move(skin_object)});

    std::vector<float> times(animation.frames);
    const float fps = animation.frames_per_second > 0.0F ? animation.frames_per_second : 30.0F;
    for (std::size_t frame = 0; frame < animation.frames; ++frame) times[frame] = static_cast<float>(frame) / fps;
    const auto time_accessor = builder.accessor_for<float>(times, 5126U, "SCALAR", std::nullopt,
        json{{"min", {times.front()}}, {"max", {times.back()}}});
    json samplers = json::array();
    json channels = json::array();
    samplers.push_back({{"input", time_accessor}, {"output", builder.accessor_for<vec3>(animation.root_translations, 5126U, "VEC3")},
                        {"interpolation", "LINEAR"}});
    channels.push_back({{"sampler", 0U}, {"target", {{"node", joint_base}, {"path", "translation"}}}});
    for (std::size_t joint = 0; joint < joint_count; ++joint) {
        std::vector<quat> values(animation.frames);
        for (std::size_t frame = 0; frame < animation.frames; ++frame)
            values[frame] = framed_rotation(binding.rig, frames, joint, animation.local_rotations[frame * joint_count + joint]);
        const auto sampler = samplers.size();
        samplers.push_back({{"input", time_accessor}, {"output", builder.accessor_for<quat>(values, 5126U, "VEC4")},
                            {"interpolation", "LINEAR"}});
        channels.push_back({{"sampler", sampler}, {"target", {{"node", joint_base + joint}, {"path", "rotation"}}}});
    }
    root["animations"] = json::array({{{"name", animation_name(animation)},
        {"samplers", std::move(samplers)}, {"channels", std::move(channels)}}});

    // The new skin replaces the source's old skeleton. Its joints and the emptied mesh nodes would
    // stay as unused nodes, and a tool that reads every node under a skeleton makes each one a bone.
    std::vector<std::uint8_t> replaced(joint_base, 0U);
    if (const auto old_skins = doc->root.find("skins"); old_skins != doc->root.end() && old_skins->is_array())
        for (const auto & old_skin : *old_skins)
            for (const auto & joint : old_skin.value("joints", json::array()))
                if (joint.is_number_unsigned() && joint.get<std::size_t>() < joint_base)
                    replaced[joint.get<std::size_t>()] = 1U;
    for (std::size_t n = 0; n < joint_base; ++n)
        if (node_done[n] != 0U && !root["nodes"][n].contains("mesh")) replaced[n] = 1U;
    const auto index_of = drop_replaced_nodes(root, replaced);
    const std::size_t new_joint_base = index_of[joint_base];
    for (auto & index : placed) index = index_of[index];
    // Godot treats a scene with one root differently from one with several: when the source has a
    // single root, the skeleton and skinned meshes go inside it, so importers keep that root.
    // Only a root without a transform can hold them: the joints and vertices are in world space.
    json & scene = root["scenes"][scene_index];
    if (!scene.contains("nodes")) scene["nodes"] = json::array();
    json * holder = &scene["nodes"];
    if (scene["nodes"].size() == 1U && scene["nodes"][0].is_number_unsigned()) {
        json & single = root["nodes"][scene["nodes"][0].get<std::size_t>()];
        auto local = node_matrix(single);
        if (local && is_identity(*local) && !single.contains("mesh") && !single.contains("skin")) {
            if (!single.contains("children")) single["children"] = json::array();
            holder = &single["children"];
        }
    }
    holder->push_back(new_joint_base);
    for (auto index : placed) holder->push_back(index);
    drop_unused_bytes(builder);
    align4(builder.binary);
    root["buffers"][0]["byteLength"] = builder.binary.size();
    return pack_glb(builder);
}

result<std::vector<std::byte>> add_animation_glb_memory(
    std::span<const std::byte> rigged, const motion & animation) {
    auto doc = parse_document(rigged);
    if (!doc) return std::unexpected(doc.error());
    glb_builder builder;
    builder.root = doc->root;
    builder.binary = doc->binary;
    json & root = builder.root;
    if (!root.contains("skins") || !root["skins"].is_array() || root["skins"].size() != 1U ||
        !root["skins"][0].contains("joints") || !root.contains("buffers") || root["buffers"].empty() ||
        root["buffers"][0].contains("uri"))
        return std::unexpected(detail::fail(error_code::invalid_format, "rigged GLB must hold one skin and a binary chunk"));
    const std::size_t joint_count = animation.rig.names.size();
    if (animation.frames == 0U || animation.root_translations.size() != animation.frames ||
        animation.local_rotations.size() != animation.frames * joint_count ||
        root["skins"][0]["joints"].size() != joint_count)
        return std::unexpected(detail::fail(error_code::invalid_argument, "animation does not match the skin skeleton"));
    // The clip's joints are the skin's, found by name (the skeleton loader names them the same way).
    std::vector<std::size_t> targets(joint_count);
    for (std::size_t joint = 0; joint < joint_count; ++joint) {
        std::size_t found = no_node;
        for (std::size_t slot = 0; slot < joint_count; ++slot) {
            const std::size_t node = root["skins"][0]["joints"][slot].get<std::size_t>();
            if (root["nodes"][node].value("name", "joint_" + std::to_string(slot)) != animation.rig.names[joint]) continue;
            if (found != no_node)
                return std::unexpected(detail::fail(error_code::invalid_format, "skin joint names are not unique"));
            found = node;
        }
        if (found == no_node)
            return std::unexpected(detail::fail(error_code::invalid_argument, "clip joint is not in the skin: " + animation.rig.names[joint]));
        targets[joint] = found;
    }
    // Each node's world rest rotation: the clip's rotations are turned into the joints' own frames.
    std::vector<std::int32_t> node_parent(root["nodes"].size(), -1);
    for (std::size_t node = 0; node < root["nodes"].size(); ++node)
        for (const auto & child : root["nodes"][node].value("children", json::array()))
            if (child.is_number_unsigned() && child.get<std::size_t>() < node_parent.size())
                node_parent[child.get<std::size_t>()] = static_cast<std::int32_t>(node);
    std::function<quat(std::int32_t, int)> world_rest = [&](std::int32_t node, int depth) -> quat {
        if (node < 0 || depth > 4096) return quat{};
        auto local = node_matrix(root["nodes"][static_cast<std::size_t>(node)]);
        const quat own = local ? matrix_rotation(*local) : quat{};
        return q_norm(q_mul(world_rest(node_parent[static_cast<std::size_t>(node)], depth + 1), own));
    };
    std::vector<quat> own_frame(joint_count), parent_frame(joint_count);
    for (std::size_t joint = 0; joint < joint_count; ++joint) {
        own_frame[joint] = world_rest(static_cast<std::int32_t>(targets[joint]), 0);
        parent_frame[joint] = world_rest(node_parent[targets[joint]], 0);
    }
    std::vector<float> times(animation.frames);
    const float fps = animation.frames_per_second > 0.0F ? animation.frames_per_second : 30.0F;
    for (std::size_t frame = 0; frame < animation.frames; ++frame) times[frame] = static_cast<float>(frame) / fps;
    const auto time_accessor = builder.accessor_for<float>(times, 5126U, "SCALAR", std::nullopt,
        json{{"min", {times.front()}}, {"max", {times.back()}}});
    json samplers = json::array();
    json channels = json::array();
    samplers.push_back({{"input", time_accessor}, {"output", builder.accessor_for<vec3>(animation.root_translations, 5126U, "VEC3")},
                        {"interpolation", "LINEAR"}});
    channels.push_back({{"sampler", 0U}, {"target", {{"node", targets[0]}, {"path", "translation"}}}});
    for (std::size_t joint = 0; joint < joint_count; ++joint) {
        std::vector<quat> values(animation.frames);
        for (std::size_t frame = 0; frame < animation.frames; ++frame)
            values[frame] = q_norm(q_mul(q_mul(q_conj(parent_frame[joint]),
                animation.local_rotations[frame * joint_count + joint]), own_frame[joint]));
        const auto sampler = samplers.size();
        samplers.push_back({{"input", time_accessor}, {"output", builder.accessor_for<quat>(values, 5126U, "VEC4")},
                            {"interpolation", "LINEAR"}});
        channels.push_back({{"sampler", sampler}, {"target", {{"node", targets[joint]}, {"path", "rotation"}}}});
    }
    const std::string name = animation_name(animation);
    // The new clip goes first: readers that take animations[0] (load_kimodo_glb_file) get the motion.
    json clips = json::array({{{"name", name}, {"samplers", std::move(samplers)}, {"channels", std::move(channels)}}});
    for (auto & clip : root.value("animations", json::array()))
        if (clip.value("name", std::string{}) != name) clips.push_back(clip);
    root["animations"] = std::move(clips);
    align4(builder.binary);
    root["buffers"][0]["byteLength"] = builder.binary.size();
    return pack_glb(builder);
}

namespace {
matrix4 rigid_matrix(vec3 t, quat q) {
    const vec3 x = q_rotate(q, {1, 0, 0}), y = q_rotate(q, {0, 1, 0}), z = q_rotate(q, {0, 0, 1});
    return {x.x, x.y, x.z, 0, y.x, y.y, y.z, 0, z.x, z.y, z.z, 0, t.x, t.y, t.z, 1};
}
matrix4 rigid_inverse(const matrix4 & m) {
    // m = [R | t] with R a rotation: inverse = [R^T | -R^T t]
    matrix4 out{m[0], m[4], m[8], 0, m[1], m[5], m[9], 0, m[2], m[6], m[10], 0, 0, 0, 0, 1};
    out[12] = -(out[0]*m[12] + out[4]*m[13] + out[8]*m[14]);
    out[13] = -(out[1]*m[12] + out[5]*m[13] + out[9]*m[14]);
    out[14] = -(out[2]*m[12] + out[6]*m[13] + out[10]*m[14]);
    return out;
}
matrix4 general_inverse(const matrix4 & m) {
    // affine inverse: [A | t] -> [A^-1 | -A^-1 t]
    const float a = m[0], b = m[4], c = m[8], d = m[1], e = m[5], f = m[9], g = m[2], h = m[6], i = m[10];
    const float det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
    const float k = std::abs(det) > 1e-30F ? 1.0F/det : 0.0F;
    matrix4 out{(e*i - f*h)*k, (f*g - d*i)*k, (d*h - e*g)*k, 0, (c*h - b*i)*k, (a*i - c*g)*k, (b*g - a*h)*k, 0,
                (b*f - c*e)*k, (c*d - a*f)*k, (a*e - b*d)*k, 0, 0, 0, 0, 1};
    out[12] = -(out[0]*m[12] + out[4]*m[13] + out[8]*m[14]);
    out[13] = -(out[1]*m[12] + out[5]*m[13] + out[9]*m[14]);
    out[14] = -(out[2]*m[12] + out[6]*m[13] + out[10]*m[14]);
    return out;
}
void write_trs(json & node, const matrix4 & m) {
    const vec3 c0{m[0], m[1], m[2]}, c1{m[4], m[5], m[6]}, c2{m[8], m[9], m[10]};
    const auto len = [](vec3 v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); };
    const quat r = matrix_rotation(m);
    node.erase("matrix");
    node["translation"] = {m[12], m[13], m[14]};
    node["rotation"] = {r.x, r.y, r.z, r.w};
    const float s0 = len(c0), s1 = len(c1), s2 = len(c2);
    if (std::abs(s0 - 1.0F) > 1e-5F || std::abs(s1 - 1.0F) > 1e-5F || std::abs(s2 - 1.0F) > 1e-5F)
        node["scale"] = {s0, s1, s2};
    else
        node.erase("scale");
}
} // namespace

result<std::vector<std::byte>> reframe_skin_joints_glb_memory(std::span<const std::byte> bytes) {
    auto doc = parse_document(bytes);
    if (!doc) return std::unexpected(doc.error());
    glb_builder builder;
    builder.root = doc->root;
    builder.binary = doc->binary;
    json & root = builder.root;
    if (!root.contains("skins") || !root["skins"].is_array() || root["skins"].empty() || !root.contains("nodes"))
        return std::unexpected(detail::fail(error_code::invalid_format, "GLB has no skin to re-frame"));
    if (!root.contains("buffers") || root["buffers"].empty() || root["buffers"][0].contains("uri"))
        return std::unexpected(detail::fail(error_code::invalid_format, "GLB has no binary chunk"));
    json & nodes = root["nodes"];
    const std::size_t count = nodes.size();
    std::vector<std::int32_t> parent(count, -1);
    for (std::size_t n = 0; n < count; ++n)
        for (const auto & child : nodes[n].value("children", json::array()))
            if (child.is_number_unsigned() && child.get<std::size_t>() < count) parent[child.get<std::size_t>()] = static_cast<std::int32_t>(n);
    std::vector<matrix4> world(count);
    std::vector<std::uint8_t> state(count, 0U);
    std::function<result<matrix4>(std::size_t)> world_of = [&](std::size_t n) -> result<matrix4> {
        if (state[n] == 2U) return world[n];
        if (state[n] == 1U) return std::unexpected(detail::fail(error_code::invalid_format, "node hierarchy contains a cycle"));
        state[n] = 1U;
        auto local = node_matrix(nodes[n]);
        if (!local) return std::unexpected(local.error());
        if (parent[n] < 0) world[n] = *local;
        else {
            auto up = world_of(static_cast<std::size_t>(parent[n]));
            if (!up) return std::unexpected(up.error());
            world[n] = multiply(*up, *local);
        }
        state[n] = 2U;
        return world[n];
    };
    for (std::size_t n = 0; n < count; ++n) if (auto w = world_of(n); !w) return std::unexpected(w.error());

    std::vector<std::uint8_t> is_joint(count, 0U), keep_frame(count, 0U);
    for (const auto & skin_value : root["skins"])
        for (const auto & joint : skin_value.value("joints", json::array()))
            if (joint.is_number_unsigned() && joint.get<std::size_t>() < count) is_joint[joint.get<std::size_t>()] = 1U;
    // A node constraint states its axes in the node's own frame: those nodes keep their frame.
    for (std::size_t n = 0; n < count; ++n) {
        auto ext = nodes[n].find("extensions");
        if (ext == nodes[n].end() || !ext->contains("VRMC_node_constraint")) continue;
        keep_frame[n] = 1U;
        const json & constraint = (*ext)["VRMC_node_constraint"].value("constraint", json::object());
        for (const auto & [kind, body] : constraint.items())
            if (body.contains("source") && body["source"].is_number_unsigned() && body["source"].get<std::size_t>() < count)
                keep_frame[body["source"].get<std::size_t>()] = 1U;
    }
    // The joints as a skeleton, parents first, each joint's parent the nearest joint above it.
    std::vector<std::size_t> order;
    std::vector<std::int32_t> slot(count, -1);
    std::function<void(std::size_t)> visit = [&](std::size_t n) {
        if (is_joint[n]) { slot[n] = static_cast<std::int32_t>(order.size()); order.push_back(n); }
        for (const auto & child : nodes[n].value("children", json::array()))
            if (child.is_number_unsigned() && child.get<std::size_t>() < count) visit(child.get<std::size_t>());
    };
    for (std::size_t n = 0; n < count; ++n) if (parent[n] < 0) visit(n);
    skeleton rig;
    for (std::size_t s = 0; s < order.size(); ++s) {
        const std::size_t n = order[s];
        const matrix4 & w = world[n];
        const auto len = [](float a, float b, float c) { return std::sqrt(a*a + b*b + c*c); };
        if (std::abs(len(w[0], w[1], w[2]) - 1.0F) > 1e-3F || std::abs(len(w[4], w[5], w[6]) - 1.0F) > 1e-3F ||
            std::abs(len(w[8], w[9], w[10]) - 1.0F) > 1e-3F)
            return std::unexpected(detail::fail(error_code::invalid_format, "a scaled joint cannot be re-framed: " + nodes[n].value("name", std::string{})));
        std::int32_t up = parent[n];
        while (up >= 0 && !is_joint[static_cast<std::size_t>(up)]) up = parent[static_cast<std::size_t>(up)];
        rig.names.push_back(nodes[n].value("name", std::string{}));
        rig.parents.push_back(up < 0 ? -1 : slot[static_cast<std::size_t>(up)]);
        rig.rest_positions.push_back({w[12], w[13], w[14]});
    }
    // A VRM humanoid bone aims at the next humanoid bone of its chain, even past twist or skirt
    // bones in between.
    std::vector<std::optional<vec3>> aims(order.size());
    if (auto ext = root.find("extensions"); ext != root.end() && ext->contains("VRMC_vrm")) {
        const json bones = (*ext)["VRMC_vrm"].value("humanoid", json::object()).value("humanBones", json::object());
        const auto node_of = [&](const std::string & name) -> std::int32_t {
            if (!bones.contains(name)) return -1;
            const std::size_t n = bones[name].value("node", count);
            return n < count ? static_cast<std::int32_t>(n) : -1;
        };
        std::vector<std::pair<std::string, std::vector<std::string>>> chains = {
            {"hips", {"spine"}}, {"spine", {"chest", "upperChest", "neck"}}, {"chest", {"upperChest", "neck"}},
            {"upperChest", {"neck"}}, {"neck", {"head"}}};
        for (const std::string side : {"left", "right"}) {
            chains.push_back({side + "Shoulder", {side + "UpperArm"}});
            chains.push_back({side + "UpperArm", {side + "LowerArm"}});
            chains.push_back({side + "LowerArm", {side + "Hand"}});
            chains.push_back({side + "Hand", {side + "MiddleProximal"}});
            chains.push_back({side + "UpperLeg", {side + "LowerLeg"}});
            chains.push_back({side + "LowerLeg", {side + "Foot"}});
            chains.push_back({side + "Foot", {side + "Toes"}});
            chains.push_back({side + "ThumbMetacarpal", {side + "ThumbProximal"}});
            chains.push_back({side + "ThumbProximal", {side + "ThumbDistal"}});
            for (const std::string finger : {"Index", "Middle", "Ring", "Little"}) {
                chains.push_back({side + finger + "Proximal", {side + finger + "Intermediate"}});
                chains.push_back({side + finger + "Intermediate", {side + finger + "Distal"}});
            }
        }
        for (const auto & [from, to] : chains) {
            const std::int32_t a = node_of(from);
            if (a < 0 || slot[static_cast<std::size_t>(a)] < 0) continue;
            for (const auto & next : to) {
                const std::int32_t b = node_of(next);
                if (b < 0) continue;
                const matrix4 & w = world[static_cast<std::size_t>(b)];
                aims[static_cast<std::size_t>(slot[static_cast<std::size_t>(a)])] = vec3{w[12], w[13], w[14]};
                break;
            }
        }
    }
    const std::vector<quat> frames = joint_rest_frames(rig, aims);
    // New world rest of every node; delta[n] = old^-1 * new is a pure rotation.
    std::vector<matrix4> renewed = world;
    std::vector<quat> delta(count);
    for (std::size_t s = 0; s < order.size(); ++s) {
        const std::size_t n = order[s];
        if (keep_frame[n]) continue;
        renewed[n] = rigid_matrix(rig.rest_positions[s], frames[s]);
        delta[n] = q_norm(q_mul(q_conj(matrix_rotation(world[n])), frames[s]));
    }
    for (std::size_t n = 0; n < count; ++n) {
        if (!is_joint[n] && (parent[n] < 0 || !is_joint[static_cast<std::size_t>(parent[n])])) continue;
        const matrix4 local = parent[n] < 0 ? renewed[n] : multiply(general_inverse(renewed[static_cast<std::size_t>(parent[n])]), renewed[n]);
        write_trs(nodes[n], local);
    }
    // Inverse binds: new = new_world^-1 * old_world * old_inverse_bind, so skinning is unchanged.
    for (auto & skin_value : root["skins"]) {
        const auto & joints = skin_value.value("joints", json::array());
        std::vector<std::array<float, 16>> binds(joints.size());
        std::optional<accessor> old_binds;
        if (skin_value.contains("inverseBindMatrices") && skin_value["inverseBindMatrices"].is_number_unsigned()) {
            auto read = get_accessor(*doc, skin_value["inverseBindMatrices"].get<std::size_t>());
            if (!read) return std::unexpected(read.error());
            if (read->count != joints.size()) return std::unexpected(detail::fail(error_code::invalid_format, "inverse binds do not match the joints"));
            old_binds = *read;
        }
        for (std::size_t j = 0; j < joints.size(); ++j) {
            const std::size_t n = joints[j].get<std::size_t>();
            matrix4 old = identity_matrix();
            if (old_binds) {
                if (old_binds->component != 5126U || old_binds->shape != "MAT4")
                    return std::unexpected(detail::fail(error_code::invalid_format, "inverse binds must be float MAT4"));
                for (std::size_t k = 0; k < 16U; ++k)
                    old[k] = little<float>(old_binds->bytes.data() + j * old_binds->stride + k * 4U);
            }
            binds[j] = multiply(multiply(rigid_inverse(renewed[n]), world[n]), old);
        }
        skin_value["inverseBindMatrices"] = builder.accessor_for<std::array<float, 16>>(binds, 5126U, "MAT4");
    }
    // Clips: local'(t) = delta[parent]^-1 * local(t) * delta[node].
    if (root.contains("animations")) {
        for (auto & clip : root["animations"]) {
            for (auto & channel : clip["channels"]) {
                const std::size_t n = channel["target"].value("node", count);
                if (n >= count) continue;
                const quat own = delta[n];
                const quat above = parent[n] < 0 ? quat{} : delta[static_cast<std::size_t>(parent[n])];
                if (own.x == 0.0F && own.y == 0.0F && own.z == 0.0F && above.x == 0.0F && above.y == 0.0F && above.z == 0.0F) continue;
                const std::string path = channel["target"].value("path", std::string{});
                json & sampler = clip["samplers"][channel["sampler"].get<std::size_t>()];
                auto values = get_accessor(*doc, sampler["output"].get<std::size_t>());
                if (!values) return std::unexpected(values.error());
                if (path == "rotation") {
                    std::vector<quat> out(values->count);
                    for (std::size_t i = 0; i < values->count; ++i) {
                        auto q = quaternion(*values, i);
                        if (!q) return std::unexpected(q.error());
                        out[i] = q_norm(q_mul(q_mul(q_conj(above), *q), own));
                    }
                    sampler["output"] = builder.accessor_for<quat>(out, 5126U, "VEC4");
                } else if (path == "translation") {
                    std::vector<vec3> out(values->count);
                    for (std::size_t i = 0; i < values->count; ++i) {
                        auto v = vector3(*values, i);
                        if (!v) return std::unexpected(v.error());
                        out[i] = q_rotate(q_conj(above), *v);
                    }
                    sampler["output"] = builder.accessor_for<vec3>(out, 5126U, "VEC3");
                } else if (path == "scale") {
                    return std::unexpected(detail::fail(error_code::invalid_format, "a scaled clip cannot be re-framed"));
                }
            }
        }
    }
    // Offsets stated in a node's own frame move with it: spring-bone colliders, the look-at origin.
    const auto refit = [&](json & value, std::size_t n) {
        if (!value.is_array() || value.size() != 3U || n >= count) return;
        const vec3 v{value[0].get<float>(), value[1].get<float>(), value[2].get<float>()};
        const vec3 moved = q_rotate(q_conj(delta[n]), v);
        value = {moved.x, moved.y, moved.z};
    };
    if (auto ext = root.find("extensions"); ext != root.end()) {
        if (ext->contains("VRMC_springBone"))
            for (auto & collider : (*ext)["VRMC_springBone"]["colliders"]) {
                const std::size_t n = collider.value("node", count);
                for (auto & [kind, shape] : collider["shape"].items()) {
                    if (shape.contains("offset")) refit(shape["offset"], n);
                    if (shape.contains("tail")) refit(shape["tail"], n);
                }
            }
        if (ext->contains("VRMC_vrm")) {
            json & vrm = (*ext)["VRMC_vrm"];
            const json bones = vrm.value("humanoid", json::object()).value("humanBones", json::object());
            if (bones.contains("head") && vrm.contains("lookAt") && vrm["lookAt"].contains("offsetFromHeadBone"))
                refit(vrm["lookAt"]["offsetFromHeadBone"], bones["head"].value("node", count));
        }
    }
    align4(builder.binary);
    root["buffers"][0]["byteLength"] = builder.binary.size();
    return pack_glb(builder);
}

result<std::vector<std::byte>> pose_to_t_pose_glb_memory(std::span<const std::byte> bytes) {
    auto doc = parse_document(bytes);
    if (!doc) return std::unexpected(doc.error());
    glb_builder builder;
    builder.root = doc->root;
    builder.binary = doc->binary;
    json & root = builder.root;
    auto ext = root.find("extensions");
    if (ext == root.end() || !ext->contains("VRMC_vrm"))
        return std::unexpected(detail::fail(error_code::invalid_format, "a T-pose needs a VRM 1.0 humanoid"));
    if (root.contains("animations") && !root["animations"].empty())
        return std::unexpected(detail::fail(error_code::invalid_format, "a file with clips cannot change its rest pose"));
    if (!root.contains("buffers") || root["buffers"].empty() || root["buffers"][0].contains("uri"))
        return std::unexpected(detail::fail(error_code::invalid_format, "GLB has no binary chunk"));
    json & nodes = root["nodes"];
    const std::size_t count = nodes.size();
    std::vector<std::int32_t> parent(count, -1);
    for (std::size_t n = 0; n < count; ++n)
        for (const auto & child : nodes[n].value("children", json::array()))
            if (child.is_number_unsigned() && child.get<std::size_t>() < count) parent[child.get<std::size_t>()] = static_cast<std::int32_t>(n);
    std::vector<matrix4> world(count);
    std::vector<std::uint8_t> state(count, 0U);
    std::function<result<matrix4>(std::size_t)> world_of = [&](std::size_t n) -> result<matrix4> {
        if (state[n] == 2U) return world[n];
        if (state[n] == 1U) return std::unexpected(detail::fail(error_code::invalid_format, "node hierarchy contains a cycle"));
        state[n] = 1U;
        auto local = node_matrix(nodes[n]);
        if (!local) return std::unexpected(local.error());
        if (parent[n] < 0) world[n] = *local;
        else {
            auto up = world_of(static_cast<std::size_t>(parent[n]));
            if (!up) return std::unexpected(up.error());
            world[n] = multiply(*up, *local);
        }
        state[n] = 2U;
        return world[n];
    };
    for (std::size_t n = 0; n < count; ++n) if (auto w = world_of(n); !w) return std::unexpected(w.error());
    const json bones = (*ext)["VRMC_vrm"].value("humanoid", json::object()).value("humanBones", json::object());
    const auto node_of = [&](const std::string & name) -> std::int32_t {
        if (!bones.contains(name)) return -1;
        const std::size_t n = bones[name].value("node", count);
        return n < count ? static_cast<std::int32_t>(n) : -1;
    };
    // T-pose (VRM 1.0): facing +Z, the left arm and its fingers along +X, the right along -X,
    // the legs straight down. Spine, neck, head, feet and thumbs keep their pose.
    struct aim { std::string from; std::vector<std::string> to; vec3 direction; };
    std::vector<aim> aims;
    for (const std::string side : {"left", "right"}) {
        const vec3 out{side == "left" ? 1.0F : -1.0F, 0.0F, 0.0F};
        aims.push_back({side + "Shoulder", {side + "UpperArm"}, out});
        aims.push_back({side + "UpperArm", {side + "LowerArm"}, out});
        aims.push_back({side + "LowerArm", {side + "Hand"}, out});
        aims.push_back({side + "Hand", {side + "MiddleProximal"}, out});
        for (const std::string finger : {"Index", "Middle", "Ring", "Little"}) {
            aims.push_back({side + finger + "Proximal", {side + finger + "Intermediate"}, out});
            aims.push_back({side + finger + "Intermediate", {side + finger + "Distal"}, out});
        }
        aims.push_back({side + "UpperLeg", {side + "LowerLeg"}, {0.0F, -1.0F, 0.0F}});
        aims.push_back({side + "LowerLeg", {side + "Foot"}, {0.0F, -1.0F, 0.0F}});
    }
    std::vector<std::vector<std::size_t>> children(count);
    for (std::size_t n = 0; n < count; ++n) if (parent[n] >= 0) children[static_cast<std::size_t>(parent[n])].push_back(n);
    std::vector<matrix4> posed = world;
    const auto dot = [](vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; };
    for (const auto & item : aims) {
        const std::int32_t a = node_of(item.from);
        std::int32_t b = -1;
        for (const auto & next : item.to) if ((b = node_of(next)) >= 0) break;
        if (a < 0 || b < 0) continue;
        const matrix4 & wa = posed[static_cast<std::size_t>(a)];
        const matrix4 & wb = posed[static_cast<std::size_t>(b)];
        vec3 bone{wb[12] - wa[12], wb[13] - wa[13], wb[14] - wa[14]};
        const float length = std::sqrt(dot(bone, bone));
        if (length < 1e-6F) continue;
        bone = {bone.x/length, bone.y/length, bone.z/length};
        const float cosine = std::clamp(dot(bone, item.direction), -1.0F, 1.0F);
        quat turn;
        if (cosine < -0.999999F) turn = {0.0F, 0.0F, 1.0F, 0.0F};
        else { const vec3 axis = cross(bone, item.direction); turn = q_norm({axis.x, axis.y, axis.z, 1.0F + cosine}); }
        // Turn the bone and everything below it about the bone's own joint.
        const vec3 pivot{wa[12], wa[13], wa[14]};
        const matrix4 about = multiply(rigid_matrix(pivot, turn), rigid_matrix({-pivot.x, -pivot.y, -pivot.z}, quat{}));
        std::vector<std::size_t> stack{static_cast<std::size_t>(a)};
        while (!stack.empty()) {
            const std::size_t n = stack.back(); stack.pop_back();
            posed[n] = multiply(about, posed[n]);
            for (auto c : children[n]) stack.push_back(c);
        }
    }
    for (std::size_t n = 0; n < count; ++n) {
        if (posed[n] == world[n]) continue;
        const matrix4 local = parent[n] < 0 ? posed[n] : multiply(general_inverse(posed[static_cast<std::size_t>(parent[n])]), posed[n]);
        write_trs(nodes[n], local);
    }
    // Bake the pose into each skinned mesh (linear blend skinning, as a renderer would), then bind
    // it at the new rest: new inverse bind = new joint world^-1, vertices in world space.
    std::vector<std::uint8_t> rebound(root["skins"].size(), 0U);
    for (std::size_t n = 0; n < count; ++n) {
        if (!nodes[n].contains("mesh") || !nodes[n].contains("skin")) continue;
        const std::size_t skin_index = nodes[n]["skin"].get<std::size_t>();
        json & skin_value = root["skins"][skin_index];
        const auto & joints = skin_value["joints"];
        std::vector<matrix4> skinning(joints.size());
        std::optional<accessor> binds;
        if (skin_value.contains("inverseBindMatrices")) {
            auto read = get_accessor(*doc, skin_value["inverseBindMatrices"].get<std::size_t>());
            if (!read) return std::unexpected(read.error());
            binds = *read;
        }
        for (std::size_t j = 0; j < joints.size(); ++j) {
            matrix4 old = identity_matrix();
            if (binds) for (std::size_t k = 0; k < 16U; ++k) old[k] = little<float>(binds->bytes.data() + j * binds->stride + k * 4U);
            skinning[j] = multiply(posed[joints[j].get<std::size_t>()], old);
        }
        json & mesh_value = root["meshes"][nodes[n]["mesh"].get<std::size_t>()];
        for (auto & primitive : mesh_value["primitives"]) {
            json & attributes = primitive["attributes"];
            if (!attributes.contains("JOINTS_0") || !attributes.contains("WEIGHTS_0") || !attributes.contains("POSITION")) continue;
            auto joint_values = get_accessor(*doc, attributes["JOINTS_0"].get<std::size_t>());
            auto weight_values = get_accessor(*doc, attributes["WEIGHTS_0"].get<std::size_t>());
            auto positions = read_vec3(*doc, attributes["POSITION"]);
            if (!joint_values || !weight_values || !positions)
                return std::unexpected(detail::fail(error_code::invalid_format, "skinned primitive cannot be read"));
            const std::size_t vertices = positions->size();
            std::vector<matrix4> blend(vertices);
            for (std::size_t v = 0; v < vertices; ++v) {
                auto ji = joint_vector(*joint_values, v);
                auto wi = weight_vector(*weight_values, v);
                if (!ji || !wi) return std::unexpected(detail::fail(error_code::invalid_format, "skin weights cannot be read"));
                matrix4 m{};
                for (std::size_t k = 0; k < 4U; ++k) {
                    if ((*wi)[k] == 0.0F) continue;
                    if ((*ji)[k] >= skinning.size()) return std::unexpected(detail::fail(error_code::invalid_format, "joint index out of range"));
                    for (std::size_t e = 0; e < 16U; ++e) m[e] += (*wi)[k] * skinning[(*ji)[k]][e];
                }
                blend[v] = m;
            }
            const auto normal_of = [&](const matrix4 & m, vec3 v) {
                const matrix4 inverse = general_inverse(m);  // inverse transpose for normals
                return unit({inverse[0]*v.x + inverse[1]*v.y + inverse[2]*v.z, inverse[4]*v.x + inverse[5]*v.y + inverse[6]*v.z,
                             inverse[8]*v.x + inverse[9]*v.y + inverse[10]*v.z}, v);
            };
            std::vector<vec3> moved(vertices);
            for (std::size_t v = 0; v < vertices; ++v) moved[v] = transform_point(blend[v], (*positions)[v]);
            attributes["POSITION"] = builder.accessor_for<vec3>(moved, 5126U, "VEC3", 34962U, vec3_bounds(moved));
            if (attributes.contains("NORMAL")) {
                auto normals = read_vec3(*doc, attributes["NORMAL"]);
                if (!normals) return std::unexpected(normals.error());
                for (std::size_t v = 0; v < vertices; ++v) (*normals)[v] = normal_of(blend[v], (*normals)[v]);
                attributes["NORMAL"] = builder.accessor_for<vec3>(*normals, 5126U, "VEC3", 34962U);
            }
            if (attributes.contains("TANGENT")) {
                auto tangents = get_accessor(*doc, attributes["TANGENT"].get<std::size_t>());
                if (!tangents) return std::unexpected(tangents.error());
                std::vector<quat> out(vertices);
                for (std::size_t v = 0; v < vertices; ++v) {
                    auto t = quaternion(*tangents, v);
                    if (!t) return std::unexpected(t.error());
                    const vec3 turned = unit(linear(blend[v], {t->x, t->y, t->z}), {t->x, t->y, t->z});
                    out[v] = {turned.x, turned.y, turned.z, t->w};
                }
                attributes["TANGENT"] = builder.accessor_for<quat>(out, 5126U, "VEC4", 34962U);
            }
            if (primitive.contains("targets"))
                for (auto & target : primitive["targets"])
                    for (const char * name : {"POSITION", "NORMAL", "TANGENT"}) {
                        if (!target.contains(name)) continue;
                        auto deltas = read_vec3(*doc, target[name]);
                        if (!deltas) return std::unexpected(deltas.error());
                        for (std::size_t v = 0; v < vertices && v < deltas->size(); ++v)
                            (*deltas)[v] = std::string_view{name} == "NORMAL" ?
                                [&] { const matrix4 inv = general_inverse(blend[v]); const vec3 d = (*deltas)[v];
                                      return vec3{inv[0]*d.x + inv[1]*d.y + inv[2]*d.z, inv[4]*d.x + inv[5]*d.y + inv[6]*d.z,
                                                  inv[8]*d.x + inv[9]*d.y + inv[10]*d.z}; }() :
                                linear(blend[v], (*deltas)[v]);
                        target[name] = builder.accessor_for<vec3>(*deltas, 5126U, "VEC3", 34962U,
                            std::string_view{name} == "POSITION" ? vec3_bounds(*deltas) : json{});
                    }
        }
        rebound[skin_index] = 1U;
    }
    for (std::size_t s_index = 0; s_index < root["skins"].size(); ++s_index) {
        json & skin_value = root["skins"][s_index];
        const auto & joints = skin_value["joints"];
        std::vector<std::array<float, 16>> binds(joints.size());
        for (std::size_t j = 0; j < joints.size(); ++j) binds[j] = general_inverse(posed[joints[j].get<std::size_t>()]);
        if (rebound[s_index] || !skin_value.contains("inverseBindMatrices"))
            skin_value["inverseBindMatrices"] = builder.accessor_for<std::array<float, 16>>(binds, 5126U, "MAT4");
    }
    drop_unused_bytes(builder);
    align4(builder.binary);
    root["buffers"][0]["byteLength"] = builder.binary.size();
    return pack_glb(builder);
}

result<void> save_skinned_animation_glb_file(
    const std::filesystem::path & path, const mesh & geometry, const skin & binding,
    const motion & animation) {
    auto bytes = save_skinned_animation_glb_memory(geometry, binding, animation);
    if (!bytes) return std::unexpected(bytes.error());
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) return std::unexpected(detail::fail(error_code::io, "cannot create GLB: " + path.string()));
    stream.write(reinterpret_cast<const char *>(bytes->data()), static_cast<std::streamsize>(bytes->size()));
    if (!stream) return std::unexpected(detail::fail(error_code::io, "failed while writing GLB"));
    return {};
}

} // namespace skintokens
