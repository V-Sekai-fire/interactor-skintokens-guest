#include "internal.hpp"

#include <fstream>
#include <streambuf>

namespace skintokens::detail {
namespace {

class source_buffer final : public std::streambuf {
public:
    source_buffer(std::filesystem::path path, const file_source & files)
        : path_(std::move(path)), files_(files), window_(1U << 20U) {}

protected:
    int_type underflow() override {
        if (gptr() < egptr()) return traits_type::to_int_type(*gptr());
        start_ += static_cast<std::uint64_t>(egptr() - eback());
        const std::size_t have = std::min(window_.size(),
            files_.read(path_, start_, std::as_writable_bytes(std::span{window_})));
        if (have == 0) return traits_type::eof();
        setg(window_.data(), window_.data(), window_.data() + have);
        return traits_type::to_int_type(window_[0]);
    }

    pos_type seekoff(off_type offset, std::ios_base::seekdir way, std::ios_base::openmode) override {
        const auto here = static_cast<off_type>(start_) + (gptr() - eback());
        if (way == std::ios_base::cur) return seekpos(here + offset, std::ios_base::in);
        if (way == std::ios_base::beg) return seekpos(offset, std::ios_base::in);
        return pos_type(off_type(-1));
    }

    pos_type seekpos(pos_type position, std::ios_base::openmode) override {
        const auto target = static_cast<off_type>(position);
        if (target < 0) return pos_type(off_type(-1));
        const auto begin = static_cast<off_type>(start_);
        if (target >= begin && target < begin + (egptr() - eback())) {
            setg(eback(), eback() + (target - begin), egptr());
        } else {
            start_ = static_cast<std::uint64_t>(target);
            setg(window_.data(), window_.data(), window_.data());
        }
        return position;
    }

private:
    std::filesystem::path path_;
    const file_source & files_;
    std::vector<char> window_;
    std::uint64_t start_ = 0;
};

class source_stream final : public std::istream {
public:
    source_stream(std::filesystem::path path, const file_source & files)
        : std::istream(nullptr), buffer_(std::move(path), files) { rdbuf(&buffer_); }

private:
    source_buffer buffer_;
};

} // namespace

std::unique_ptr<std::istream> open_source(const std::filesystem::path & path, const file_source & files) {
    if (files.read) return std::make_unique<source_stream>(path, files);
    return std::make_unique<std::ifstream>(path, std::ios::binary);
}

} // namespace skintokens::detail
