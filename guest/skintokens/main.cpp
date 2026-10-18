// skintokens.elf: SkinTokens / TokenRig auto-rigging on ggml-rd.
//
// The model code is skin-tokens-ggml's own (vendor/skin-tokens-ggml). Only where bytes come from
// changes: the guest cannot open files, so GGUF metadata and the mesh come through the pump (READ)
// and every weight goes from the host file straight into the RenderingDevice buffer (UPLOAD).
//
// Host side:
//   skintokens_attach(rd, total_mb)                once; rd may be null (ggml-rd CPU kernels)
//   skintokens_start(model_dir, mesh_glb, opts)    opts: "beams=1 seed=0 temperature=1 max_tokens=2048 robust=1 K=V ..."
//                                                   (K=V other than those = GGML_RD_* switches)
//   skintokens_pump(data)                          every frame until DONE / ERROR
//   skintokens_result_size(), skintokens_result_chunk(off, n)   the rigged GLB, <= 8 MiB a call
//   skintokens_trace_size(), skintokens_trace_chunk(off, n)     with opts trace=1: the weight
//                                                   transfer's inputs (skintokens-transfer-replay)
#include <api.hpp>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <skintokens/skintokens.hpp>

#include "binding.hpp"

#include "ggml-backend.h"
#include "ggml-rd.h"
#include "ggml.h"
#include "pump/pump.h"
#include "rd_compute.h"

static rdc::Device g_dev;
static bool g_attached = false;
static std::string g_log;

static void logf_(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void logf_(const char *fmt, ...) {
	char buf[1024];
	va_list ap;
	va_start(ap, fmt);
	std::vsnprintf(buf, sizeof buf, fmt, ap);
	va_end(ap);
	g_log += buf;
	g_log += '\n';
	std::printf("skintokens: %s\n", buf);
}

static void hook_wait_gpu(void *) {
	pump::wait_gpu();
}

static void hook_coop(void *) {
	pump::coop();
}

static bool hook_upload(void *, const std::string &path, uint64_t file_offset, uint64_t bytes, ::RID rid,
		uint64_t dst_offset) {
	return pump::upload(path, file_offset, bytes, rid, dst_offset);
}

static size_t read_into(const std::string &path, uint64_t offset, uint8_t *dst, uint64_t bytes) {
	const uint64_t chunk = uint64_t(16) << 20;
	uint64_t done = 0;
	while (done < bytes) {
		const uint64_t n = std::min(bytes - done, chunk);
		std::vector<uint8_t> v = pump::read(path, offset + done, n);
		std::memcpy(dst + done, v.data(), v.size());
		done += v.size();
		if (v.size() < n) {
			break;
		}
	}
	return size_t(done);
}

static bool hook_read(void *, const std::string &path, uint64_t file_offset, uint64_t bytes, void *dst) {
	return read_into(path, file_offset, static_cast<uint8_t *>(dst), bytes) == bytes;
}

static void on_ggml_abort(const char *message) {
	std::fflush(stdout);
	pump::fail(std::string("ggml abort: ") + message);
}

static std::vector<std::byte> read_whole(const std::string &path) {
	std::vector<std::byte> out;
	const uint64_t chunk = uint64_t(16) << 20;
	for (;;) {
		const size_t at = out.size();
		out.resize(at + chunk);
		const size_t got = read_into(path, at, reinterpret_cast<uint8_t *>(out.data() + at), chunk);
		out.resize(at + got);
		if (got < chunk) {
			return out;
		}
	}
}

struct Job {
	std::string dir;
	std::string mesh;
	skintokens::generation_options gen;
};
static Job g_job;
static std::optional<skintokens::model> g_model;
static std::vector<uint8_t> g_result;
static std::vector<uint8_t> g_trace;

static bool load_model(const std::string &dir) {
	ggml_backend_reg_t reg = ggml_backend_rd_reg();
	if (ggml_backend_reg_dev_count(reg) == 0) {
		logf_("no ggml-rd device (call skintokens_attach first)");
		return false;
	}
	skintokens::runtime_options rt;
	rt.device_handle = ggml_backend_reg_dev_get(reg, 0);
	rt.files.read = [](const std::filesystem::path &path, uint64_t offset, std::span<std::byte> out) {
		return read_into(path.string(), offset, reinterpret_cast<uint8_t *>(out.data()), out.size());
	};
	rt.files.load_tensor = [](ggml_tensor *t, const std::filesystem::path &path, uint64_t offset) {
		if (!ggml_backend_rd_tensor_upload(t, 0, path.string(), offset, ggml_nbytes(t))) {
			logf_("upload of %s failed: %s", ggml_get_name(t), ggml_backend_rd_last_error().c_str());
			return false;
		}
		return true;
	};
	auto m = skintokens::model::load(dir, rt);
	if (!m) {
		logf_("model load failed: %s", m.error().message.c_str());
		return false;
	}
	g_model.emplace(std::move(*m));
	logf_("model loaded from %s", dir.c_str());
	return true;
}

static void rig_job(void *) {
	if (!g_model && !load_model(g_job.dir)) {
		pump::fail("skintokens: model load failed:\n" + g_log);
	}
	const std::vector<std::byte> glb = read_whole(g_job.mesh);
	auto mesh = skintokens::load_glb_memory(glb);
	if (!mesh) {
		pump::fail("skintokens: " + g_job.mesh + ": " + mesh.error().message);
	}
	logf_("mesh: %zu vertices, %zu faces", mesh->vertices.size(), mesh->faces.size());
	auto binding = g_model->rig(*mesh, g_job.gen);
	if (!binding) {
		pump::fail("skintokens: rig failed: " + binding.error().message + "\n" + g_log);
	}
	skintokens::motion rest;
	rest.frames = 1U;
	rest.frames_per_second = 30.0F;
	rest.rig = binding->rig;
	rest.root_translations.assign(1U, rest.rig.rest_positions.front());
	rest.local_rotations.assign(rest.rig.names.size(), {});
	auto bytes = skintokens::save_skinned_animation_glb_memory(*mesh, *binding, rest);
	if (!bytes) {
		pump::fail("skintokens: GLB export failed: " + bytes.error().message);
	}
	g_result.assign(reinterpret_cast<const uint8_t *>(bytes->data()),
			reinterpret_cast<const uint8_t *>(bytes->data()) + bytes->size());
	const auto &t = binding->transfer;
	logf_("transfer: robust=%d welded=%zu matched=%zu outliers=%zu inpainted=%zu fallback=%zu trimmed_parts=%zu joints_solved=%zu iterations=%zu residual=%.1e",
			int(t.robust), t.welded, t.matched, t.outliers, t.inpainted, t.fallback, t.trimmed_parts, t.solved_joints, t.iterations, double(t.residual));
	logf_("rigged: %zu joints, %zu bytes", binding->rig.names.size(), g_result.size());
}

static std::vector<std::string> split_ws(const std::string &s) {
	std::vector<std::string> r;
	size_t i = 0;
	while (i < s.size()) {
		while (i < s.size() && s[i] == ' ') {
			++i;
		}
		size_t j = i;
		while (j < s.size() && s[j] != ' ') {
			++j;
		}
		if (j > i) {
			r.push_back(s.substr(i, j - i));
		}
		i = j;
	}
	return r;
}

static Variant skintokens_attach(Object rd, int64_t total_mb) {
	if (rd.is_valid()) {
		g_dev.adopt(rd);
	}
	g_attached = g_dev.ok();
	static bool registered = false;
	if (!registered) {
		registered = true;
		ggml_backend_register(ggml_backend_rd_reg());
	}
	ggml_backend_rd_attach(g_attached ? &g_dev : nullptr, size_t(total_mb) << 20);
	ggml_rd_hooks h;
	h.wait_gpu = hook_wait_gpu;
	h.coop = hook_coop;
	h.upload = hook_upload;
	h.read = hook_read;
	ggml_backend_rd_set_hooks(h);
	ggml_set_abort_callback(on_ggml_abort);
	return Variant(String(g_attached ? "attached device=" + g_dev.device_name() : std::string("no RD device")));
}

static Variant skintokens_start(String model_dir, String mesh_glb, String opts) {
	for (const char *k : { "GGML_RD_FAULT", "GGML_RD_BARRIER_ALL", "GGML_RD_TIMESTAMPS" }) {
		unsetenv(k);
	}
	g_job = Job{};
	g_trace.clear();
	skintokens::detail::transfer_capture() = nullptr;
	g_job.dir = model_dir.utf8();
	g_job.mesh = mesh_glb.utf8();
	for (const std::string &kv : split_ws(opts.utf8())) {
		const size_t eq = kv.find('=');
		if (eq == std::string::npos) {
			continue;
		}
		const std::string k = kv.substr(0, eq);
		const std::string v = kv.substr(eq + 1);
		if (k == "beams") {
			g_job.gen.beams = uint32_t(std::stoul(v));
		} else if (k == "seed") {
			g_job.gen.seed = std::stoull(v);
		} else if (k == "temperature") {
			g_job.gen.temperature = std::stof(v);
		} else if (k == "max_tokens") {
			g_job.gen.max_tokens = std::stoul(v);
		} else if (k == "trace") {
			if (v != "0") {
				skintokens::detail::transfer_capture() = [](const skintokens::detail::transfer_inputs &in) {
					const auto bytes = skintokens::detail::serialize_transfer_inputs(in);
					g_trace.assign(reinterpret_cast<const uint8_t *>(bytes.data()),
							reinterpret_cast<const uint8_t *>(bytes.data()) + bytes.size());
				};
			}
		} else if (k == "robust") {
			// robust=0 is the control: upstream eight-neighbour interpolation.
			g_job.gen.robust_transfer = v != "0";
		} else {
			setenv(k.c_str(), v.c_str(), 1);
		}
	}
	g_log.clear();
	g_result.clear();
	if (!pump::start(&rig_job, nullptr, size_t(16) << 20)) {
		return Variant(String("FAIL a job is running"));
	}
	return Variant(String("STARTED beams=" + std::to_string(g_job.gen.beams)));
}

static Variant skintokens_pump(PackedArray<uint8_t> in) {
	return pump::step(in);
}

static Variant skintokens_result_size() {
	return Variant(int64_t(g_result.size()));
}

static Variant chunk_of(const std::vector<uint8_t> &data, int64_t offset, int64_t bytes) {
	const size_t off = size_t(std::max<int64_t>(offset, 0));
	if (off >= data.size()) {
		return Variant(PackedArray<uint8_t>(std::vector<uint8_t>()));
	}
	const size_t n = std::min<size_t>({ size_t(std::max<int64_t>(bytes, 0)), size_t(8) << 20, data.size() - off });
	return Variant(PackedArray<uint8_t>(data.data() + off, n));
}

static Variant skintokens_result_chunk(int64_t offset, int64_t bytes) {
	return chunk_of(g_result, offset, bytes);
}

static Variant skintokens_trace_size() {
	return Variant(int64_t(g_trace.size()));
}

static Variant skintokens_trace_chunk(int64_t offset, int64_t bytes) {
	return chunk_of(g_trace, offset, bytes);
}

static Variant skintokens_output() {
	return Variant(String(g_log));
}

static Variant skintokens_stats() {
	std::string r = ggml_backend_rd_stats();
	r += " syncs=" + std::to_string(g_dev.syncs());
	r += " submits=" + std::to_string(g_dev.submits());
	r += " permanent_slots=" + std::to_string(g_dev.permanent_slots());
	return Variant(String(r));
}

int main() {
	ADD_API_FUNCTION(skintokens_attach, "String", "Object rd, int total_mb", "Attach the host's RenderingDevice");
	ADD_API_FUNCTION(skintokens_start, "String", "String model_dir, String mesh_glb, String opts", "Rig a GLB on the pump");
	ADD_API_FUNCTION(skintokens_pump, "Array", "PackedByteArray data", "Resume the job once: [header, text, rid]");
	ADD_API_FUNCTION(skintokens_result_size, "int", "", "Bytes of the rigged GLB");
	ADD_API_FUNCTION(skintokens_result_chunk, "PackedByteArray", "int offset, int bytes", "Rigged GLB bytes (<= 8 MiB)");
	ADD_API_FUNCTION(skintokens_trace_size, "int", "", "Bytes of the transfer trace (opts trace=1)");
	ADD_API_FUNCTION(skintokens_trace_chunk, "PackedByteArray", "int offset, int bytes", "Transfer trace bytes (<= 8 MiB)");
	ADD_API_FUNCTION(skintokens_output, "String", "", "What the job logged");
	ADD_API_FUNCTION(skintokens_stats, "String", "", "ggml-rd and rd_compute counters");
	halt();
}
