# skintokens.elf in a Sandbox on a local RenderingDevice: rigs one GLB and writes the rigged GLB.
#   godot --path tests/rig --script run_rig.gd -- --elf=E --models=M --mesh=G --out=O [--opts="beams=1"] [--cpu]
# --cpu attaches no RenderingDevice, so ggml-rd runs its CPU kernels (the control).
extends SceneTree

const WAIT_GPU := 1
const READ := 2
const UPLOAD := 3
const COOP := 4
const DONE := 5
const ERROR := 6
const CHUNK := 64 * 1024 * 1024

var _a := {}
var _sb
var _rd: RenderingDevice
var _feed := PackedByteArray()
var _t0 := 0
var _frames := 0
var _stats := {"reads": 0, "read_bytes": 0, "uploads": 0, "upload_bytes": 0, "waits": 0, "coops": 0}


func _initialize() -> void:
	for arg in OS.get_cmdline_user_args():
		var kv := arg.trim_prefix("--").split("=", true, 1)
		_a[kv[0]] = kv[1] if kv.size() > 1 else ""
	if not ClassDB.class_exists("Sandbox"):
		_fail("the Sandbox class is not registered")
		return
	_sb = ClassDB.instantiate("Sandbox")
	_sb.memory_max = int(_a.get("mem", "4096"))
	_sb.references_max = int(_a.get("refs", "4096"))
	_sb.allocations_max = 10000000
	_sb.execution_timeout = 0
	var elf := FileAccess.get_file_as_bytes(_a.elf)
	_sb.load_buffer(elf)
	if not _sb.has_function("skintokens_start"):
		_fail("%s did not load" % _a.elf)
		return
	if not _a.has("cpu"):
		_rd = RenderingServer.create_local_rendering_device()
		if _rd == null:
			_fail("no local RenderingDevice (run windowed, not --headless)")
			return
	print("STEP attach %s" % _sb.vmcall("skintokens_attach", _rd, 4096))
	print("STEP start %s" % _sb.vmcall("skintokens_start", _a.models, _a.mesh, _a.get("opts", "beams=1")))
	_t0 = Time.get_ticks_msec()


func _process(_d: float) -> bool:
	if _sb == null:
		return true
	_frames += 1
	var budget := 512 * 1024 * 1024
	while true:
		var r = _sb.vmcall("skintokens_pump", _feed)
		_feed = PackedByteArray()
		if typeof(r) != TYPE_ARRAY or r.size() < 3:
			return _fail("skintokens_pump returned %s" % str(r))
		var hdr: PackedInt64Array = r[0]
		match hdr[0]:
			DONE:
				return _finish()
			ERROR:
				return _fail("guest: %s" % str(r[1]).replace("\n", " | "))
			WAIT_GPU:
				_stats.waits += 1
				return false
			COOP:
				_stats.coops += 1
				return false
			READ:
				var f := FileAccess.open(str(r[1]), FileAccess.READ)
				if f == null:
					return _fail("READ %s: %s" % [r[1], error_string(FileAccess.get_open_error())])
				f.seek(hdr[1])
				_feed = f.get_buffer(hdr[2] if hdr[2] > 0 else f.get_length() - hdr[1])
				_stats.reads += 1
				_stats.read_bytes += _feed.size()
				budget -= _feed.size()
			UPLOAD:
				if _rd == null:
					return _fail("UPLOAD without a RenderingDevice")
				var f := FileAccess.open(str(r[1]), FileAccess.READ)
				var done := 0
				while done < hdr[2]:
					var n: int = min(hdr[2] - done, CHUNK)
					f.seek(hdr[1] + done)
					var b := f.get_buffer(n)
					if b.size() != n:
						return _fail("UPLOAD %s: short read" % r[1])
					var err := _rd.buffer_update(r[2], hdr[3] + done, n, b)
					if err != OK:
						return _fail("UPLOAD buffer_update: %s" % error_string(err))
					done += n
				_stats.uploads += 1
				_stats.upload_bytes += done
				budget -= done
			_:
				return _fail("unknown request kind %d" % hdr[0])
		if budget <= 0:
			return false
	return false


func _finish() -> bool:
	var ms := Time.get_ticks_msec() - _t0
	print("guest_log %s" % str(_sb.vmcall("skintokens_output")).strip_edges().replace("\n", " | "))
	print("stats %s" % _sb.vmcall("skintokens_stats"))
	var size := int(_sb.vmcall("skintokens_result_size"))
	var out := PackedByteArray()
	while out.size() < size:
		var chunk: PackedByteArray = _sb.vmcall("skintokens_result_chunk", out.size(), size - out.size())
		if chunk.is_empty():
			return _fail("result_chunk returned nothing at %d of %d" % [out.size(), size])
		out.append_array(chunk)
	var w := FileAccess.open(_a.out, FileAccess.WRITE)
	w.store_buffer(out)
	w.close()
	print("RIG-OK bytes=%d ms=%d frames=%d host=%s" % [out.size(), ms, _frames, str(_stats)])
	quit(0)
	return true


func _fail(why: String) -> bool:
	print("FAIL %s" % why)
	quit(1)
	return true
