# skintokens.elf on Godot's WebGPU RenderingDevice, in a browser.
# URL: index.html?models=<dir URL>&mesh=<GLB URL>&opts=<K=V+...>&mem=<MiB>
# The guest names files by the paths it was given; this host turns each path into an HTTP
# range request on the page's own origin. Progress and the result go to window.__rig.
extends Node

const WAIT_GPU := 1
const READ := 2
const UPLOAD := 3
const COOP := 4
const DONE := 5
const ERROR := 6
const BLOCK := 1 << 20
const UPLOAD_CHUNK := 32 << 20
# Weights are stored back to back, so one large range request serves many tensor uploads.
const WINDOW := 64 << 20

var _sb
var _rd: RenderingDevice
var _http: HTTPRequest
var _blocks := {}
var _origin := ""
var _win_path := ""
var _win_first := 0
var _win := PackedByteArray()
var _stats := {"reads": 0, "fetches": 0, "fetch_bytes": 0, "uploads": 0, "upload_bytes": 0, "waits": 0, "coops": 0}


func _ready() -> void:
	_http = HTTPRequest.new()
	_http.use_threads = false
	# The default 64 KiB per frame caps a download at ~4 MB/s.
	_http.download_chunk_size = 16 << 20
	add_child(_http)
	run()


func say(text: String) -> void:
	print(text)
	if OS.has_feature("web"):
		JavaScriptBridge.eval("(window.__rig_log = window.__rig_log || []).push(%s)" % JSON.stringify(text))


func publish(state: Dictionary) -> void:
	if OS.has_feature("web"):
		JavaScriptBridge.eval("window.__rig = %s" % JSON.stringify(state))


func query() -> Dictionary:
	var out := {"models": "w", "mesh": "m/sophia.glb", "opts": "beams=1+temperature=0", "mem": "1024"}
	if not OS.has_feature("web"):
		return out
	var search: String = str(JavaScriptBridge.eval("location.search")).trim_prefix("?")
	for pair in search.split("&", false):
		var kv := pair.split("=", true, 1)
		out[kv[0].uri_decode()] = kv[1].uri_decode() if kv.size() > 1 else ""
	return out


func fetch(path: String, first: int, count: int) -> PackedByteArray:
	# count < 0: to the end of the file. A range past the end comes back empty.
	var span := "bytes=%d-" % first if count < 0 else "bytes=%d-%d" % [first, first + count - 1]
	var url := path if path.begins_with("http") else _origin + "/" + path.trim_prefix("/")
	var err := _http.request(url, PackedStringArray(["Range: " + span]))
	if err != OK:
		return PackedByteArray()
	var r: Array = await _http.request_completed
	_stats.fetches += 1
	var code: int = r[1]
	if code == 416:
		return PackedByteArray()
	if code != 206 and code != 200:
		say("FETCH %s %s -> HTTP %d" % [path, span, code])
		return PackedByteArray()
	var body: PackedByteArray = r[3]
	if code == 200 and (first > 0 or (count >= 0 and body.size() > count)):
		body = body.slice(first, first + count if count >= 0 else body.size())
	_stats.fetch_bytes += body.size()
	return body


func window_bytes(path: String, first: int, count: int) -> PackedByteArray:
	if path != _win_path or first < _win_first or first + count > _win_first + _win.size():
		_win = PackedByteArray()
		_win = await fetch(path, first, max(count, WINDOW))
		_win_path = path
		_win_first = first
	var lo := first - _win_first
	return _win.slice(lo, min(lo + count, _win.size()))


func read_cached(path: String, first: int, count: int) -> PackedByteArray:
	var out := PackedByteArray()
	var at := first
	var end := first + count
	while at < end:
		var index := at / BLOCK
		var key := "%s#%d" % [path, index]
		if not _blocks.has(key):
			_blocks[key] = await fetch(path, index * BLOCK, BLOCK)
		var block: PackedByteArray = _blocks[key]
		var lo := at - index * BLOCK
		if lo >= block.size():
			break
		var hi: int = min(block.size(), end - index * BLOCK)
		out.append_array(block.slice(lo, hi))
		if block.size() < BLOCK:
			break
		at = index * BLOCK + hi
	return out


func run():
	var q := query()
	if OS.has_feature("web"):
		_origin = str(JavaScriptBridge.eval("location.origin"))
	say("STEP query %s" % JSON.stringify(q))
	if not ClassDB.class_exists("Sandbox"):
		return fail("the Sandbox class is not registered")
	_sb = ClassDB.instantiate("Sandbox")
	_sb.memory_max = int(q.mem)
	_sb.references_max = 4096
	_sb.allocations_max = 10000000
	_sb.execution_timeout = 0
	add_child(_sb)
	_sb.load_buffer(FileAccess.get_file_as_bytes("res://skintokens.elf"))
	if not _sb.has_function("skintokens_start"):
		return fail("skintokens.elf did not load")
	_rd = RenderingServer.create_local_rendering_device()
	if _rd == null:
		return fail("no local RenderingDevice")
	say("STEP attach %s" % _sb.vmcall("skintokens_attach", _rd, 4096))
	var opts: String = q.opts.replace("+", " ") + " GGML_RD_MAX_BUFFER_MB=128"
	say("STEP start %s" % _sb.vmcall("skintokens_start", q.models, q.mesh, opts))
	var t0 := Time.get_ticks_msec()
	var frames := 0
	var feed := PackedByteArray()
	var last_report := 0
	while true:
		var r = _sb.vmcall("skintokens_pump", feed)
		feed = PackedByteArray()
		if typeof(r) != TYPE_ARRAY or r.size() < 3:
			return fail("skintokens_pump returned %s" % str(r))
		var hdr: PackedInt64Array = r[0]
		match hdr[0]:
			DONE:
				break
			ERROR:
				return fail("guest: %s" % str(r[1]).replace("\n", " | "))
			WAIT_GPU, COOP:
				if hdr[0] == WAIT_GPU:
					_stats.waits += 1
				else:
					_stats.coops += 1
				frames += 1
				await get_tree().process_frame
			READ:
				_stats.reads += 1
				feed = await read_cached(str(r[1]), hdr[1], hdr[2] if hdr[2] > 0 else 1 << 40)
			UPLOAD:
				var done := 0
				while done < hdr[2]:
					var n: int = min(hdr[2] - done, UPLOAD_CHUNK)
					var b := await window_bytes(str(r[1]), hdr[1] + done, n)
					if b.size() != n:
						return fail("UPLOAD %s: got %d of %d bytes" % [r[1], b.size(), n])
					var err := _rd.buffer_update(r[2], hdr[3] + done, n, b)
					if err != OK:
						return fail("UPLOAD buffer_update: %s" % error_string(err))
					done += n
				_stats.uploads += 1
				_stats.upload_bytes += done
			_:
				return fail("unknown request kind %d" % hdr[0])
		var now := Time.get_ticks_msec()
		if now - last_report > 10000:
			last_report = now
			publish({"state": "running", "ms": now - t0, "frames": frames, "host": _stats})
			say("PROGRESS ms=%d frames=%d host=%s" % [now - t0, frames, JSON.stringify(_stats)])
	var ms := Time.get_ticks_msec() - t0
	say("guest_log %s" % str(_sb.vmcall("skintokens_output")).strip_edges().replace("\n", " | "))
	say("stats %s" % _sb.vmcall("skintokens_stats"))
	var size := int(_sb.vmcall("skintokens_result_size"))
	var out := PackedByteArray()
	while out.size() < size:
		var chunk: PackedByteArray = _sb.vmcall("skintokens_result_chunk", out.size(), size - out.size())
		if chunk.is_empty():
			return fail("result_chunk returned nothing at %d of %d" % [out.size(), size])
		out.append_array(chunk)
	var hasher := HashingContext.new()
	hasher.start(HashingContext.HASH_SHA256)
	hasher.update(out)
	var sha := hasher.finish().hex_encode()
	say("RIG-OK bytes=%d ms=%d frames=%d sha256=%s host=%s" % [out.size(), ms, frames, sha, JSON.stringify(_stats)])
	publish({"state": "done", "ms": ms, "frames": frames, "bytes": out.size(), "sha256": sha,
		"host": _stats, "glb": Marshalls.raw_to_base64(out)})


func fail(why: String):
	say("FAIL %s" % why)
	publish({"state": "failed", "why": why, "host": _stats})
