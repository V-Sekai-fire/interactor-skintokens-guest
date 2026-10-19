# skintokens.elf in a Sandbox: probe loads it, fixture writes a small rigged humanoid GLB, run makes the
# weight-free motion step on it; tools/check_bintr.exs reads the "key value" lines.
#   godot --path tests/bintr --script run_motion_step.gd -- --mode=probe|fixture|run --translate=yes|no
#         --elf=E --bintr=D [--glb=G --out=O]
extends SceneTree

const READ := 2
const COOP := 4
const DONE := 5
const ERROR := 6
const WAIT_GPU := 1
# SOMA30, the skeleton motion.elf returns, with a hand-made T-pose (metres, Y up, facing +Z).
const SOMA := [
	["Hips", -1, Vector3(0, 1.0, 0)], ["Spine1", 0, Vector3(0, 0.1, 0)],
	["Spine2", 1, Vector3(0, 0.12, 0)], ["Chest", 2, Vector3(0, 0.14, 0)],
	["Neck1", 3, Vector3(0, 0.16, 0)], ["Neck2", 4, Vector3(0, 0.05, 0)],
	["Head", 5, Vector3(0, 0.07, 0)], ["Jaw", 6, Vector3(0, 0.02, 0.05)],
	["LeftEye", 6, Vector3(0.03, 0.07, 0.08)], ["RightEye", 6, Vector3(-0.03, 0.07, 0.08)],
	["LeftShoulder", 3, Vector3(0.04, 0.12, 0)], ["LeftArm", 10, Vector3(0.14, 0, 0)],
	["LeftForeArm", 11, Vector3(0.28, 0, 0)], ["LeftHand", 12, Vector3(0.26, 0, 0)],
	["LeftHandThumbEnd", 13, Vector3(0.06, 0, 0.05)], ["LeftHandMiddleEnd", 13, Vector3(0.18, 0, 0)],
	["RightShoulder", 3, Vector3(-0.04, 0.12, 0)], ["RightArm", 16, Vector3(-0.14, 0, 0)],
	["RightForeArm", 17, Vector3(-0.28, 0, 0)], ["RightHand", 18, Vector3(-0.26, 0, 0)],
	["RightHandThumbEnd", 19, Vector3(-0.06, 0, 0.05)], ["RightHandMiddleEnd", 19, Vector3(-0.18, 0, 0)],
	["LeftLeg", 0, Vector3(0.1, -0.05, 0)], ["LeftShin", 22, Vector3(0, -0.43, 0)],
	["LeftFoot", 23, Vector3(0, -0.44, 0)], ["LeftToeBase", 24, Vector3(0, -0.06, 0.13)],
	["RightLeg", 0, Vector3(-0.1, -0.05, 0)], ["RightShin", 26, Vector3(0, -0.43, 0)],
	["RightFoot", 27, Vector3(0, -0.44, 0)], ["RightToeBase", 28, Vector3(0, -0.06, 0.13)],
]


func _initialize() -> void:
	var a := {}
	for arg in OS.get_cmdline_user_args():
		var kv := arg.trim_prefix("--").split("=", true, 1)
		a[kv[0]] = kv[1] if kv.size() > 1 else ""
	quit(_run(a))


func _run(a: Dictionary) -> int:
	if a.get("mode") == "fixture":
		return _fixture(a.glb)
	if not ClassDB.class_exists("Sandbox"):
		return _fail("the Sandbox class is not registered")
	ProjectSettings.set_setting("sandbox/binary_translation/cache_dir", a.bintr + "/")
	ProjectSettings.set_setting("sandbox/binary_translation/enabled", a.get("translate") == "yes")
	var sb = ClassDB.instantiate("Sandbox")
	sb.memory_max = 1024
	sb.references_max = 4096
	sb.allocations_max = 10000000
	var elf := FileAccess.get_file_as_bytes(a.elf)
	if elf.is_empty():
		return _fail("%s did not read" % a.elf)
	sb.load_buffer(elf)
	if not sb.has_function("skintokens_motion_start"):
		return _fail("%s did not load (no skintokens_motion_start)" % a.elf.get_file())
	print("hash %08X" % (int(sb.get_translation_hash()) & 0xFFFFFFFF))
	print("translated %s" % ("yes" if sb.is_binary_translated() else "no"))
	if a.get("mode") != "run":
		sb.free()
		return 0

	var clip := _clip()
	var out := PackedStringArray()
	out.append(str(sb.vmcall("skintokens_motion_start", a.glb, "walk", "spin 1.1", clip.names, clip.parents, clip.xf, 30.0)))
	var t0 := Time.get_ticks_usec()
	out.append(str(sb.vmcall("skintokens_motion_start", a.glb, "walk", "loop 1.1", clip.names, clip.parents, clip.xf, 30.0)))
	var f := FileAccess.open(a.glb, FileAccess.READ)
	var feed := PackedByteArray()
	var pumps := 0
	while pumps < 100000:
		var r = sb.vmcall("skintokens_pump", feed)
		feed = PackedByteArray()
		pumps += 1
		if typeof(r) != TYPE_ARRAY or r.size() < 3:
			return _fail("skintokens_pump returned %s" % str(r))
		var hdr: PackedInt64Array = r[0]
		if hdr[0] == READ:
			f.seek(hdr[1])
			feed = f.get_buffer(min(hdr[2] if hdr[2] > 0 else 1 << 30, f.get_length() - hdr[1]))
		elif hdr[0] == DONE or hdr[0] == ERROR:
			out.append("%s %s" % ["DONE" if hdr[0] == DONE else "ERROR", str(r[1]).strip_edges()])
			break
		elif hdr[0] != COOP and hdr[0] != WAIT_GPU:
			return _fail("unknown request kind %d" % hdr[0])
	print("vm_ms %d" % ((Time.get_ticks_usec() - t0) / 1000))
	var size := int(sb.vmcall("skintokens_result_size"))
	var glb := PackedByteArray()
	while glb.size() < size:
		var chunk: PackedByteArray = sb.vmcall("skintokens_result_chunk", glb.size(), min(size - glb.size(), 8 << 20))
		if chunk.is_empty():
			return _fail("the result did not read back")
		glb.append_array(chunk)
	var gw := FileAccess.open(a.out.get_basename() + ".glb", FileAccess.WRITE)
	if gw:
		gw.store_buffer(glb)
		gw.close()
	var ctx := HashingContext.new()
	ctx.start(HashingContext.HASH_SHA256)
	ctx.update(glb)
	out.append(str(sb.vmcall("skintokens_output")).strip_edges())
	out.append("RESULT %d bytes sha256 %s" % [glb.size(), ctx.finish().hex_encode()])
	print("pumps %d" % pumps)
	var text := "\n".join(out) + "\n"
	var w := FileAccess.open(a.out, FileAccess.WRITE)
	if w == null:
		return _fail("cannot write %s" % a.out)
	w.store_string(text)
	w.close()
	print("result_bytes %d" % text.to_utf8_buffer().size())
	print("translated_after %s" % ("yes" if sb.is_binary_translated() else "no"))
	sb.free()
	return 0


# Six seconds of a walk-like swing (33-frame period) that ramps up from rest and back down, moving forward at
# up to 1.5 m/s: the loop cut must find the period in the steady middle.
func _clip() -> Dictionary:
	var names := ""
	var parents := PackedInt32Array()
	for b in SOMA:
		names += b[0] + "\n"
		parents.append(b[1])
	var xf := PackedFloat32Array()
	var z := 0.0
	for fr in 180:
		var amp := clampf(fr / 30.0, 0.0, 1.0) * clampf((165.0 - fr) / 30.0, 0.0, 1.0)
		z += 1.5 * amp / 30.0
		var ph := TAU * fr / 33.0
		for j in SOMA.size():
			var n: String = SOMA[j][0]
			var ang := 0.0
			if n == "LeftLeg" or n == "RightArm":
				ang = 0.5 * amp * sin(ph)
			elif n == "RightLeg" or n == "LeftArm":
				ang = -0.5 * amp * sin(ph)
			elif n.ends_with("Shin") or n.ends_with("ForeArm"):
				ang = 0.3 * amp * (1.0 + sin(ph + float(j)))
			var m := Basis(Vector3.RIGHT, ang)
			var t: Vector3 = Vector3(0, 1.0 + 0.02 * amp * cos(2 * ph), z) if j == 0 else SOMA[j][2]
			xf.append_array([m.x.x, m.y.x, m.z.x, m.x.y, m.y.y, m.z.y, m.x.z, m.y.z, m.z.z, t.x, t.y, t.z])
	return {names = names, parents = parents, xf = xf}


# A rigged humanoid: the SOMA T-pose skeleton without the face, eye and finger-end joints, and one skinned box
# per bone.
func _fixture(path: String) -> int:
	var keep := []
	for b in SOMA:
		if not (b[0] in ["Jaw", "LeftEye", "RightEye"] or b[0].ends_with("End")):
			keep.append(b)
	var root := Node3D.new()
	root.name = "Fixture"
	var sk := Skeleton3D.new()
	sk.name = "Skeleton3D"
	root.add_child(sk)
	var idx := {}
	var glob := {}
	for b in keep:
		var i := sk.add_bone(b[0])
		idx[b[0]] = i
		var p := -1 if b[1] < 0 else int(idx[SOMA[b[1]][0]])
		if p >= 0:
			sk.set_bone_parent(i, p)
		sk.set_bone_rest(i, Transform3D(Basis(), b[2]))
		glob[i] = b[2] + (glob[p] if p >= 0 else Vector3.ZERO)
	sk.reset_bone_poses()
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	st.set_skin_weight_count(SurfaceTool.SKIN_4_WEIGHTS)
	var skin := Skin.new()
	for i in sk.get_bone_count():
		skin.add_named_bind(sk.get_bone_name(i), Transform3D(Basis(), -glob[i]))
		var box := BoxMesh.new()
		box.size = Vector3(0.06, 0.06, 0.06)
		var arr := box.get_mesh_arrays()
		var v: PackedVector3Array = arr[Mesh.ARRAY_VERTEX]
		var ix: PackedInt32Array = arr[Mesh.ARRAY_INDEX]
		for k in ix:
			st.set_bones(PackedInt32Array([i, 0, 0, 0]))
			st.set_weights(PackedFloat32Array([1, 0, 0, 0]))
			st.add_vertex(v[k] + glob[i])
	var mi := MeshInstance3D.new()
	mi.name = "Body"
	mi.mesh = st.commit()
	mi.skin = skin
	sk.add_child(mi)
	mi.owner = root
	sk.owner = root
	mi.skeleton = NodePath("..")
	var doc := GLTFDocument.new()
	var state := GLTFState.new()
	if doc.append_from_scene(root, state) != OK:
		return _fail("the fixture did not export")
	var bytes := doc.generate_buffer(state)
	var w := FileAccess.open(path, FileAccess.WRITE)
	if w == null:
		return _fail("cannot write %s" % path)
	w.store_buffer(bytes)
	w.close()
	print("fixture %d bytes" % bytes.size())
	root.free()
	return 0


func _fail(why: String) -> int:
	print("FAIL %s" % why)
	return 1
