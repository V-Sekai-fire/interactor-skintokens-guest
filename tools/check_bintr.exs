# skintokens.elf under godot-sandbox's libriscv, natively translated and interpreted, on the same weight-free motion
# step (a made-up SOMA30 clip cut and added to a small rigged GLB); every check has a control that must fail.
#   elixir tools/check_bintr.exs --elf=<skintokens.elf> [--work=<dir>]
defmodule CheckBintr do
  @root Path.expand("..", __DIR__)
  @project Path.join(@root, "tests/bintr")
  @godot_build "https://github.com/V-Sekai-fire/service-godot-build/releases/download"
  @engine %{
    linux: {"v20260930-double.1", "godot.linuxbsd.editor.double.x86_64", "37c6962fcc76d1773a596440f310f52ad5cf18011e74a712a5b465c3240e919b"},
    macos: {"v20260930-double.1", "godot.macos.editor.double.arm64", "5f6f2df33708d8f522e0cb5f40af58bd44c1eccb2dda7128081f4056b1de8dce"}
  }
  @addon %{
    linux: {"v20261019-addon.1", "libgodot_riscv.linux.template_release.double.x86_64.so", "8ab42de8462452e22bc091af0aedf2446934d8ee691adcd1bbccb9818642aed6"},
    macos: {"v20261019-addon.1", "libgodot_riscv.macos.template_release.double.universal", "14f542887f322bbe9847fa0f24734a46d68458db366f139118b45fc8eb7479f0"}
  }
  @wall_s 600
  @expected [
    "FAIL kind is loop or once, got spin 1.1",
    "STARTED motion walk",
    "DONE",
    "motion walk (loop): start ",
    ", 33 frames,",
    "humanoid confidence ",
    "RESULT "
  ]

  def main(argv) do
    {kv, _, _} = OptionParser.parse(argv, strict: [elf: :string, work: :string])
    elf = Path.expand(kv[:elf] || fail("no --elf"))
    File.exists?(elf) || fail("no #{elf}")
    work = Path.expand(kv[:work] || Path.join(@root, "build/bintr"))
    env = setup(host(), work)
    glb = Path.join(work, "fixture.glb")
    fx = godot(env, ["--mode=fixture", "--glb=#{glb}"])
    if fx.rc != 0 or not File.exists?(glb), do: fail("the fixture was not written:\n#{fx.out}")
    env = Map.put(env, :glb, glb)
    t = Path.join(work, "translated")
    i = Path.join(work, "interpreted")
    results = translated(env, elf, t) ++ interpreted(env, elf, i) ++ compare(t, i)
    failed = Enum.count(results, &(&1 != :ok))
    say("#{length(results) - failed} of #{length(results)} checks pass")
    if failed > 0, do: System.halt(1)
  end

  defp translated(env, elf, dir) do
    reset(dir)
    lib = Path.join(dir, "bintr")
    src = Path.join(dir, "bintr-c")
    File.mkdir_p!(lib)
    File.mkdir_p!(src)
    emit = godot(env, ["--mode=probe", "--translate=yes", "--elf=#{elf}", "--bintr=#{lib}"], [{"GODOT_SANDBOX_BINTR_EMIT", src}])
    hash = emit["hash"] || fail("the emit probe printed no hash:\n#{emit.out}")
    c = Path.join(src, "bintr-#{hash}.c")
    File.exists?(c) || fail("the addon wrote no #{Path.basename(c)} (#{inspect(File.ls!(src))})")
    say("translation: #{Path.basename(c)}, #{File.stat!(c).size} bytes of C")
    sab = Path.join(dir, "sabotaged")
    sab_c = Path.join(src, "sabotaged-#{hash}.c")
    File.mkdir_p!(sab)
    File.write!(sab_c, sabotage(File.read!(c)))
    [{c, lib, "-O2"}, {sab_c, sab, "-O0"}]
    |> Task.async_stream(fn {from, to, opt} -> compile(env, from, Path.join(to, "bintr-#{hash}#{env.suffix}"), opt) end,
                         timeout: :infinity)
    |> Stream.run()

    probe = fn e, l -> godot(env, ["--mode=probe", "--translate=yes", "--elf=#{e}", "--bintr=#{l}"]) end
    loaded = probe.(elf, lib)
    disabled = godot(env, ["--mode=probe", "--translate=no", "--elf=#{elf}", "--bintr=#{lib}"])
    absent = Path.join(dir, "no-bintr")
    File.mkdir_p!(absent)
    missing = probe.(elf, absent)
    bytes = File.read!(elf)
    flipped = Path.join(dir, "flipped.elf")
    File.write!(flipped, flip_text(bytes))
    flip = probe.(flipped, lib)
    truncated = Path.join(dir, "truncated.elf")
    File.write!(truncated, binary_part(bytes, 0, div(byte_size(bytes), 2)))
    trunc = probe.(truncated, lib)

    run = full_run(env, elf, dir, "yes", lib)
    sab_on = full_run(env, elf, Path.join(dir, "sab-on"), "yes", sab)
    sab_off = full_run(env, elf, Path.join(dir, "sab-off"), "no", sab)
    good = File.read(Path.join(dir, "out.txt"))
    [
      check("the loaded program carries the native translation (hash #{hash})", loads_translated(loaded, hash)),
      check("control: with binary translation disabled it is refused", refused(loads_translated(disabled, hash))),
      check("control: with no library for the hash it is refused", refused(loads_translated(missing, hash))),
      check("control: an ELF with one .text byte flipped is refused", refused(loads_translated(flip, hash))),
      check("control: an ELF cut in half is refused", refused(loads_translated(trunc, hash))),
      check("the translated run finished on its translation", ran(run, hash, "yes")),
      check("the translated code ran the calls: a translation with its sqrt sabotaged changes the outputs",
            sabotaged(sab_on, good, "yes")),
      check("control: with translation disabled the sabotaged library changes nothing, and the check refuses it", refused(sabotaged(sab_off, good, "no")))
    ]
  end

  # Every FSQRT the translation executes goes through api.sqrtf32; the sabotaged copy answers x itself.
  defp sabotage(c) do
    hook = "\tapi = *table;\n"
    String.contains?(c, hook) && String.contains?(c, "api.sqrtf32(") || fail("the translation has no api.sqrtf32 to sabotage")
    "static float sabotaged_sqrtf32(float x) { return x; }\n" <>
      String.replace(c, hook, hook <> "\tapi.sqrtf32 = sabotaged_sqrtf32;\n", global: false)
  end

  defp sabotaged(%{rc: rc} = r, good, translated) do
    out = File.read(Path.join(r.dir, "out.txt"))
    cond do
      r["translated"] != translated -> {:error, "translated #{r["translated"]}, not #{translated}"}
      rc == 0 and out == good -> {:error, "the outputs equal the unsabotaged translation's"}
      true -> say("  the outputs changed (rc #{rc})")
    end
  end

  defp interpreted(env, elf, dir) do
    reset(dir)
    lib = Path.join(dir, "bintr")
    File.mkdir_p!(lib)
    run = full_run(env, elf, dir, "no", lib)
    [check("the interpreted run finished interpreted", ran(run, run["hash"], "no"))]
  end

  defp full_run(env, elf, dir, translate, lib) do
    File.mkdir_p!(dir)
    out = Path.join(dir, "out.txt")
    r = godot(env, ["--mode=run", "--translate=#{translate}", "--elf=#{elf}", "--bintr=#{lib}", "--glb=#{env.glb}",
                    "--out=#{out}"])
    report = Map.drop(r, [:out, :dir]) |> Map.put("elf_sha256", sha256(File.read!(elf))) |> Map.put("rc", to_string(r.rc))
    File.write!(Path.join(dir, "report.txt"), Enum.map_join(Enum.sort(report), "", fn {k, v} -> "#{k} #{v}\n" end))
    File.write!(Path.join(dir, "godot.log"), r.out)
    say("#{Path.basename(dir)}: rc #{r.rc}, translated #{r["translated"]}, vm_ms #{r["vm_ms"]} for the motion step")
    Map.put(r, :dir, dir)
  end

  defp compare(tdir, idir) do
    t = read_report(tdir)
    i = read_report(idir)
    tout = File.read(Path.join(tdir, "out.txt"))
    iout = File.read(Path.join(idir, "out.txt"))
    changed = with {:ok, b} <- iout, true <- byte_size(b) > 0, do: {:ok, flip_byte(b, div(byte_size(b), 2))}, else: (_ -> iout)
    planted = with {:ok, b} <- tout, do: {:ok, String.replace(b, "STARTED motion walk", "STARTED motion run")}
    [
      check("both runs ran the same ELF (sha256 #{t["elf_sha256"]})", same(t, i, "elf_sha256")),
      check("both runs loaded the same program (hash #{t["hash"]})", same(t, i, "hash")),
      check("the translated outputs equal the interpreter's, byte for byte", equal(tout, iout)),
      check("control: the interpreter's outputs with one byte changed are refused", refused(equal(tout, changed))),
      check("the outputs carry every expected answer, in order", expected(tout)),
      check("control: outputs with a changed answer are refused", refused(expected(planted)))
    ]
  end

  defp loads_translated(%{rc: 0} = r, hash) do
    cond do
      r["hash"] != hash -> {:error, "hash #{r["hash"]}, not #{hash}"}
      r["translated"] != "yes" -> {:error, "is_binary_translated() is false"}
      true -> :ok
    end
  end

  defp loads_translated(r, _), do: {:error, "godot exited #{r.rc}: #{r["FAIL"] || "no FAIL line"}"}

  defp ran(%{rc: 0} = r, hash, translated) do
    cond do
      r["hash"] != hash -> {:error, "hash #{r["hash"]}, not #{hash}"}
      r["translated"] != translated or r["translated_after"] != translated ->
        {:error, "translated #{r["translated"]} before and #{r["translated_after"]} after, not #{translated}"}
      true -> :ok
    end
  end

  defp ran(r, _, _), do: {:error, "godot exited #{r.rc}: #{r["FAIL"] || "no FAIL line"}"}

  defp same(t, i, key) do
    if t[key] != nil and t[key] == i[key], do: :ok, else: {:error, "#{inspect(t[key])} and #{inspect(i[key])}"}
  end

  defp equal({:ok, a}, {:ok, a}) when byte_size(a) > 0, do: say("  #{byte_size(a)} bytes, sha256 #{sha256(a)}")
  defp equal({:ok, a}, {:ok, b}), do: {:error, "#{byte_size(a)} and #{byte_size(b)} bytes, first difference at #{first_diff(a, b)}"}
  defp equal(a, b), do: {:error, "an output is missing: #{inspect(elem(a, 0))}, #{inspect(elem(b, 0))}"}

  defp expected({:ok, b}) do
    Enum.reduce_while(@expected, {:ok, 0}, fn want, {:ok, at} ->
      case :binary.match(b, want, scope: {at, byte_size(b) - at}) do
        {pos, len} -> {:cont, {:ok, pos + len}}
        :nomatch -> {:halt, {:error, "no #{inspect(want)} after byte #{at}"}}
      end
    end)
    |> case do
      {:ok, _} -> :ok
      e -> e
    end
  end

  defp expected(_), do: {:error, "no output"}

  defp host do
    case {:os.type(), to_string(:erlang.system_info(:system_architecture))} do
      {{:unix, :linux}, "x86_64" <> _} -> :linux
      {{:unix, :darwin}, _} -> :macos
      other -> fail("no pinned double engine and addon for #{inspect(other)}")
    end
  end

  defp setup(host, work) do
    cache = Path.join(work, "cache")
    {etag, ename, esha} = @engine[host]
    engine = fetch("#{@godot_build}/#{etag}/#{ename}", Path.join(cache, ename), esha)
    File.chmod!(engine, 0o755)
    {atag, aname, asha} = @addon[host]
    addon = fetch("#{@godot_build}/#{atag}/#{aname}", Path.join(cache, aname), asha)
    File.cp!(addon, Path.join(@project, "addons/godot_sandbox/bin/#{aname}"))
    File.mkdir_p!(Path.join(@project, ".godot"))
    File.write!(Path.join(@project, ".godot/extension_list.cfg"), "res://addons/godot_sandbox/bin/godot-riscv.gdextension\n")
    {suffix, flags} = if host == :macos, do: {".dylib", ~w(-dynamiclib)}, else: {".so", ~w(-fPIC)}
    xvfb = host == :linux and System.get_env("DISPLAY") in [nil, ""] and System.find_executable("xvfb-run")
    %{engine: engine, suffix: suffix, flags: flags, host: host, xvfb: xvfb, cc: System.get_env("BINTR_CC") || "clang"}
  end

  defp fetch(url, path, sha) do
    unless File.exists?(path) and sha256(File.read!(path)) == sha do
      File.mkdir_p!(Path.dirname(path))
      part = path <> ".part"
      say("$ curl #{url}")
      {out, rc} = System.cmd("curl", ["-fsSL", "--retry", "3", "-o", part, url], stderr_to_stdout: true)
      if rc != 0, do: fail("curl exited #{rc}: #{out}")
      got = sha256(File.read!(part))
      if got != sha, do: fail("#{Path.basename(path)}: sha256 #{got}, pinned #{sha}")
      File.rename!(part, path)
    end
    say("#{Path.basename(path)}: sha256 #{sha}")
    path
  end

  defp compile(env, c, lib, opt) do
    args = [opt | ~w(-s -std=c99 -shared -x c -fexceptions -fvisibility=hidden -fomit-frame-pointer)] ++ env.flags ++ [c, "-o", lib]
    say("$ #{env.cc} #{Enum.join(args, " ")}")
    t0 = System.monotonic_time(:millisecond)
    {out, rc} = System.cmd(env.cc, args, stderr_to_stdout: true)
    if rc != 0, do: fail("#{env.cc} exited #{rc}: #{out}")
    say("translation compiled in #{System.monotonic_time(:millisecond) - t0} ms: #{Path.basename(lib)}")
  end

  # Godot under a wall clock; the window goes to xvfb on a Linux runner without a display.
  defp godot(env, args, extra_env \\ []) do
    driver = if env.host == :linux, do: ["--rendering-driver", "opengl3"], else: []
    cmd = [env.engine, "--audio-driver", "Dummy"] ++ driver ++ ["--path", @project, "--script", "run_motion_step.gd", "--"] ++ args
    cmd = if env.xvfb, do: [env.xvfb, "-a" | cmd], else: cmd
    say("$ #{Enum.join(cmd, " ")}")
    port = Port.open({:spawn_executable, hd(cmd)}, [:binary, :exit_status, :stderr_to_stdout, args: tl(cmd),
                                                   env: Enum.map(extra_env, fn {k, v} -> {~c"#{k}", ~c"#{v}"} end)])
    deadline = System.monotonic_time(:millisecond) + @wall_s * 1000
    {out, rc} = collect(port, deadline, [])
    lines = for l <- String.split(out, "\n"), [k, v] <- [String.split(String.trim(l), " ", parts: 2)], do: {k, v}
    Map.merge(Map.new(lines), %{out: out, rc: rc})
  end

  defp collect(port, deadline, acc) do
    receive do
      {^port, {:data, d}} -> collect(port, deadline, [acc, d])
      {^port, {:exit_status, rc}} -> {IO.iodata_to_binary(acc), rc}
    after
      max(deadline - System.monotonic_time(:millisecond), 0) ->
        {:os_pid, pid} = Port.info(port, :os_pid)
        System.cmd("kill", ["-9", to_string(pid)])
        {IO.iodata_to_binary(acc) <> "\nFAIL past the wall clock\n", 124}
    end
  end

  # The byte at the middle of .text, low bit flipped: the code, and so the translation hash, change.
  defp flip_text(elf) do
    <<_::binary-size(0x28), shoff::little-64, _::binary-size(10), entsize::little-16, n::little-16, shstrndx::little-16, _::binary>> = elf
    shdr = fn i -> <<name::little-32, _::binary-size(20), off::little-64, size::little-64, _::binary>> = binary_part(elf, shoff + i * entsize, entsize); {name, off, size} end
    {_, stroff, _} = shdr.(shstrndx)
    {_, off, size} =
      Enum.find_value(0..(n - 1), fn i ->
        {name, _, _} = s = shdr.(i)
        if binary_part(elf, stroff + name, 6) == ".text" <> <<0>>, do: s
      end) || fail("no .text section")
    flip_byte(elf, off + div(size, 2))
  end

  defp flip_byte(b, at), do: binary_part(b, 0, at) <> <<Bitwise.bxor(:binary.at(b, at), 1)>> <> binary_part(b, at + 1, byte_size(b) - at - 1)

  defp first_diff(a, b) do
    Enum.find(0..(min(byte_size(a), byte_size(b)) - 1)//1, min(byte_size(a), byte_size(b)), &(:binary.at(a, &1) != :binary.at(b, &1)))
  end

  defp read_report(dir) do
    case File.read(Path.join(dir, "report.txt")) do
      {:ok, s} -> Map.new(for l <- String.split(s, "\n", trim: true), [k, v] <- [String.split(l, " ", parts: 2)], do: {k, v})
      _ -> %{}
    end
  end

  defp reset(dir) do
    File.rm_rf!(dir)
    File.mkdir_p!(dir)
  end

  defp refused(:ok), do: {:error, "accepted"}
  defp refused({:error, why}), do: say("  refused: #{why}")

  defp check(name, :ok), do: (say("PASS #{name}"); :ok)
  defp check(name, {:error, why}), do: (say("FAIL #{name}: #{why}"); :fail)

  defp sha256(b), do: Base.encode16(:crypto.hash(:sha256, b), case: :lower)
  defp say(msg), do: (IO.puts("== #{msg}"); :ok)

  defp fail(msg) do
    IO.puts(:stderr, "check_bintr: #{msg}")
    System.halt(1)
  end
end

CheckBintr.main(System.argv())
