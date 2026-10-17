# Builds skintokens.elf from this checkout and the goal manifest's sibling checkouts (2-contract/guest-runtime,
# ggml-rd and ggml), the way transport-meshing-pen's build.sh builds its stage ELFs.
#   elixir tools/build.exs [--sysroot=<repository-riscv64-sysroot>] [--build=<dir>]
defmodule Build do
  @root Path.expand("..", __DIR__)

  def main(argv) do
    {kv, _, _} = OptionParser.parse(argv, strict: [sysroot: :string, build: :string])
    weft = Path.expand(System.get_env("WEFT_ROOT") || Path.join(@root, "../.."))
    for t <- ~w(cmake ninja clang++ ld.lld slangc spirv-val python3 bash), do: System.find_executable(t) || fail("#{t} is not on PATH")
    sysroot = Path.expand(kv[:sysroot] || System.get_env("RISCV64_SYSROOT") || Path.join(weft, "5-repository/riscv64-sysroot"))
    toolchain = Path.join(sysroot, "toolchain.cmake")
    File.exists?(toolchain) || fail("no toolchain.cmake under #{sysroot}")
    for repo <- ~w(2-contract/guest-runtime 2-contract/ggml-rd 2-contract/ggml) do
      File.dir?(Path.join(weft, repo)) || fail("no #{repo} under #{weft} (the goal manifest's sibling checkout)")
    end
    build = Path.expand(kv[:build] || Path.join(@root, "build/rv64"))
    File.mkdir_p!(build)

    env = [{"BUILD_DIR", build}, {"PYTHON", "python3"}, {"GUEST_RUNTIME_ROOT", Path.join(weft, "2-contract/guest-runtime")}]
    ref = keep_inline_emit(Path.join(weft, "2-contract/ggml-rd"), Path.join(build, "prelude"))
    run("bash", [Path.join(weft, "2-contract/ggml-rd/kernels/ggml/gen.sh"), "--no-emit"], env)
    run("python3", [Path.join(weft, "2-contract/guest-runtime/tools/inline_prelude.py"), Path.join(weft, "2-contract/ggml-rd"), ref])
    unless File.exists?(Path.join(build, "build.ninja")) do
      run("cmake", ["-S", @root, "-B", build, "-G", "Ninja", "-DCMAKE_TOOLCHAIN_FILE=#{toolchain}", "-DWEFT_ROOT=#{weft}"])
    end
    run("cmake", ["--build", build, "--target", "skintokens"])
    elf = File.read!(Path.join(build, "skintokens.elf"))
    IO.puts("== skintokens.elf: #{byte_size(elf)} bytes, sha256 #{Base.encode16(:crypto.hash(:sha256, elf), case: :lower)}")
  end

  # A Linux slangc rewrites every emit with an include of its prelude; one committed emit kept aside restores the inline form.
  defp keep_inline_emit(repo, ref) do
    dir = Path.join(ref, "kernels/ggml/cpp")
    unless Path.wildcard(Path.join(dir, "*_emit.cpp")) != [] do
      src = Enum.find(Path.wildcard(Path.join(repo, "kernels/ggml/cpp/*_emit.cpp")), &String.starts_with?(File.read!(&1), "#ifndef SLANG_CPP_PRELUDE_H")) ||
        fail("no committed emit with the inline prelude under #{repo}")
      File.mkdir_p!(dir)
      File.cp!(src, Path.join(dir, Path.basename(src)))
    end
    ref
  end

  defp run(cmd, args, env \\ []) do
    IO.puts("$ #{cmd} #{Enum.join(args, " ")}")
    {_, rc} = System.cmd(cmd, args, env: env, into: IO.stream(:stdio, :line), stderr_to_stdout: true, cd: @root)
    if rc != 0, do: fail("#{cmd} exited #{rc}")
  end

  defp fail(msg) do
    IO.puts(:stderr, "build: #{msg}")
    System.halt(1)
  end
end

Build.main(System.argv())
