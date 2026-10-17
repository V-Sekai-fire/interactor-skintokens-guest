# interactor-skintokens-guest

SkinTokens auto-rigging on ggml-rd, built as a sandboxed guest program. The host passes it a mesh and the model's weights, and it returns the rigged model as a GLB.

## What it is for

The guest has no file system. The host reads the mesh and the GGUF metadata for it through the pump, and copies each weight from its file straight into a GPU buffer. The model code is skin-tokens-ggml's, vendored under `vendor/skin-tokens-ggml`. Its file input goes through `runtime_options::files`, and its GLB input and output go through the memory API, so the guest changes nothing in it.

## Build and run

`elixir tools/build.exs` builds `skintokens.elf` with contract-guest-runtime's shared guest build, against the sibling checkouts that the goal manifest puts beside this one. It needs an LLVM clang with the RISC-V target; Apple's clang has none.

`tests/rig/run_rig.gd` rigs one GLB in a Godot window on a local RenderingDevice:

```sh
godot --path tests/rig --resolution 64x64 --script run_rig.gd -- \
  --elf=<skintokens.elf> --models=<GGUF dir> --mesh=<in.glb> --out=<out.glb> --opts="beams=1"
```

`--cpu` attaches no RenderingDevice, and ggml-rd then runs its CPU kernels. The `opts` string takes `beams`, `seed`, `temperature` and `max_tokens`. Any other `K=V` is set in the environment, for example `GGML_RD_FAULT=1`.

## Licence

The licence is not stated. skin-tokens-ggml and the SkinTokens model are MIT; `vendor/nlohmann/json.hpp` is MIT.
