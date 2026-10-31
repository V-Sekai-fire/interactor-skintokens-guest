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

This repository is licensed under the MIT License; see `LICENSE`. Vendored files keep their own licences: `vendor/skin-tokens-ggml/LICENSE` (Apache-2.0, with upstream notices in its `NOTICE`; SkinTokens is MIT) and the MIT notice in `vendor/nlohmann/json.hpp`.

## Run in a browser

`tests/web/` is the same rig on Godot's WebGPU RenderingDevice. Export it with `godot --headless --path tests/web --export-pack Web index.pck`, put that pck and `skintokens.elf`'s dependencies into a Godot web export built with WebGPU and godot-sandbox, and serve it with `serve_range.py` (cross-origin isolation plus HTTP range requests). Put the GGUF directory under `w/` and meshes under `m/`. `node rig.mjs chrome <url> out.glb log.txt` opens the page in a window and saves the rigged GLB.

The host fetches weights in 64 MiB range requests and caps ggml-rd's buffers at 128 MiB, WebGPU's default binding size.
