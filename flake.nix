{
  description = "skintokens.elf, built offline from this repository";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    guest-runtime = {
      url = "github:V-Sekai-fire/contract-guest-runtime";
      flake = false;
    };
    ggml-rd = {
      url = "github:V-Sekai-fire/contract-ggml-rd";
      flake = false;
    };
    ggml = {
      url = "github:V-Sekai-fire/ggml";
      flake = false;
    };
    riscv64-sysroot = {
      url = "github:V-Sekai-fire/repository-riscv64-sysroot";
      flake = false;
    };
    # The slangc release the guest workflow pins.
    slang = {
      url = "https://github.com/shader-slang/slang/releases/download/v2026.13.1/slang-2026.13.1-linux-x86_64.tar.gz";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, ... }@inputs:
    let
      system = "x86_64-linux";
      # Where tools/build.exs looks for each sibling under WEFT_ROOT.
      siblings = {
        "2-contract/guest-runtime" = inputs.guest-runtime;
        "2-contract/ggml-rd" = inputs.ggml-rd;
        "2-contract/ggml" = inputs.ggml;
        "5-repository/riscv64-sysroot" = inputs.riscv64-sysroot;
      };

      slangc = { stdenv, autoPatchelfHook, zlib, libx11 }:
        stdenv.mkDerivation {
          pname = "slangc-bin";
          version = "2026.13.1";
          src = inputs.slang;
          nativeBuildInputs = [ autoPatchelfHook ];
          buildInputs = [ stdenv.cc.cc.lib zlib libx11 ];
          dontConfigure = true;
          dontBuild = true;
          installPhase = ''
            mkdir -p $out
            cp -r bin lib $out/
          '';
        };

      skintokens = { lib, stdenvNoCC, cmake, ninja, python3, bash, elixir
        , spirv-tools, llvmPackages, slangc-bin }:
        let
          # The build writes generated kernels into ggml-rd, so the siblings are copies.
          place = path: src: ''
            mkdir -p weft/${dirOf path}
            cp -r ${src} weft/${path}
          '';
          pin = path: src: "${path} ${src.rev}";
        in
        stdenvNoCC.mkDerivation {
          pname = "skintokens-elf";
          version = self.shortRev or "dirty";
          src = lib.cleanSourceWith {
            src = self;
            filter = path: _:
              !(builtins.elem (baseNameOf path) [
                "flake.nix"
                "flake.lock"
                ".github"
              ]);
          };
          nativeBuildInputs = [
            cmake
            ninja
            python3
            bash
            elixir
            spirv-tools
            slangc-bin
            llvmPackages.clang-unwrapped
            llvmPackages.lld
            llvmPackages.bintools-unwrapped
          ];
          dontUseCmakeConfigure = true;
          # The guest ELF ships as the linker wrote it.
          dontFixup = true;
          buildPhase = ''
            export HOME=$TMPDIR LANG=C.UTF-8
            ${lib.concatStrings (lib.mapAttrsToList place siblings)}
            chmod -R u+w weft
            export WEFT_ROOT=$PWD/weft
            elixir tools/build.exs --build=$PWD/rv64
          '';
          installPhase = ''
            mkdir -p $out
            cp rv64/skintokens.elf $out/skintokens.elf
            cat > $out/skintokens.elf.siblings.txt <<'TXT'
            ${lib.concatStringsSep "\n" (lib.mapAttrsToList pin siblings)}
            TXT
          '';
        };

      overlay = final: _: {
        slangc-bin = final.callPackage slangc { };
        skintokens-elf = final.callPackage skintokens {
          llvmPackages = final.llvmPackages_21;
        };
      };
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ overlay ];
      };
    in
    {
      overlays.default = overlay;
      packages.${system} = {
        skintokens-elf = pkgs.skintokens-elf;
        default = pkgs.skintokens-elf;
      };
    };
}
