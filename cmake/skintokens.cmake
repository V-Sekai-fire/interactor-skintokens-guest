# skintokens.elf: SkinTokens auto-rigging on ggml-rd, the model from vendor/skin-tokens-ggml unchanged.
get_filename_component(SKINTOKENS_GUEST_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_st ${SKINTOKENS_GUEST_ROOT}/vendor/skin-tokens-ggml)
add_stage_elf(skintokens
	${SKINTOKENS_GUEST_ROOT}/guest/skintokens/main.cpp
	${_st}/src/backend.cpp
	${_st}/src/binding.cpp
	${_st}/src/file_source.cpp
	${_st}/src/gguf.cpp
	${_st}/src/glb.cpp
	${_st}/src/mesh_encoder.cpp
	${_st}/src/model.cpp
	${_st}/src/qwen.cpp
	${_st}/src/retarget.cpp
	${_st}/src/skin_vae.cpp
	${_st}/src/skinning.cpp
	${_st}/src/tokenizer.cpp
	${_st}/src/weights.cpp
)
set_target_properties(skintokens PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
target_include_directories(skintokens PRIVATE ${GUEST_RUNTIME_ROOT}/guest ${_st}/include ${_st}/src
	${SKINTOKENS_GUEST_ROOT}/vendor)
target_link_libraries(skintokens PRIVATE ggml_rd pump ggml ggml-base)
