# skintokens.elf on its own: contract-guest-runtime's shared guest build, the model from vendor/skin-tokens-ggml.
cmake_minimum_required(VERSION 3.16)
project(interactor_skintokens_guest C CXX)

set(WEFT_ROOT "${CMAKE_SOURCE_DIR}/../.." CACHE PATH "Root of the goal manifest's checkout")
get_filename_component(WEFT_ROOT "${WEFT_ROOT}" ABSOLUTE)
set(GUEST_RUNTIME_ROOT ${WEFT_ROOT}/2-contract/guest-runtime CACHE PATH "contract-guest-runtime's checkout")
set(GUEST_RUNTIME_GGML ON CACHE BOOL "" FORCE)
if(NOT DEFINED GUEST_ELF_DIR)
	set(GUEST_ELF_DIR ${CMAKE_BINARY_DIR})
endif()

include(${GUEST_RUNTIME_ROOT}/cmake/guest_runtime.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/skintokens.cmake)
