# Test helpers for moss-transcribe.cpp. Individual tests are registered from
# Task 4 onwards via mt_add_test(); none are registered yet.
macro(mt_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE moss-transcribe)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endmacro()

mt_add_test(test_audio_io)
mt_add_test(test_cpu_softmax)
mt_add_test(test_cpu_threads)
mt_add_test(test_cpu_q8)
target_include_directories(test_cpu_q8 PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ggml/src)
add_test(NAME test_cpu_q8_input_lifetime COMMAND test_cpu_q8 --audit-unretained-input)
set_tests_properties(test_cpu_q8_input_lifetime PROPERTIES SKIP_RETURN_CODE 77)
mt_add_test(test_cpu_activation)
mt_add_test(test_cpu_shared_layer)
add_test(NAME test_cpu_fused_layer COMMAND test_cpu_shared_layer --fused)
set_tests_properties(test_cpu_fused_layer PROPERTIES SKIP_RETURN_CODE 77)
mt_add_test(test_cpu_projection)
mt_add_test(test_cpu_encoder_q8)
target_include_directories(test_cpu_encoder_q8 PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ggml/src)
mt_add_test(test_cpu_encoder_qkv)
target_include_directories(test_cpu_encoder_qkv PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ggml/src)
add_executable(moss_encoder_vnni_gate ${CMAKE_SOURCE_DIR}/cpu-lab/moss_encoder_vnni_gate.cpp)
target_include_directories(moss_encoder_vnni_gate PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(moss_encoder_vnni_gate PRIVATE moss-transcribe)

# Model-backed state/bit audit, deliberately outside CTest. A mismatch returns
# failure and forbids timing; synthetic fixtures do not establish audio parity.
add_executable(moss_target_batch_gate ${CMAKE_SOURCE_DIR}/cpu-lab/moss_target_batch_gate.cpp)
target_include_directories(moss_target_batch_gate PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(moss_target_batch_gate PRIVATE moss-transcribe)
add_executable(moss_target_batch_trace ${CMAKE_SOURCE_DIR}/cpu-lab/moss_target_batch_trace.cpp)
target_include_directories(moss_target_batch_trace PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party/ggml/src)
target_link_libraries(moss_target_batch_trace PRIVATE moss-transcribe)
add_test(NAME moss_target_batch_trace_metadata COMMAND moss_target_batch_trace --audit-metadata)
add_test(NAME moss_target_batch_context_layout COMMAND moss_target_batch_trace --audit-context-layout)
target_include_directories(test_cpu_projection PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ggml/src)
target_include_directories(test_cpu_activation PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ggml/src)

# Research executable, deliberately outside CTest: it reports arithmetic drift
# from packed kernels and returns failure when raw float bits differ.
add_executable(moss_packed_q8_bench ../cpu-lab/moss_packed_q8_bench.cpp)
target_link_libraries(moss_packed_q8_bench PRIVATE moss-transcribe)

# Strict arithmetic probe, separate from the AMX path. Runtime skips on hosts
# without the required ISA; graph integration needs its own end-to-end gate.
add_executable(moss_exact_q8_bench ../cpu-lab/moss_exact_q8_bench.cpp)
target_include_directories(moss_exact_q8_bench PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ggml/src)
target_link_libraries(moss_exact_q8_bench PRIVATE moss-transcribe)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(moss_exact_q8_bench PRIVATE -march=native -ffp-contract=off)
  endif()
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(moss_exact_q8_bench PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()
add_test(NAME moss_exact_q8_bench COMMAND moss_exact_q8_bench)
set_tests_properties(moss_exact_q8_bench PROPERTIES SKIP_RETURN_CODE 77)

# Sparse AMX integer lanes preserve the pinned Q8 reduction tree. Deliberately
# outside CTest until the arithmetic probe has run on supported hardware.
add_executable(moss_amx_q8_lanes ../cpu-lab/moss_amx_q8_lanes.cpp)
target_include_directories(moss_amx_q8_lanes PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ggml/src)
target_link_libraries(moss_amx_q8_lanes PRIVATE moss-transcribe)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(moss_amx_q8_lanes PRIVATE -march=native -ffp-contract=off)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(moss_amx_q8_lanes PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# Dense input columns, four-code AMX reductions and original Q8 FMA order.
# Owned-vector/ordinary-graph research probe; no model routing or CTest default.
add_executable(moss_amx_q8_batch ../cpu-lab/moss_amx_q8_batch.cpp)
target_include_directories(moss_amx_q8_batch PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ggml/src)
target_link_libraries(moss_amx_q8_batch PRIVATE moss-transcribe)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(moss_amx_q8_batch PRIVATE -march=native -ffp-contract=off)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(moss_amx_q8_batch PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# Standalone exact dense VNNI batch probe; no MOSS model routing.
add_executable(moss_vnni_q8_batch ${CMAKE_SOURCE_DIR}/cpu-lab/moss_vnni_q8_batch.cpp)
target_include_directories(moss_vnni_q8_batch PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ggml/src)
target_link_libraries(moss_vnni_q8_batch PRIVATE moss-transcribe)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(moss_vnni_q8_batch PRIVATE -march=native -ffp-contract=off)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(moss_vnni_q8_batch PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

mt_add_test(test_mel_cpu)

# Paired serial/single-team packing experiment; outside default CTest.
add_executable(moss_vnni_q8_parallel ${CMAKE_SOURCE_DIR}/cpu-lab/moss_vnni_q8_parallel.cpp)
target_include_directories(moss_vnni_q8_parallel PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ggml/src)
target_link_libraries(moss_vnni_q8_parallel PRIVATE moss-transcribe)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(moss_vnni_q8_parallel PRIVATE -march=native -ffp-contract=off)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(moss_vnni_q8_parallel PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# Exact inline-half conversion probe with paired unchanged-kernel controls.
add_executable(moss_vnni_q8_inline ${CMAKE_SOURCE_DIR}/cpu-lab/moss_vnni_q8_inline.cpp)
target_include_directories(moss_vnni_q8_inline PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ggml/src)
target_link_libraries(moss_vnni_q8_inline PRIVATE moss-transcribe)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(moss_vnni_q8_inline PRIVATE -march=native -ffp-contract=off)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(moss_vnni_q8_inline PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# Exact input-panel-first traversal with paired prior scheduling controls.
add_executable(moss_vnni_q8_cache ${CMAKE_SOURCE_DIR}/cpu-lab/moss_vnni_q8_cache.cpp)
target_include_directories(moss_vnni_q8_cache PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ggml/src)
target_link_libraries(moss_vnni_q8_cache PRIVATE moss-transcribe)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(moss_vnni_q8_cache PRIVATE -march=native -ffp-contract=off)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(moss_vnni_q8_cache PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# F32-input custom graph parity and protected timing, outside default CTest.
add_executable(moss_vnni_q8_f32 ${CMAKE_SOURCE_DIR}/cpu-lab/moss_vnni_q8_f32.cpp)
target_include_directories(moss_vnni_q8_f32 PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ggml/src)
target_link_libraries(moss_vnni_q8_f32 PRIVATE moss-transcribe)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(moss_vnni_q8_f32 PRIVATE -march=native -ffp-contract=off)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(moss_vnni_q8_f32 PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

mt_add_test(test_mel)
set_tests_properties(test_mel PROPERTIES LABELS "model")

mt_add_test(test_whisper_encoder)
set_tests_properties(test_whisper_encoder PROPERTIES LABELS "model")

mt_add_test(test_audio_adaptor)
set_tests_properties(test_audio_adaptor PROPERTIES LABELS "model")

mt_add_test(test_audio_path_e2e)
set_tests_properties(test_audio_path_e2e PROPERTIES LABELS "model")

mt_add_test(test_tokenizer)
set_tests_properties(test_tokenizer PROPERTIES LABELS "model")

mt_add_test(test_audio_span)
set_tests_properties(test_audio_span PROPERTIES LABELS "model")

mt_add_test(test_qwen3_decoder)
set_tests_properties(test_qwen3_decoder PROPERTIES LABELS "model")

mt_add_test(test_fusion)
set_tests_properties(test_fusion PROPERTIES LABELS "model")

mt_add_test(test_generate_e2e)
set_tests_properties(test_generate_e2e PROPERTIES LABELS "model")

# Embed lookup vs reference dequant for all token_embd types (LocalAI#10862
# CUDA K-quant GET_ROWS regression) -- model-INDEPENDENT (no LABELS "model").
mt_add_test(test_embed_lookup)

# M6: subtitle + diarization export -- model-INDEPENDENT (no LABELS "model").
mt_add_test(test_transcript_parser)
mt_add_test(test_subtitle_postprocess)
mt_add_test(test_subtitle_export)
