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

mt_add_test(test_mel_cpu)

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
