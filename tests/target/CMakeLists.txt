cmake_minimum_required(VERSION 3.22...4.2)
project(ttsim_copy_test)

add_executable(ttsim_copy_test)
target_sources(ttsim_copy_test PRIVATE ttsim_copy_test.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(ttsim_copy_test PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    target_precompile_headers(ttsim_copy_test REUSE_FROM TT::CommonPCH)
endif()
