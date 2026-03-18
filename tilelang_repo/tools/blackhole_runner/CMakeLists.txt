cmake_minimum_required(VERSION 3.22...4.2)
project(tilelang_blackhole_runner)

add_executable(tilelang_blackhole_runner)
target_sources(tilelang_blackhole_runner PRIVATE runner.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(tilelang_blackhole_runner PUBLIC TT::Metalium)

if(TARGET TT::CommonPCH)
    target_precompile_headers(tilelang_blackhole_runner REUSE_FROM TT::CommonPCH)
endif()
