cmake_minimum_required(VERSION 3.22...4.2)
project(tilelang_blackhole_runner)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TT_METAL_HOME_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/../../../tt_metal_repo")
if(DEFINED ENV{TT_METAL_HOME})
    set(TT_METAL_HOME_DEFAULT "$ENV{TT_METAL_HOME}")
endif()
set(TT_METAL_HOME "${TT_METAL_HOME_DEFAULT}" CACHE PATH "Path to TT-Metal repository")
set(TT_METAL_BUILD_DIR "${TT_METAL_HOME}/build_Release" CACHE PATH "Path to TT-Metal build directory")

add_executable(tilelang_blackhole_runner)
target_sources(tilelang_blackhole_runner PRIVATE runner.cpp)
target_compile_features(tilelang_blackhole_runner PRIVATE cxx_std_20)

find_package(nlohmann_json REQUIRED)

target_compile_definitions(tilelang_blackhole_runner PRIVATE
    ENCHANTUM_ENABLE_MSVC_SPEEDUP=1
    FMT_HEADER_ONLY=1
    NTEST
    OVERRIDE_KERNEL_PREFIX="tt_metal/programming_examples/"
    SPDLOG_COMPILED_LIB
    SPDLOG_FMT_EXTERNAL
    TRACY_IMPORTS
    TRACY_TIMER_FALLBACK
    TT_ENABLE_LIGHT_METAL_TRACE=1)

target_compile_options(tilelang_blackhole_runner PRIVATE
    -Wno-int-to-pointer-cast)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(tilelang_blackhole_runner PRIVATE
        -Wno-c++11-narrowing)
endif()

target_include_directories(tilelang_blackhole_runner PRIVATE
    "${TT_METAL_HOME}/tt_stl/."
    "${TT_METAL_HOME}/tt_metal/api"
    "${TT_METAL_BUILD_DIR}/tt_metal/api"
    "${TT_METAL_HOME}/tt_metal/hostdevcommon/api"
    "${TT_METAL_HOME}")

target_include_directories(tilelang_blackhole_runner SYSTEM PRIVATE
    "${TT_METAL_HOME}/.cpmcache/reflect/f93e77475670eaeacf332927dfe8b50e3f3812e0"
    "${TT_METAL_HOME}/.cpmcache/enchantum/2fb7ab238e36c101b9848892ddb6382276b65837/enchantum/include"
    "${TT_METAL_HOME}/.cpmcache/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308/include"
    "${TT_METAL_HOME}/.cpmcache/fmt/69912fb6b71fcb1f7e5deca191a2bb4748c4e7b6/include"
    "${TT_METAL_HOME}/.cpmcache/tt-logger/d2339ce68562cae34cd95f3fece7fd94eb0529b7/include"
    "${TT_METAL_HOME}/.cpmcache/spdlog/b1c2586bb5c35a7929362e87f62433eb68206873/include"
    "${TT_METAL_HOME}/tt_metal/third_party/umd/device/api"
    "${TT_METAL_HOME}/tt_metal/third_party/tracy/public")

target_link_directories(tilelang_blackhole_runner PRIVATE
    "${TT_METAL_BUILD_DIR}/lib"
    "${TT_METAL_BUILD_DIR}/tt_metal"
    "${TT_METAL_BUILD_DIR}/tt_stl"
    "${TT_METAL_BUILD_DIR}/tt_metal/third_party/umd/device"
    "${TT_METAL_BUILD_DIR}/_deps/spdlog-build")

target_link_libraries(tilelang_blackhole_runner PRIVATE
    tt_metal
    tt_stl
    spdlog
    tracy
    device
    dl)

set_target_properties(tilelang_blackhole_runner PROPERTIES
    BUILD_RPATH "${TT_METAL_BUILD_DIR}/lib;${TT_METAL_BUILD_DIR}/tt_metal;${TT_METAL_BUILD_DIR}/tt_stl;${TT_METAL_BUILD_DIR}/tt_metal/third_party/umd/device"
    INSTALL_RPATH "${TT_METAL_BUILD_DIR}/lib;${TT_METAL_BUILD_DIR}/tt_metal;${TT_METAL_BUILD_DIR}/tt_stl;${TT_METAL_BUILD_DIR}/tt_metal/third_party/umd/device")
