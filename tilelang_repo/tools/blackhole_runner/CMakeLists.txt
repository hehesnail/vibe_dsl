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

function(find_single_dir out_var glob_pattern)
    file(GLOB matches LIST_DIRECTORIES true "${glob_pattern}")
    set(dir_matches "")
    foreach(match IN LISTS matches)
        if(IS_DIRECTORY "${match}")
            list(APPEND dir_matches "${match}")
        endif()
    endforeach()
    list(LENGTH dir_matches match_count)
    if(match_count EQUAL 0)
        message(FATAL_ERROR "No match for ${glob_pattern}")
    endif()
    list(GET dir_matches 0 first_match)
    set(${out_var} "${first_match}" PARENT_SCOPE)
endfunction()

find_single_dir(REFLECT_DIR "${TT_METAL_HOME}/.cpmcache/reflect/*")
find_single_dir(ENCHANTUM_DIR "${TT_METAL_HOME}/.cpmcache/enchantum/*")
find_single_dir(FMT_DIR "${TT_METAL_HOME}/.cpmcache/fmt/*")
find_single_dir(TT_LOGGER_DIR "${TT_METAL_HOME}/.cpmcache/tt-logger/*")
find_single_dir(SPDLOG_DIR "${TT_METAL_HOME}/.cpmcache/spdlog/*")
find_single_dir(NLOHMANN_JSON_DIR "${TT_METAL_HOME}/.cpmcache/nlohmann_json/*")

find_package(nlohmann_json REQUIRED)

target_compile_definitions(tilelang_blackhole_runner PRIVATE
    OVERRIDE_KERNEL_PREFIX="tt_metal/programming_examples/"
    ENCHANTUM_ENABLE_MSVC_SPEEDUP=1
    FMT_HEADER_ONLY=1
    NTEST
    SPDLOG_COMPILED_LIB
    SPDLOG_FMT_EXTERNAL
    TRACY_IMPORTS
    TRACY_TIMER_FALLBACK
    TT_ENABLE_LIGHT_METAL_TRACE=1
)

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
    "${REFLECT_DIR}"
    "${ENCHANTUM_DIR}/enchantum/include"
    "${NLOHMANN_JSON_DIR}/include"
    "${FMT_DIR}/include"
    "${TT_LOGGER_DIR}/include"
    "${SPDLOG_DIR}/include"
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
