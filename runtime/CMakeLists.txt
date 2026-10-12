# Built on the Modalix DevKit: find_package(SimaLMM) and the aarch64 MLA runtime
# exist only there, at /usr/lib/aarch64-linux-gnu/cmake/SimaLMM/SimaLMMConfig.cmake.
#
# The DevKit has 16 cores, so `cmake --build -j8` is reasonable.

cmake_minimum_required(VERSION 3.16)
project(polima_native CXX)

# C++20: SimaLMM's public headers assume it.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(SimaLMM REQUIRED)

# SimaLMM's mla_model.hpp pulls in configInterface.h, which includes
# <json/json.h> -- jsoncpp, installed under /usr/include/jsoncpp rather than on
# the default include path. Without it the build fails inside a system header
# with no obvious connection to this project.
find_path(JSONCPP_INCLUDE_DIR json/json.h PATHS /usr/include/jsoncpp /usr/local/include/jsoncpp)
if(NOT JSONCPP_INCLUDE_DIR)
  message(FATAL_ERROR "jsoncpp headers not found; SimaLMM needs <json/json.h>")
endif()

# nlohmann/json is a system header on the DevKit (/usr/include/nlohmann/json.hpp),
# so nothing is vendored.
find_package(nlohmann_json QUIET)

add_library(polima_core STATIC src/plan.cpp src/repl.cpp src/lineedit.cpp src/service.cpp src/robot.cpp)
target_include_directories(polima_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${JSONCPP_INCLUDE_DIR}
)
target_link_libraries(polima_core PUBLIC SimaLMM::sima_lmm_runtime)
if(nlohmann_json_FOUND)
  target_link_libraries(polima_core PUBLIC nlohmann_json::nlohmann_json)
endif()
# -O3 is load bearing: Runner::run's bf16 conversion loops are header-inline and
# compiled into plan.cpp, so -O2 de-optimizes every MLA call. Measured on the
# DevKit, ACT end to end: 40.5 ms at -O2, 20.5 ms at -O3.
target_compile_options(polima_core PRIVATE -Wall -Wextra -O3)


add_executable(polima_server src/server_main.cpp)
target_link_libraries(polima_server PRIVATE polima_core)

add_executable(polima_cli src/cli_main.cpp)
target_link_libraries(polima_cli PRIVATE polima_core)

install(TARGETS polima_server polima_cli RUNTIME DESTINATION bin)

# Bit-exactness tests for the boundary-conversion kernels (no MLA needed).
# On the DevKit: build/polima_kernel_tests [--exhaustive]. Not installed.
find_package(Threads REQUIRED)
add_executable(polima_kernel_tests tests/test_kernels.cpp)
target_include_directories(polima_kernel_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(polima_kernel_tests PRIVATE -Wall -Wextra -O3)
target_link_libraries(polima_kernel_tests PRIVATE Threads::Threads)
# The worker pool in kernels.hpp starts threads from the runtime itself.
target_link_libraries(polima_core PUBLIC Threads::Threads)
