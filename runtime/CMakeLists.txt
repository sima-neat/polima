# Release builds run on Vulcan (scripts/ci/build_runtime.sh) against the exact
# LLiMa packages recorded in resolved-dependencies.lock.json. LLiMa 0.6 changed
# its C++ ABI, so the binaries record and check the LLiMa they were built for.

cmake_minimum_required(VERSION 3.16)
project(polima_native CXX)

# C++20: SimaLMM's public headers assume it.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Release artifacts use the matching LLiMa package installed in the target
# image. Never retain a Vulcan sysroot or staging-prefix path in the ELF.
set(CMAKE_SKIP_RPATH TRUE)

find_package(SimaLMM 0.6 REQUIRED)
set(POLIMA_REQUIRED_LLIMA_VERSION "" CACHE STRING
    "Fail unless SimaLMM reports exactly this version (release builds)")
if(POLIMA_REQUIRED_LLIMA_VERSION AND
   NOT SimaLMM_VERSION VERSION_EQUAL POLIMA_REQUIRED_LLIMA_VERSION)
  message(FATAL_ERROR "SimaLMM ${SimaLMM_VERSION} found at ${SimaLMM_DIR}; "
                      "this build requires ${POLIMA_REQUIRED_LLIMA_VERSION}")
endif()

set(POLIMA_PLATFORM_VERSION "3.0.0" CACHE STRING "Target Modalix platform version")
set(POLIMA_LLIMA_PACKAGE_VERSION "${SimaLMM_VERSION}" CACHE STRING
    "Exact sima-lmm-core package version expected at runtime")
set(POLIMA_SOURCE_COMMIT "unknown" CACHE STRING "PoLiMa source commit")

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

add_library(polima_core STATIC
  src/compatibility.cpp
  src/plan.cpp
  src/repl.cpp
  src/lineedit.cpp
  src/service.cpp
  src/robot.cpp
)
target_include_directories(polima_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${JSONCPP_INCLUDE_DIR}
)
target_link_libraries(polima_core PUBLIC SimaLMM::sima_lmm_runtime)
target_compile_definitions(polima_core PRIVATE
  POLIMA_EXPECTED_PLATFORM="${POLIMA_PLATFORM_VERSION}"
  POLIMA_EXPECTED_LLIMA="${POLIMA_LLIMA_PACKAGE_VERSION}"
  POLIMA_SOURCE_COMMIT="${POLIMA_SOURCE_COMMIT}"
)
if(nlohmann_json_FOUND)
  target_link_libraries(polima_core PUBLIC nlohmann_json::nlohmann_json)
endif()
# -O3 is required for Runner::run's header-inline BF16 conversion loops in
# plan.cpp; lower optimization adds overhead to every MLA call.
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
