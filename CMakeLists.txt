


cmake_minimum_required(VERSION 3.20)

project(EdgeSightNative LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(
    ORT_ROOT
    "${CMAKE_SOURCE_DIR}/third_party/onnxruntime"
    CACHE PATH "ONNX Runtime C++ release directory"
)

if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime C++ header not found")
endif()

if(WIN32)
    set(ORT_LIB "${ORT_ROOT}/lib/onnxruntime.lib")
    set(ORT_DLL "${ORT_ROOT}/lib/onnxruntime.dll")
else()
    message(FATAL_ERROR
        "Step 15 currently targets Windows x64")
endif()

if(NOT EXISTS "${ORT_LIB}" OR NOT EXISTS "${ORT_DLL}")
    message(FATAL_ERROR
        "ONNX Runtime libraries not found")
endif()

add_executable(
    edgesight_cpp
    cpp/main.cpp
)

target_include_directories(
    edgesight_cpp PRIVATE
    "${ORT_ROOT}/include"
)

target_link_libraries(
    edgesight_cpp PRIVATE
    "${ORT_LIB}"
)

add_custom_command(
    TARGET edgesight_cpp
    POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${ORT_DLL}"
        "$<TARGET_FILE_DIR:edgesight_cpp>"
)
