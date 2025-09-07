cmake_minimum_required(VERSION 3.22)
project(mppi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------ 查找 OpenCL ------------------
find_package(OpenCL REQUIRED)
if(NOT OpenCL_FOUND)
    message(FATAL_ERROR "OpenCL not found!")
endif()

# ------------------ 包含头文件 ------------------
include_directories(
    include
    ${OpenCL_INCLUDE_DIRS}
)

# ------------------ 构建 MPPI 库 ------------------
add_library(mppi
    src/mppi/mppi.cc
)

target_link_libraries(mppi
    ${OpenCL_LIBRARIES}
)

# ------------------ 构建测试程序 ------------------
add_executable(test_mppi
    test/test_circle.cpp   # 你的测试程序路径
)

target_link_libraries(test_mppi
    mppi
    ${OpenCL_LIBRARIES}
)
