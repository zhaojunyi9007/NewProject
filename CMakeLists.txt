cmake_minimum_required(VERSION 3.10)
project(HSR_Lidar_Camera_Calibration)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer conda environment libraries when available.
if(DEFINED ENV{CONDA_PREFIX})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    message(STATUS "Conda prefix detected: $ENV{CONDA_PREFIX}")
endif()

# --- Find Packages ---
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

# --- Include Directories ---
include_directories(
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}  # 项目根目录，用于包含 "include/common.h"
)

link_directories(
    ${PCL_LIBRARY_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

# --- Output Directories ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# --- Common Utils ---
# 编译 io_utils.cpp 为一个对象库或者直接添加到目标中
# 这里为了简单，直接添加到每个目标源列表中

# --- Executable: Lidar Extractor (提取 3D 点与线) ---
add_executable(lidar_extractor 
    cpp/lidar_extractor.cpp 
    cpp/io_utils.cpp
)
target_link_libraries(lidar_extractor 
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
)

# --- Executable: Optimizer (Coarse-to-Fine 标定) ---
add_executable(optimizer 
    cpp/optimizer.cpp 
    cpp/io_utils.cpp
    cpp/optimizer_data_loader.cpp
    cpp/optimizer_scoring.cpp
)
target_link_libraries(optimizer 
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    Eigen3::Eigen
)

message(STATUS "=== Build Configuration ===")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "CMake Version: ${CMAKE_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Source Dir: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Binary Dir: ${CMAKE_BINARY_DIR}")
message(STATUS "")
message(STATUS "=== Dependencies ===")
message(STATUS "PCL Version: ${PCL_VERSION}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "Eigen3 Version: ${EIGEN3_VERSION_STRING}")
message(STATUS "Ceres Version: ${Ceres_VERSION}")
message(STATUS "")
message(STATUS "=== Targets ===")
message(STATUS "lidar_extractor: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lidar_extractor")
message(STATUS "optimizer: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/optimizer")
message(STATUS "")
message(STATUS "Build configuration completed.")
