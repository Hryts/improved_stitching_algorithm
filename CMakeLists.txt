cmake_minimum_required(VERSION 3.16)
project(improvedStitching)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(glm REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if (MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()


add_executable(improvedStitching
        main.cpp
        include/Stitching/Stitcher.h
        src/Stitcher.cpp
        src/KDTree.h
        src/LMOptimizer.h
        src/MyLMOptimizer.h)

target_link_libraries( improvedStitching ${OpenCV_LIBS} Eigen3::Eigen)