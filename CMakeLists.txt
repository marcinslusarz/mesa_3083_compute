cmake_minimum_required (VERSION 3.8)
project (mesa3083_compute)

find_package(PkgConfig)

pkg_check_modules(Vulkan REQUIRED vulkan>=1.1.128)

pkg_check_modules(EGL REQUIRED egl)
pkg_check_modules(GBM REQUIRED gbm)
pkg_check_modules(GL REQUIRED gl)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set (CMAKE_CXX_STANDARD 11)

include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${EGL_INCLUDE_DIRS})
include_directories(${GBM_INCLUDE_DIRS})
include_directories(${GL_INCLUDE_DIRS})

add_executable(vulkan_compute src/vulkan.cpp src/lodepng.cpp src/shared.cpp)

set_target_properties(vulkan_compute PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

target_link_libraries(vulkan_compute ${Vulkan_LIBRARIES} ${CMAKE_DL_LIBS})


add_executable(gl_compute src/gl.c src/lodepng.cpp src/shared.cpp)

set_target_properties(gl_compute PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

target_link_libraries(gl_compute ${EGL_LIBRARIES} ${GBM_LIBRARIES} ${GL_LIBRARIES})
