cmake_minimum_required(VERSION 3.16)
project(facelandmarks LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.20 gstreamer-video-1.0>=1.20)

# Where the export script placed headers/libs
set(MP_EXPORT "${CMAKE_SOURCE_DIR}/third_party/mediapipe-export")

# Build the shared plugin
add_library(gstfacelandmarks SHARED
  src/gstfacelandmarks.cpp
)

# Put exported includes BEFORE system ones to avoid picking system FlatBuffers
target_include_directories(gstfacelandmarks
  PRIVATE
    BEFORE
      ${MP_EXPORT}/include
      /usr/include/eigen3
      ${GST_INCLUDE_DIRS}
)

# Define PACKAGE for GST_PLUGIN_DEFINE
target_compile_definitions(gstfacelandmarks PRIVATE PACKAGE="facelandmarks")

# A bit of hygiene
target_compile_options(gstfacelandmarks PRIVATE
  -Wall -Wextra -Wno-deprecated-declarations
)

# Link everything we exported from Bazel (static or shared) + GStreamer
file(GLOB MP_LIBS "${MP_EXPORT}/lib/*")
target_link_libraries(gstfacelandmarks
  PRIVATE
    ${MP_LIBS}
    ${GST_LIBRARIES}
    Threads::Threads
    dl
)

# Ensure the .so name matches our install step
set_target_properties(gstfacelandmarks PROPERTIES OUTPUT_NAME "gstfacelandmarks")