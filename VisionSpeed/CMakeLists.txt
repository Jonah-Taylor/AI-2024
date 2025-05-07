# Set the minimum required version of CMake (use cmake --version)
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(VisionAI)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add source files
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# Create an executable from the source files
add_executable(${PROJECT_NAME} ${SOURCES})

# Find SFML package
set(SFML_DIR "/usr/lib/cmake/SFML")
find_package(SFML 2.6 COMPONENTS system window graphics network audio REQUIRED)

# Link SFML libraries to your executable
target_link_libraries(${PROJECT_NAME} PRIVATE sfml-system sfml-window sfml-graphics sfml-audio sfml-network)

