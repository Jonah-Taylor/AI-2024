
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(PongAI)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/SFML)

# Add source files
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# Create an executable from the source files
add_executable(${PROJECT_NAME} ${SOURCES})

# SFML libraries to link (Adjust these if necessary based on your local libraries)
set(SFML_LIB_PATH "/path/to/your/libs")  # Set this to the directory where your SFML .so or .a files are located.

# Link SFML libraries to your executable
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${SFML_LIB_PATH}/libsfml-graphics.so
    ${SFML_LIB_PATH}/libsfml-window.so
    ${SFML_LIB_PATH}/libsfml-system.so)

