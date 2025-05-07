# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.14) # Required for FetchContent

# Set the project name
project(VisionAI)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Options for using SFML
option(USE_SYSTEM_SFML "Use system installed SFML" OFF)

# Include SFML
if(NOT USE_SYSTEM_SFML)
    # Use FetchContent to download and build SFML
    include(FetchContent)
    
    # Set SFML options before fetching
    set(SFML_BUILD_EXAMPLES OFF CACHE BOOL "Build the SFML examples")
    set(SFML_BUILD_DOC OFF CACHE BOOL "Build the SFML documentation")
    set(SFML_BUILD_TEST_SUITE OFF CACHE BOOL "Build the SFML test suite")
    set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")
    
    # Fetch SFML
    FetchContent_Declare(
        SFML
        GIT_REPOSITORY https://github.com/SFML/SFML.git
        GIT_TAG 2.5.1  # You can specify a tag/version here
    )
    
    # Make SFML available
    FetchContent_MakeAvailable(SFML)
else()
    # Use system SFML
    find_package(SFML 2.5 COMPONENTS system window graphics audio network REQUIRED)
endif()

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# If using FetchContent, add SFML's include directory explicitly
if(NOT USE_SYSTEM_SFML)
    include_directories(${sfml_SOURCE_DIR}/include)
endif()

# Add source files
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# Create an executable from the source files
add_executable(${PROJECT_NAME} ${SOURCES})

# Define preprocessor macros for SFML
target_compile_definitions(${PROJECT_NAME} PRIVATE
    SFML_STATIC=$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>
)

# Link SFML libraries to our executable
target_link_libraries(${PROJECT_NAME} PRIVATE 
    sfml-graphics
    sfml-window
    sfml-system
    sfml-audio
    sfml-network
)

# On Unix-like systems, additional libraries might be needed
if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        X11
        Xrandr
        GL
        pthread
        udev
    )
    
    # Additional OpenAL and audio libraries if using audio
    if(TARGET sfml-audio)
        find_package(OpenAL REQUIRED)
        target_link_libraries(${PROJECT_NAME} PRIVATE
            ${OPENAL_LIBRARY}
            FLAC
            vorbis
            vorbisenc
            vorbisfile
            ogg
        )
    endif()
endif()

# Copy SFML DLLs for Windows builds
if(WIN32 AND BUILD_SHARED_LIBS AND NOT USE_SYSTEM_SFML)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:sfml-system>
            $<TARGET_FILE:sfml-window>
            $<TARGET_FILE:sfml-graphics>
            $<TARGET_FILE:sfml-audio>
            $<TARGET_FILE:sfml-network>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()

# Print out important paths for debugging
message(STATUS "SFML Libraries:")
foreach(COMPONENT IN ITEMS system window graphics audio network)
    get_target_property(LOCATION sfml-${COMPONENT} LOCATION)
    message(STATUS "  sfml-${COMPONENT}: ${LOCATION}")
endforeach()