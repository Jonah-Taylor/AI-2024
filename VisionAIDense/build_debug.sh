#!/bin/bash

# Clean build directory
rm -rf build
mkdir -p build
cd build

# Run CMake with policy override and verbose output
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_VERBOSE_MAKEFILE=ON ..

# Build the project
make VERBOSE=1

# If build succeeds, print library paths
if [ $? -eq 0 ]; then
    echo "Build successful! Checking libraries..."
    ldd ./VisionAI
    
    # Set LD_LIBRARY_PATH to include the SFML libraries
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/_deps/sfml-build/lib
    echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"
    echo "You can now run: ./VisionAI"
else
    echo "Build failed. See errors above."
fi

# Return to the project root
cd ..
