# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -Wall -Wextra

# Include paths for SFML
INCLUDES = -I/usr/include/SFML

# Library paths for SFML
LDFLAGS = -L/usr/lib

# SFML libraries to link
LIBS = -lsfml-graphics -lsfml-window -lsfml-system

# Source files
SRCS = main.cpp pseument.cpp denselayer.cpp convolayer.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Executable name
EXEC = main

# Default target
all: $(EXEC)

# Link the executable and remove object files
$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^
	rm -f $(OBJS)

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(EXEC)

# Phony targets
.PHONY: all clean