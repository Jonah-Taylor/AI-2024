# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall
LDFLAGS = -lsfml-graphics -lsfml-window -lsfml-system

# Source files
SOURCES = $(wildcard *.cpp) Libraries/Pseument/Pseument.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Output binary
EXEC = VisionAI

# The build rules

all: $(EXEC)

$(EXEC): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(EXEC) $(LDFLAGS)
	rm -f $(OBJECTS)  # Delete .o files after building executable

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXEC)
