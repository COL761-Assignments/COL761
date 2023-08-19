#!/bin/bash

# Compile the main.cpp file
CXX=g++
CXXFLAGS="-std=c++11 -O2 -Wall"
$CXX $CXXFLAGS -o main -I include src/fptree.cpp src/main.cpp

# Exit with the compilation status
if [ $? -eq 0 ]; then
    echo "Compilation successful."
    exit 0
else
    echo "Compilation failed."
    exit 1
fi

