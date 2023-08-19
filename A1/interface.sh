#!/bin/bash

# Check the command-line arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 [C/D] <input_file> <output_file>"
    exit 1
fi

# Run the compiled program
if [ "$1" == "C" ]; then
    ./main compress "$2" "$3"
elif [ "$1" == "D" ]; then
    ./main decompress "$2" "$3"
else
    echo "Invalid argument: Use C for compression or D for decompression."
    exit 1
fi

