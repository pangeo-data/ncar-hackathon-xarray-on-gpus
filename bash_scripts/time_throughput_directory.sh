#!/bin/bash

# Check if a directory was provided
if [ $# -eq 0 ]; then
    echo "Error: No directory provided" >&2
    echo "Usage: $0 <directory>" >&2
    exit 1
fi

# Use the provided directory
dir="$1"

# Check if the directory exists
if [ ! -d "$dir" ]; then
    echo "Error: Directory '$dir' does not exist" >&2
    exit 2
fi

find ${dir} -type f -exec dd if={} of=/dev/null \; > timings.txt 2>&1
