#!/bin/bash

# Check if a filename was provided
if [ $# -eq 0 ]; then
    echo "Error: No filename provided" >&2
    echo "Usage: $0 <filename>" >&2
    exit 1
fi

# Use the provided filename
file="$1"

# Check if the file exists
if [ ! -f "$file" ]; then
    echo "Error: File '$file' does not exist" >&2
    exit 2
fi

# Use awk to sum up the bytes and time values
awk '
/bytes transferred in/ {
    # Extract bytes and time values
    bytes += $1;
    
    # Extract time - find the position of "in" and "secs"
    in_pos = index($0, " in ");
    secs_pos = index($0, " secs");
    
    # Extract the time value between "in" and "secs"
    time_str = substr($0, in_pos + 4, secs_pos - in_pos - 4);
    
    # Convert to numeric and add to total
    time += time_str;
}

END {
    printf "Total bytes transferred: %d bytes\n", bytes;
    printf "Total time taken: %.6f seconds\n", time;
    
    # Calculate and report throughput in GB/s
    if (time > 0) {
        # 1 GB = 1,073,741,824 bytes (2^30)
        throughput_bytes = bytes / time;
        throughput_gb = throughput_bytes / 1073741824;
        printf "Average throughput: %.6f GB/s\n", throughput_gb;
    } else {
        printf "Average throughput: N/A (zero time measured)\n";
    }
}
' "$file"
