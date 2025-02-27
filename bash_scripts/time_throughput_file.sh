# sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches" # clearing cache requires sudo permissions

# Check if a cluster name was provided
if [ $# -eq 0 ]; then
    echo "Error: No cluster name provided" >&2
    echo "Usage: $0 <cluster>" >&2
    exit 1
fi

cluster="$1"

if [ -n "${TMPDIR+x}" ]; then
    if [[ "${TMPDIR}" == */ ]]; then
        afile=${TMPDIR}${cluster}.npy
    else
        afile=${TMPDIR}/${cluster}.npy
    fi
    
else
    afile=${cluster}.npy
fi
python create_array.py ${afile}
dd if=${afile} of=/dev/null bs=10M > array-throughput-${cluster}.txt 2>&1


