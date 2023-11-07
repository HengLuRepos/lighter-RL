#!/bin/bash

CPU_CORE=6
# Run your command in the background
taskset -c $CPU_CORE python baseline.py&

# Capture the PID of the most recently started background process
pid=$!

# Measure CPU cycles using perf
sudo perf stat -e cycles -p $pid

# Optionally, you can wait for the process to complete if needed
# wait $pid
