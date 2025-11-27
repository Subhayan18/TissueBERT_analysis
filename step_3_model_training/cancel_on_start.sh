
#!/bin/bash
# Usage: ./cancel-on-start.sh JOBID1 JOBID2
job1=$1
job2=$2

echo "Monitoring jobs $job1 and $job2..."
while true; do
    state1=$(squeue --job $job1 --noheader --format=%T 2>/dev/null)
    state2=$(squeue --job $job2 --noheader --format=%T 2>/dev/null)

    # If job1 starts running, cancel job2
    if [[ "$state1" == "RUNNING" && "$state2" != "" ]]; then
        echo "Job $job1 started. Cancelling job $job2..."
        scancel $job2
        break
    fi

    # If job2 starts running, cancel job1
    if [[ "$state2" == "RUNNING" && "$state1" != "" ]]; then
        echo "Job $job2 started. Cancelling job $job1..."
        scancel $job1
        break
    fi

    # Exit if both jobs are gone (completed or canceled)
    if [[ -z "$state1" && -z "$state2" ]]; then
        echo "Both jobs are no longer in queue. Exiting."
        break
    fi

    sleep 10  # check every 10 seconds
done
