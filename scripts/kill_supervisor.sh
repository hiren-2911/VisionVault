#!/bin/bash

# Find the PID of the running Supervisor process
SUPERVISOR_PID=$(pgrep -f supervisord)

if [ -z "$SUPERVISOR_PID" ]; then
    echo "No running Supervisor process found."
    exit 1
else
    echo "Supervisor process found with PID: $SUPERVISOR_PID"
fi

# Kill the Supervisor process
sudo kill -9 $SUPERVISOR_PID

if [ $? -eq 0 ]; then
    echo "Supervisor process with PID $SUPERVISOR_PID has been killed."
else
    echo "Failed to kill Supervisor process."
    exit 1
fi

