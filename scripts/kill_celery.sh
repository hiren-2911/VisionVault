#!/bin/bash

# Find the PID of the running Supervisor process
CELERY_PID=$(pgrep -f celery)

if [ -z "$CELERY_PID" ]; then
    echo "No running Supervisor process found."
    exit 1
else
    echo "celery process found with PID: $CELERY_PID"
fi

# Kill the Supervisor process
sudo kill -9 $CELERY_PID

if [ $? -eq 0 ]; then
    echo "celery process with PID $CELERY_PID has been killed."
else
    echo "Failed to kill Supervisor process."
    exit 1
fi

