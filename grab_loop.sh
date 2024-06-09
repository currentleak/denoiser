#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <channel>"
    exit 1
fi

count=1

iterations=100

channel=$1

while [ $count -le $iterations ]
do
    echo channel = "$channel", itt = ${count}

    timestamp=$(date +"%Y%m%d_%H%M%S")
    grabbeamd "$channel" > "./beam-data/donnee_${timestamp}_${count}.txt"

    #grabbeamd "$channel" > ./beam-data/"donnee${count}.txt"

    ((count++))
done
