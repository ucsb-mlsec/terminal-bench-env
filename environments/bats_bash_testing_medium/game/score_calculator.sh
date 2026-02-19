#!/bin/bash

# score_calculator.sh - Calculate player score based on game events

score=0

# If no arguments, output 0 and exit
if [ $# -eq 0 ]; then
    echo 0
    exit 0
fi

# Process each event in order
for event in "$@"; do
    # Check if event contains a colon
    if [[ ! "$event" =~ : ]] && [ "$event" != "death" ]; then
        exit 1
    fi
    
    # Split event into type and value
    event_type="${event%%:*}"
    event_value="${event#*:}"
    
    case "$event_type" in
        treasure)
            # Validate that value is a number
            if ! [[ "$event_value" =~ ^[0-9]+$ ]]; then
                exit 1
            fi
            score=$((score + event_value))
            ;;
        enemy)
            # Validate that level is a number
            if ! [[ "$event_value" =~ ^[0-9]+$ ]]; then
                exit 1
            fi
            score=$((score + event_value * 10))
            ;;
        quest)
            if [ "$event_value" != "complete" ]; then
                exit 1
            fi
            score=$((score + 100))
            ;;
        death)
            # Death has no value parameter
            score=$((score / 2))
            ;;
        *)
            # Invalid event type
            exit 1
            ;;
    esac
done

echo "$score"
exit 0