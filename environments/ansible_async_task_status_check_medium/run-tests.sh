#!/bin/bash

# Run tests (test dependencies already installed in Dockerfile)
pytest $TEST_DIR/test_outputs.py -rA
