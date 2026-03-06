#!/bin/bash
# CBMC Formal Verification Script for Safety Certification
# This script verifies critical safety properties of the control system

/usr/bin/cbmc /workspace/controller.c \
  --bounds-check \
  --pointer-check \
  --div-by-zero-check \
  --signed-overflow-check \
  --unsigned-overflow-check \
  --nan-check \
  --unwind 10 \
  --unwinding-assertions

exit $?