; ModuleID = 'compute_kernel.ll'
source_filename = "compute_kernel.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @compute_kernel(i32 %a, i32 %b, i32 %c) {
entry:
  %const_val = add i32 0, 42
  %temp1 = mul i32 %a, %const_val
  %temp2 = add i32 %b, %c
  %temp3 = sub i32 %temp1, %temp2
  %result = add i32 %temp3, %const_val
  ret i32 %result
}