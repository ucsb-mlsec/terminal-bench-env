; ModuleID = 'dynamic_array.ll'
source_filename = "dynamic_array.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @dynamic_array(i32* %arr, i32 %idx) {
entry:
  %idx_ext = sext i32 %idx to i64
  %ptr1 = getelementptr inbounds i32, i32* %arr, i64 %idx_ext
  %val1 = load i32, i32* %ptr1, align 4
  %offset = add i32 %idx, %val1
  %offset_ext = sext i32 %offset to i64
  %ptr2 = getelementptr inbounds i32, i32* %arr, i64 %offset_ext
  %val2 = load i32, i32* %ptr2, align 4
  %sum = add i32 %val1, %val2
  %next_idx = add i32 %idx, 1
  %next_idx_ext = sext i32 %next_idx to i64
  %ptr3 = getelementptr inbounds i32, i32* %arr, i64 %next_idx_ext
  %val3 = load i32, i32* %ptr3, align 4
  %result = mul i32 %sum, %val3
  ret i32 %result
}