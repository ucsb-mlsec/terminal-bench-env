; ModuleID = 'matrix_multiply'
source_filename = "matrix_multiply.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @matrix_multiply(i32* %A, i32* %B, i32* %C, i32 %N) {
entry:
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 0, i32* %i, align 4
  %i_val = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %i_val, %N
  br i1 %cmp, label %loop_body, label %exit

loop_body:
  %idx = load i32, i32* %i, align 4
  %ptr_A = getelementptr inbounds i32, i32* %A, i32 %idx
  %val_A = load i32, i32* %ptr_A, align 4
  %ptr_B = getelementptr inbounds i32, i32* %B, i32 %idx
  %val_B = load i32, i32* %ptr_B, align 4
  %mul = mul nsw i32 %val_A, %val_B
  %ptr_C = getelementptr inbounds i32, i32* %C, i32 %idx
  store i32 %mul, i32* %ptr_C, align 4
  %next = add nsw i32 %idx, 1
  store i32 %next, i32* %i, align 4
  br label %exit

exit:
  ret void
}