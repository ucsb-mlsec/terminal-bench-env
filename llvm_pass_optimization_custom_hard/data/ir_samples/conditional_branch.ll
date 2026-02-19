; ModuleID = 'conditional_branch'
source_filename = "conditional_branch.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @conditional_branch(i32 %a, i32 %b) {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %then_block, label %else_block

then_block:
  %mul1 = mul nsw i32 %a, 2
  %add1 = add nsw i32 %mul1, %b
  br label %merge_block

else_block:
  %sub1 = sub nsw i32 %b, %a
  %mul2 = mul nsw i32 %sub1, 3
  br label %merge_block

merge_block:
  %result = phi i32 [ %add1, %then_block ], [ %mul2, %else_block ]
  %final = add nsw i32 %result, %a
  ret i32 %final
}