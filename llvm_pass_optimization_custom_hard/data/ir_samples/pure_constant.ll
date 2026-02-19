; ModuleID = 'pure_constant.ll'
source_filename = "pure_constant.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @pure_constant(i32 %arg1, i32 %arg2) {
entry:
  %a = add i32 0, 50
  %b = mul i32 0, 10
  %c = add i32 0, 25
  %d = sub i32 0, 5
  %sum1 = add i32 %a, %b
  %sum2 = add i32 %c, %d
  %product = mul i32 %sum1, %sum2
  %result = add i32 %product, %arg1
  %final = add i32 %result, %arg2
  ret i32 %final
}