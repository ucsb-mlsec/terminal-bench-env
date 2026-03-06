; ModuleID = 'basic_constant.ll'
source_filename = "basic_constant.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @basic_constant(i32 %arg) {
entry:
  %const_var = add i32 0, 100
  %result = add i32 %const_var, %arg
  ret i32 %result
}