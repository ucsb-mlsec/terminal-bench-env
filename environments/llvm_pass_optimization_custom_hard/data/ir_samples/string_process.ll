; ModuleID = 'string_process'
source_filename = "string_process.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @string_process(i8* %str, i32 %base) {
entry:
  %offset = add i32 0, 5
  %total_offset = add i32 %offset, %base
  %ptr1 = getelementptr i8, i8* %str, i32 %offset
  %char1 = load i8, i8* %ptr1, align 1
  %char1_ext = zext i8 %char1 to i32
  %length = add i32 0, 10
  %end_offset = add i32 %total_offset, %length
  %ptr2 = getelementptr i8, i8* %str, i32 %end_offset
  %char2 = load i8, i8* %ptr2, align 1
  %char2_ext = zext i8 %char2 to i32
  %result = add i32 %char1_ext, %char2_ext
  %final = add i32 %result, %length
  ret i32 %final
}