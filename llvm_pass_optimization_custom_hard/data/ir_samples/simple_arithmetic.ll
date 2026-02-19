define i32 @simple_arithmetic(i32 %input) {
entry:
  %temp = add i32 0, 25
  %result1 = mul i32 %temp, %input
  %const2 = add i32 10, 15
  %result2 = add i32 %result1, %const2
  ret i32 %result2
}