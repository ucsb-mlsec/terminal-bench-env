define i32 @loop_invariant(i32 %n, i32 %start) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.header, label %exit

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %sum = phi i32 [ %start, %entry ], [ %sum.next, %loop.body ]
  %loop.cond = icmp slt i32 %i, %n
  br i1 %loop.cond, label %loop.body, label %exit

loop.body:
  %temp = add i32 %sum, %i
  %sum.next = mul i32 %temp, 2
  %i.next = add i32 %i, 1
  br label %loop.header

exit:
  %result = phi i32 [ %start, %entry ], [ %sum, %loop.header ]
  ret i32 %result
}