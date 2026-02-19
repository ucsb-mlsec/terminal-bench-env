datatype List<T> = Nil | Cons(head: T, tail: List<T>)

function Length<T>(xs: List<T>): nat
{
  match xs
  case Nil => 0
  case Cons(_, tail) => 1 + Length(tail)
}

function Append<T>(xs: List<T>, ys: List<T>): List<T>
  ensures Length(Append(xs, ys)) == Length(xs) + Length(ys)
  ensures forall i :: 0 <= i < Length(xs) ==> ElementAt(Append(xs, ys), i) == ElementAt(xs, i)
  ensures forall i :: 0 <= i < Length(ys) ==> ElementAt(Append(xs, ys), Length(xs) + i) == ElementAt(ys, i)
{
  match xs
  case Nil => ys
  case Cons(head, tail) => Cons(head, Append(tail, ys))
}

function ElementAt<T>(xs: List<T>, index: nat): T
  requires index < Length(xs)
{
  match xs
  case Cons(head, tail) => if index == 0 then head else ElementAt(tail, index - 1)
}

function Reverse<T>(xs: List<T>): List<T>
  ensures Length(Reverse(xs)) == Length(xs)
  ensures forall i :: 0 <= i < Length(xs) ==> ElementAt(Reverse(xs), i) == ElementAt(xs, Length(xs) - 1 - i)
  ensures Reverse(Reverse(xs)) == xs
{
  match xs
  case Nil => Nil
  case Cons(head, tail) => 
    assert Length(tail) < Length(xs);
    var revTail := Reverse(tail);
    assert Length(revTail) == Length(tail);
    var result := Append(revTail, Cons(head, Nil));
    assert Length(result) == Length(revTail) + 1;
    assert Length(result) == Length(xs);
    result
}

lemma ReversePreservesLength<T>(xs: List<T>)
  ensures Length(Reverse(xs)) == Length(xs)
{
  match xs
  case Nil => 
    assert Length(Reverse(Nil)) == 0;
    assert Length(Nil) == 0;
  case Cons(head, tail) =>
    ReversePreservesLength(tail);
    assert Length(Reverse(tail)) == Length(tail);
    var revTail := Reverse(tail);
    assert Length(Append(revTail, Cons(head, Nil))) == Length(revTail) + 1;
    assert Length(Reverse(xs)) == Length(tail) + 1;
    assert Length(xs) == Length(tail) + 1;
}

lemma ReverseIsInvolutive<T>(xs: List<T>)
  ensures Reverse(Reverse(xs)) == xs
  decreases xs
{
  match xs
  case Nil => 
    assert Reverse(Nil) == Nil;
    assert Reverse(Reverse(Nil)) == Nil;
  case Cons(head, tail) =>
    ReverseIsInvolutive(tail);
    var revTail := Reverse(tail);
    assert Reverse(Reverse(tail)) == tail;
    var revXs := Append(revTail, Cons(head, Nil));
    assert revXs == Reverse(xs);
    ReverseAppendDistributive(revTail, Cons(head, Nil));
    assert Reverse(Append(revTail, Cons(head, Nil))) == Append(Reverse(Cons(head, Nil)), Reverse(revTail));
    assert Reverse(Cons(head, Nil)) == Cons(head, Nil);
    assert Reverse(revTail) == tail;
    assert Append(Cons(head, Nil), tail) == Cons(head, tail);
    assert Reverse(Reverse(xs)) == xs;
}

lemma ReverseAppendDistributive<T>(xs: List<T>, ys: List<T>)
  ensures Reverse(Append(xs, ys)) == Append(Reverse(ys), Reverse(xs))
  decreases xs
{
  match xs
  case Nil =>
    assert Append(Nil, ys) == ys;
    assert Reverse(ys) == Reverse(Append(Nil, ys));
    assert Reverse(Nil) == Nil;
    AppendNilRight(Reverse(ys));
    assert Append(Reverse(ys), Nil) == Reverse(ys);
  case Cons(head, tail) =>
    ReverseAppendDistributive(tail, ys);
    assert Reverse(Append(tail, ys)) == Append(Reverse(ys), Reverse(tail));
    var appendResult := Append(xs, ys);
    assert appendResult == Cons(head, Append(tail, ys));
    var reverseAppend := Reverse(appendResult);
    assert reverseAppend == Append(Reverse(Append(tail, ys)), Cons(head, Nil));
    assert reverseAppend == Append(Append(Reverse(ys), Reverse(tail)), Cons(head, Nil));
    AppendAssociative(Reverse(ys), Reverse(tail), Cons(head, Nil));
    assert Append(Append(Reverse(ys), Reverse(tail)), Cons(head, Nil)) == Append(Reverse(ys), Append(Reverse(tail), Cons(head, Nil)));
    assert Reverse(xs) == Append(Reverse(tail), Cons(head, Nil));
    assert reverseAppend == Append(Reverse(ys), Reverse(xs));
}

lemma AppendNilRight<T>(xs: List<T>)
  ensures Append(xs, Nil) == xs
  decreases xs
{
  match xs
  case Nil => 
    assert Append(Nil, Nil) == Nil;
  case Cons(head, tail) =>
    AppendNilRight(tail);
    assert Append(tail, Nil) == tail;
    assert Append(xs, Nil) == Cons(head, Append(tail, Nil));
    assert Append(xs, Nil) == Cons(head, tail);
    assert Cons(head, tail) == xs;
}

lemma AppendAssociative<T>(xs: List<T>, ys: List<T>, zs: List<T>)
  ensures Append(Append(xs, ys), zs) == Append(xs, Append(ys, zs))
  decreases xs
{
  match xs
  case Nil =>
    assert Append(Nil, ys) == ys;
    assert Append(ys, zs) == Append(Nil, Append(ys, zs));
    assert Append(Append(Nil, ys), zs) == Append(ys, zs);
  case Cons(head, tail) =>
    AppendAssociative(tail, ys, zs);
    assert Append(Append(tail, ys), zs) == Append(tail, Append(ys, zs));
    assert Append(xs, ys) == Cons(head, Append(tail, ys));
    assert Append(Append(xs, ys), zs) == Cons(head, Append(Append(tail, ys), zs));
    assert Append(Append(xs, ys), zs) == Cons(head, Append(tail, Append(ys, zs)));
    assert Append(xs, Append(ys, zs)) == Cons(head, Append(tail, Append(ys, zs)));
}

lemma ReverseLengthLemma<T>(xs: List<T>)
  ensures Length(Reverse(xs)) == Length(xs)
  decreases Length(xs)
{
  match xs
  case Nil => 
  case Cons(head, tail) =>
    ReverseLengthLemma(tail);
    assert Length(Reverse(tail)) == Length(tail);
}

lemma ReverseElementsLemma<T>(xs: List<T>, i: nat)
  requires i < Length(xs)
  ensures ElementAt(Reverse(xs), i) == ElementAt(xs, Length(xs) - 1 - i)
  decreases xs
{
  match xs
  case Cons(head, tail) =>
    if i < Length(tail) {
      ReverseElementsLemma(tail, i);
      assert ElementAt(Reverse(tail), i) == ElementAt(tail, Length(tail) - 1 - i);
    }
}

lemma AppendLengthLemma<T>(xs: List<T>, ys: List<T>)
  ensures Length(Append(xs, ys)) == Length(xs) + Length(ys)
  decreases xs
{
  match xs
  case Nil =>
  case Cons(head, tail) =>
    AppendLengthLemma(tail, ys);
}

lemma ElementAtAppendLeft<T>(xs: List<T>, ys: List<T>, i: nat)
  requires i < Length(xs)
  ensures ElementAt(Append(xs, ys), i) == ElementAt(xs, i)
  decreases xs
{
  match xs
  case Cons(head, tail) =>
    if i == 0 {
      assert ElementAt(Append(xs, ys), 0) == head;
      assert ElementAt(xs, 0) == head;
    } else {
      ElementAtAppendLeft(tail, ys, i - 1);
    }
}

lemma ElementAtAppendRight<T>(xs: List<T>, ys: List<T>, i: nat)
  requires i < Length(ys)
  ensures ElementAt(Append(xs, ys), Length(xs) + i) == ElementAt(ys, i)
  decreases xs
{
  match xs
  case Nil =>
    assert Append(Nil, ys) == ys;
    assert ElementAt(ys, i) == ElementAt(Append(Nil, ys), i);
  case Cons(head, tail) =>
    ElementAtAppendRight(tail, ys, i);
}