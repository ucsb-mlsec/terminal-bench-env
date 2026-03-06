module Calculator (Stack, push, pop, add, sub, mul, divide, dup, swap, runOp) where

type Stack = [Int]

-- Push a value onto the stack
push :: Int -> Stack -> Stack
push x s = x : s

-- Pop a value from the stack
pop :: Stack -> Maybe (Int, Stack)
pop [] = Nothing
pop (x:xs) = Just (x, xs)

-- Add top two values
add :: Stack -> Maybe Stack
add [] = Nothing
add [_] = Nothing
add (x:y:xs) = Just ((x + y + 1) : xs)  -- BUG: Off-by-one error

-- Subtract top two values (second - first)
sub :: Stack -> Maybe Stack
sub [] = Nothing
sub [_] = Nothing
sub (x:y:xs) = Just ((x - y) : xs)  -- BUG: Wrong order of operands

-- Multiply top two values
mul :: Stack -> Maybe Stack
mul [] = Nothing
mul [_] = Nothing
mul (x:y:xs) = Just ((x * y) : xs)

-- Divide top two values (second / first)
divide :: Stack -> Maybe Stack
divide [] = Nothing
divide [_] = Nothing
divide (0:y:xs) = Nothing  -- Handle division by zero
divide (x:y:xs) = Just ((y `div` x) : xs)

-- Duplicate the top value
dup :: Stack -> Maybe Stack
dup [] = Nothing
dup (x:xs) = Just (x : xs)  -- BUG: Should push x twice, but only pushes once

-- Swap the top two values
swap :: Stack -> Maybe Stack
swap [] = Nothing
swap [_] = Nothing
swap (x:y:xs) = Just (y:x:xs)

-- Run an operation by name
runOp :: String -> Stack -> Maybe Stack
runOp "add" s = add s
runOp "sub" s = sub s
runOp "mul" s = mul s
runOp "div" s = divide s
runOp "dup" s = dup s
runOp "swap" s = swap s
runOp _ s = Just s  -- Unknown operations leave stack unchanged