module CustomSort (customSort) where

customSort :: [Int] -> [Int]
customSort [] = []
customSort [x] = [x]
customSort (x:xs) = customSort smaller ++ [x] ++ customSort larger
  where
    smaller = [a | a <- xs, a <= x]
    larger = [b | b <- xs, b > x]