;; ACL2 Function Definitions with Various Recursion Patterns
;; For recursion pattern classification exercise

;; SINGLE_VAR examples - recurse on one variable

;; Computes the length of a list
(defun list-length (lst)
  (if (endp lst)
      0
      (+ 1 (list-length (cdr lst)))))

;; Appends two lists together
(defun append-lists (x y)
  (if (endp x)
      y
      (cons (car x) (append-lists (cdr x) y))))

;; Computes factorial of n
(defun factorial (n)
  (if (zp n)
      1
      (* n (factorial (- n 1)))))

;; MULTI_VAR examples - recurse on multiple variables simultaneously

;; Merges two sorted lists into one sorted list
(defun merge-sorted (x y)
  (if (endp x)
      y
      (if (endp y)
          x
          (if (<= (car x) (car y))
              (cons (car x) (merge-sorted (cdr x) y))
              (cons (car y) (merge-sorted x (cdr y)))))))

;; Zips two lists together into pairs
(defun zip-lists (a b)
  (if (endp a)
      nil
      (if (endp b)
          nil
          (cons (list (car a) (car b))
                (zip-lists (cdr a) (cdr b))))))

;; ACCUMULATOR examples - use accumulator parameter

;; Sums list elements using accumulator
(defun sum-list-acc (lst acc)
  (if (endp lst)
      acc
      (sum-list-acc (cdr lst) (+ (car lst) acc))))

;; Reverses a list using accumulator
(defun reverse-acc (lst acc)
  (if (endp lst)
      acc
      (reverse-acc (cdr lst) (cons (car lst) acc))))

;; MUTUAL recursion examples - functions call each other

;; Checks if n is even by mutual recursion
(defun is-even (n)
  (if (zp n)
      t
      (is-odd (- n 1))))

;; Checks if n is odd by mutual recursion
(defun is-odd (n)
  (if (zp n)
      nil
      (is-even (- n 1))))

;; NESTED recursion examples - recursive calls with recursive arguments

;; Ackermann function with nested recursion
(defun ackermann (m n)
  (if (zp m)
      (+ n 1)
      (if (zp n)
          (ackermann (- m 1) 1)
          (ackermann (- m 1) (ackermann m (- n 1))))))

;; Processes list by skipping every other element with nested recursion
(defun skip-alternate (lst)
  (if (endp lst)
      nil
      (if (endp (cdr lst))
          (list (car lst))
          (cons (car lst) (skip-alternate (cdr (cdr lst)))))))