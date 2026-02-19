(defun append-lists (xs ys)
  (if (null xs)
      ys
    (cons (car xs) (append-lists (cdr xs) ys))))

(defun reverse-list (lst)
  (if (null lst)
      nil
    (append-lists (reverse-list (cdr lst)) (cons (car lst) nil))))

(defun list-length (lst)
  (if (null lst)
      0
    (+ 1 (list-length (cdr lst)))))

(defun member-p (x lst)
  (cond ((null lst) nil)
        ((equal x (car lst)) t)
        (t (member-p x (cdr lst)))))