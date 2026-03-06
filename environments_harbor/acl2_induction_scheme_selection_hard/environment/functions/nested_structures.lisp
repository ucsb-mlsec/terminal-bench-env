(defun flatten (lst)
  (if (null lst)
      nil
      (if (atom (car lst))
          (cons (car lst) (flatten (cdr lst)))
          (append (flatten (car lst)) (flatten (cdr lst))))))

(defun deep-copy (structure)
  (if (atom structure)
      structure
      (cons (deep-copy (car structure))
            (deep-copy (cdr structure)))))

(defun merge-sorted (list1 list2)
  (if (null list1)
      list2
      (if (null list2)
          list1
          (if (< (car list1) (car list2))
              (cons (car list1) (merge-sorted (cdr list1) list2))
              (cons (car list2) (merge-sorted list1 (cdr list2)))))))

(defun is-atom (x)
  (atom x))