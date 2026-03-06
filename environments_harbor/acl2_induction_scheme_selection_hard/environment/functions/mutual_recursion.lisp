(defun even-p (n)
  (if (equal n 0)
      t
      (odd-p (- n 1))))

(defun odd-p (n)
  (if (equal n 0)
      nil
      (even-p (- n 1))))

(defun process-a (items)
  (if (null items)
      nil
      (process-b (cdr items))))

(defun process-b (items)
  (if (null items)
      t
      (process-a (cdr items))))