(defun tree-height (tree)
  (if (null tree)
      0
      (+ 1 (max (tree-height (left tree))
                (tree-height (right tree))))))

(defun count-nodes (tree)
  (if (null tree)
      0
      (+ 1 
         (count-nodes (left tree))
         (count-nodes (right tree)))))

(defun tree-sum (tree)
  (if (null tree)
      0
      (+ (value tree)
         (tree-sum (left tree))
         (tree-sum (right tree)))))