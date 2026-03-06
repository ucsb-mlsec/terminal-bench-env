(set-logic QF_S)

(declare-const password String)

; Password must be between 8 and 16 characters long
(assert (>= (str.len password) 8))
(assert (<= (str.len password) 16))

; Password must contain at least one uppercase letter (A-Z)
(assert (str.in_re password 
  (re.++ re.all 
    (re.range "A" "Z") 
    re.all)))

; Password must contain at least one lowercase letter (a-z)
(assert (str.in_re password 
  (re.++ re.all 
    (re.range "a" "z") 
    re.all)))

; Password must contain at least one digit (0-9)
(assert (str.in_re password 
  (re.++ re.all 
    (re.range "0" "9") 
    re.all)))

; Password must contain at least one special character (@, #, $, !, %)
(assert (str.in_re password 
  (re.++ re.all 
    (re.union 
      (str.to_re "@")
      (str.to_re "#")
      (str.to_re "$")
      (str.to_re "!")
      (str.to_re "%"))
    re.all)))

; Password must only contain valid characters (alphanumeric and allowed special chars)
(assert (str.in_re password 
  (re.+ (re.union 
    (re.range "A" "Z")
    (re.range "a" "z")
    (re.range "0" "9")
    (str.to_re "@")
    (str.to_re "#")
    (str.to_re "$")
    (str.to_re "!")
    (str.to_re "%")))))

; Password must not start with a digit
(assert (not (str.in_re password 
  (re.++ (re.range "0" "9") re.all))))

(check-sat)
(get-model)