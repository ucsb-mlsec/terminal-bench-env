(* High-level API wrapper for the Numbers module *)

open Numbers

(* These wrapper functions are meant to provide a simpler interface
   but currently have type mismatches with the core module *)

let compute_sum x y =
  (* ERROR: Trying to pass floats directly to Numbers.add
     which expects Numbers.number type *)
  let result = Numbers.add x y in
  result

let compute_product x y =
  (* ERROR: Trying to pass floats directly to Numbers.multiply
     which expects Numbers.number type *)
  let result = Numbers.multiply x y in
  result

let compute_difference x y =
  (* ERROR: Type mismatch - passing primitives instead of Numbers.number *)
  let result = Numbers.subtract x y in
  result

let compute_division x y =
  (* ERROR: Type mismatch with division function *)
  if y = 0.0 then
    failwith "Division by zero"
  else
    let result = Numbers.divide x y in
    result

let run_computation a b c =
  (* Perform: (a + b) * c - (a / b)
     ERROR: All these operations expect Numbers.number but receive floats *)
  let sum = compute_sum a b in
  let prod = compute_product sum c in
  let quot = compute_division a b in
  let final = compute_difference prod quot in
  (* ERROR: Trying to convert Numbers.number to string incorrectly *)
  Printf.sprintf "Result: %f" final

let validate_positive x =
  (* ERROR: Comparison expects Numbers.number type *)
  Numbers.is_positive x