type number = {
  value: float;
  precision: int;
}

type result = number

let make ?(precision=2) value =
  { value = float_of_int value; precision }

let make_float ?(precision=2) value =
  { value; precision }

let add n1 n2 =
  let max_precision = max n1.precision n2.precision in
  { value = n1.value +. n2.value; precision = max_precision }

let subtract n1 n2 =
  let max_precision = max n1.precision n2.precision in
  { value = n1.value -. n2.value; precision = max_precision }

let multiply n1 n2 =
  let max_precision = max n1.precision n2.precision in
  { value = n1.value *. n2.value; precision = max_precision }

let divide n1 n2 =
  if n2.value = 0.0 then
    failwith "Division by zero"
  else
    let max_precision = max n1.precision n2.precision in
    { value = n1.value /. n2.value; precision = max_precision }

let to_float n = n.value

let to_int n = int_of_float n.value

let to_string n =
  let format_str = Printf.sprintf "%%.%df" n.precision in
  Printf.sprintf format_str n.value

let compare n1 n2 =
  if n1.value < n2.value then -1
  else if n1.value > n2.value then 1
  else 0

let equal n1 n2 =
  n1.value = n2.value

let zero = { value = 0.0; precision = 2 }

let one = { value = 1.0; precision = 2 }