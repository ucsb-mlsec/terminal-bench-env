(* Test suite for the mathematical computation library *)
(* This validates the API wrapper functionality *)

let test_compute_sum () =
  let result1 = Api.compute_sum 5.0 10.0 in
  let result2 = Api.compute_sum 3.0 7.0 in
  print_endline ("Sum test 1: " ^ result1);
  print_endline ("Sum test 2: " ^ result2);
  if result1 = "15" && result2 = "10" then
    print_endline "✓ compute_sum tests passed"
  else
    failwith "compute_sum test failed"

let test_compute_product () =
  let result1 = Api.compute_product 5.0 10.0 in
  let result2 = Api.compute_product 3.0 7.0 in
  print_endline ("Product test 1: " ^ result1);
  print_endline ("Product test 2: " ^ result2);
  if result1 = "50" && result2 = "21" then
    print_endline "✓ compute_product tests passed"
  else
    failwith "compute_product test failed"

let test_edge_cases () =
  let result1 = Api.compute_sum 0.0 0.0 in
  let result2 = Api.compute_product 1.0 1.0 in
  print_endline ("Edge case sum: " ^ result1);
  print_endline ("Edge case product: " ^ result2);
  if result1 = "0" && result2 = "1" then
    print_endline "✓ Edge case tests passed"
  else
    failwith "Edge case test failed"

let () =
  print_endline "Running mathematical computation library tests...";
  print_endline "";
  test_compute_sum ();
  test_compute_product ();
  test_edge_cases ();
  print_endline "";
  print_endline "All tests passed!";
  exit 0