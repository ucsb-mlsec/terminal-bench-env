type number = {
  value : float;
  precision : int;
}

val make : float -> int -> number
(** [make v p] creates a number with value [v] and precision [p].
    Precision must be non-negative. *)

val add : number -> number -> number
(** [add n1 n2] adds two numbers, using the minimum precision of the two. *)

val multiply : number -> number -> number
(** [multiply n1 n2] multiplies two numbers, using the minimum precision of the two. *)

val to_string : number -> string
(** [to_string n] converts a number to its string representation,
    formatted according to its precision. *)