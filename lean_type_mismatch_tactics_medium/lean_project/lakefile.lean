import Lake
open Lake DSL

package leanProof where
  version := v!"0.1.0"

@[default_target]
lean_lib LeanProof where
  globs := #[.andSubmodules `LeanProof]