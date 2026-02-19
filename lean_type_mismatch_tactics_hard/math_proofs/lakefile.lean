import Lake
open Lake DSL

package «MathProofs» where
  version := "0.1.0"

lean_lib «BasicArithmetic» where
  globs := #[.submodules `BasicArithmetic]