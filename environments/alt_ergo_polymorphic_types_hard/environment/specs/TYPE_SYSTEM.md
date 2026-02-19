# Type System Reference for AE Specifications

## Introduction

The AE specification language uses a static type system with explicit annotations to resolve polymorphic ambiguities. While the type checker can infer types in many contexts, certain polymorphic operations require explicit type annotations to disambiguate between multiple valid numeric types. This guide explains when and how to add type annotations to ensure specifications pass type-checking.

## Numeric Types

The AE type system supports three primary numeric types:

- **Int**: Integer type for whole numbers (e.g., -2, 0, 42, 1000)
- **Real**: Real number type for floating-point values (e.g., 3.14, -0.5, 2.71828)
- **Rational**: Rational number type for exact fractions (e.g., 1/2, 3/4, 22/7)

Each type has distinct properties and operations. The type checker must know which type is intended when expressions could be valid in multiple numeric domains.

## Type Annotation Syntax

### Annotating Literals

Literals can be annotated with their type using the colon syntax:

```
42:Int          # Integer literal
3.14:Real       # Real number literal
1/3:Rational    # Rational literal
0:Real          # Zero as a real number
1:Int           # One as an integer
```

### Annotating Variables

Variables can be annotated at declaration or assignment:

```
x:Real = 5.0
count:Int = 10
ratio:Rational = 2/3
```

### Annotating Function Parameters

When calling polymorphic functions, specify type parameters explicitly:

```
distance<Real>(p1, p2)
dot_product<Int>(v1, v2)
normalize<Real>(vector)
```

### Annotating Complex Expressions

Parenthesized expressions can be annotated:

```
(x + y):Real
(a * b):Int
(numerator / denominator):Rational
```

## Common Type Ambiguities

Type annotations are required in the following scenarios:

### Numeric Literals in Polymorphic Contexts

When a literal like `0` or `1` appears in a context where multiple types are valid:

```
# Ambiguous - could be Int, Real, or Rational
result = scale(vector, 0)

# Resolved - explicitly Real
result = scale(vector, 0:Real)
```

### Division Operations

Division is particularly ambiguous because it has different semantics across types:

```
# Ambiguous - integer, real, or rational division?
half = 1 / 2

# Resolved options:
half_int = 1:Int / 2:Int           # Integer division (result: 0)
half_real = 1:Real / 2:Real        # Real division (result: 0.5)
half_rational = 1:Rational / 2:Rational  # Rational (result: 1/2)
```

### Mathematical Functions with Generic Parameters

Functions operating on generic numeric types need type specification:

```
# Ambiguous
magnitude = sqrt(x^2 + y^2)

# Resolved
magnitude:Real = sqrt(x:Real^2 + y:Real^2)
```

### Zero and Identity Elements

Zero and identity values in generic operations must specify their type:

```
# Ambiguous - what type of zero?
if distance < 0 then ...

# Resolved
if distance < 0:Real then ...
```

### Conversion Operations

When converting between types, both source and target must be clear:

```
# Ambiguous
rounded = floor(value)

# Resolved
rounded:Int = floor(value:Real)
```

## Examples

### Example 1: Fixing Literal Ambiguity

**Before (fails type-checking):**
```
axiom origin_distance:
  distance(Point(0, 0), p) >= 0
```

**After (passes type-checking):**
```
axiom origin_distance:
  distance(Point(0:Real, 0:Real), p) >= 0:Real
```

### Example 2: Fixing Division Ambiguity

**Before (fails type-checking):**
```
function midpoint(p1, p2):
  Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
```

**After (passes type-checking):**
```
function midpoint(p1, p2):
  Point((p1.x + p2.x) / 2:Real, (p1.y + p2.y) / 2:Real)
```

### Example 3: Fixing Generic Function Call

**Before (fails type-checking):**
```
property unit_vector_length:
  forall v: magnitude(normalize(v)) = 1
```

**After (passes type-checking):**
```
property unit_vector_length:
  forall v: magnitude(normalize<Real>(v)) = 1:Real
```

## Tips and Best Practices

1. **Start with error messages**: Run the type checker and read error messages carefully - they usually indicate the exact location needing annotations.

2. **Be consistent**: If you annotate one part of an expression, annotate related parts with compatible types.

3. **Prefer explicit over implicit**: When in doubt, add more annotations rather than fewer.

4. **Match mathematical intent**: Choose the type that best represents your mathematical meaning (e.g., use Real for continuous geometry, Int for discrete counting).

5. **Check boundary values**: Zero, one, and other special values often need explicit annotation in polymorphic contexts.

6. **Annotate function results**: When a function could return multiple types, annotate the result type explicitly.

7. **Use parentheses**: When annotating complex expressions, use parentheses to clarify what the annotation applies to.

8. **Test incrementally**: Fix one file at a time and verify it passes before moving to the next.