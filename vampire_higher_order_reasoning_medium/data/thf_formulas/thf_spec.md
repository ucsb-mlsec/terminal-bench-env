# THF (Typed Higher-order Form) Syntax Specification

## 1. Basic Structure

THF formulas follow the TPTP standard for typed higher-order logic. Each formula is declared using the following structure:

```
thf(name, role, formula).
```

Where:
- `name` is a unique identifier for the formula (alphanumeric with underscores)
- `role` specifies the formula's purpose: `axiom`, `conjecture`, `definition`, `lemma`, `hypothesis`, `theorem`, etc.
- `formula` is the actual logical expression
- Each declaration must end with a period (`.`)

## 2. Type Annotations

Type annotations use the colon (`:`) operator to associate variables and terms with their types.

**Built-in types:**
- `$i` - individual type (for terms/objects)
- `$o` - boolean type (for propositions)
- `$int` - integer type
- `$real` - real number type

**Function types:**
Function types use the arrow `>` operator:
- `$i > $o` - function from individuals to booleans (unary predicate)
- `$i > $i > $o` - binary predicate
- `$i > $i` - function from individuals to individuals

**Type declaration examples:**
- `X:$i` - X is an individual
- `p:$i>$o` - p is a unary predicate
- `f:$i>$i>$i` - f is a binary function on individuals

## 3. Lambda Abstractions

Lambda abstractions create anonymous functions using the following syntax:

```
^[Variables]: Body
```

**Rules:**
- Use `^` to introduce lambda abstraction
- Variables must be enclosed in square brackets `[...]`
- Each variable must include a type annotation
- Multiple variables are separated by commas
- The body follows after the colon

**Examples:**
- `^[X:$i]: (p @ X)` - lambda taking individual X, applying predicate p
- `^[X:$i, Y:$i]: (r @ X @ Y)` - lambda taking two individuals
- `^[F:$i>$o, X:$i]: (F @ X)` - lambda with higher-order parameter

## 4. Function Application

The `@` operator is used for function application. Application is left-associative.

**Syntax:**
- `f @ X` - apply function f to argument X
- `(p @ X) @ Y` - apply binary predicate p to X and Y (equivalent to curried application)
- `p @ X @ Y` - same as above due to left-associativity

**Rules:**
- Use parentheses to control application order when needed
- Higher-order applications: `(F @ g) @ X` applies result of (F @ g) to X
- Partial application is supported due to currying

## 5. Logical Connectives

THF supports standard logical operators:

- `&` - conjunction (and)
- `|` - disjunction (or)
- `=>` - implication
- `<=>` - bi-implication (equivalence)
- `~` - negation (not)
- `!=` - inequality
- `=` - equality

**Precedence (highest to lowest):**
1. `~` (negation)
2. `&` (conjunction)
3. `|` (disjunction)
4. `=>` (implication)
5. `<=>` (bi-implication)

## 6. Quantifiers

Quantifiers bind variables with type annotations.

**Universal Quantification:**
```
![Variables]: Formula
```

**Existential Quantification:**
```
?[Variables]: Formula
```

**Rules:**
- Variables must include type annotations
- Multiple variables can be quantified together: `![X:$i, Y:$i]: ...`
- Quantified variables are bound within the scope of the formula
- Nested quantification is allowed

**Examples:**
- `![X:$i]: (p @ X)` - for all individuals X, p(X) holds
- `?[Y:$i]: (q @ Y)` - there exists an individual Y such that q(Y)
- `![X:$i]: ?[Y:$i]: (r @ X @ Y)` - nested quantifiers

## 7. Parentheses and Brackets

**Balanced expressions:**
- All opening parentheses `(` must have matching closing `)
- All opening square brackets `[` must have matching closing `]`
- Use parentheses to group subexpressions and control precedence

**Best practices:**
- Use parentheses for clarity even when not strictly required
- Quantifier bodies should be parenthesized if they contain connectives
- Lambda bodies with complex expressions should use parentheses

## 8. Common Errors

**Missing type annotations:**
- ❌ `![X]: (p @ X)` - missing type for X
- ✓ `![X:$i]: (p @ X)` - correct

**Invalid lambda syntax:**
- ❌ `^X: (p @ X)` - missing brackets around variable
- ❌ `^[X]: (p @ X)` - missing type annotation
- ✓ `^[X:$i]: (p @ X)` - correct

**Unmatched parentheses:**
- ❌ `((p @ X) & (q @ Y)` - missing closing parenthesis
- ✓ `((p @ X) & (q @ Y))` - correct

**Undefined or unbound variables:**
- ❌ `(p @ X)` - X used without quantification or binding
- ✓ `![X:$i]: (p @ X)` - X properly quantified

**Incorrect function application:**
- ❌ `p X Y` - missing @ operators
- ✓ `p @ X @ Y` - correct

**Missing commas in declarations:**
- ❌ `^[X:$i Y:$i]: (r @ X @ Y)` - missing comma
- ✓ `^[X:$i, Y:$i]: (r @ X @ Y)` - correct

## 9. Complete Example Formulas

**Example 1: Type declaration**
```
thf(person_type, type, person: $i).
```

**Example 2: Simple axiom with quantifier**
```
thf(all_humans_mortal, axiom, ![X:$i]: ((human @ X) => (mortal @ X))).
```

**Example 3: Lambda abstraction with higher-order function**
```
thf(map_definition, definition, 
    map = (^[F:$i>$i, X:$i]: (F @ X))).
```

**Example 4: Existential quantification with conjunction**
```
thf(exists_wise_person, conjecture, 
    ?[X:$i]: ((person @ X) & (wise @ X))).
```

**Example 5: Complex formula with nested quantifiers**
```
thf(transitivity, axiom, 
    ![R:$i>$i>$o]: (![X:$i, Y:$i, Z:$i]: (((R @ X @ Y) & (R @ Y @ Z)) => (R @ X @ Z)))).
```

**Example 6: Higher-order predicate application**
```
thf(predicate_composition, definition,
    compose = (^[P:$i>$o, Q:$i>$o, X:$i]: ((P @ X) & (Q @ X)))).