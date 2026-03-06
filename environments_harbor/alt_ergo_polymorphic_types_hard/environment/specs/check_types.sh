#!/bin/bash

# Type Checker for Geometry Library Specifications
# Validates type annotations in .ae specification files

SPECS_DIR="/workspace/specs"
ERROR_COUNT=0
FILES_CHECKED=0
FILES_WITH_ERRORS=0

# Type checker function - detects missing type annotations
check_file_types() {
    local filename="$1"
    local file_errors=0
    local line_num=0
    
    echo "Checking $filename..."
    
    while IFS= read -r line; do
        ((line_num++))
        
        # Check for bare numeric literals in expressions (without type suffix like :Real, :Int)
        if echo "$line" | grep -qE '[=:]\s*[0-9]+\s*[+\-*/]' && ! echo "$line" | grep -qE ':[A-Z][a-z]*'; then
            echo "ERROR in $filename: Line $line_num: Type annotation required for numeric expression"
            echo "  Type ambiguity: Cannot infer concrete type for polymorphic operation"
            ((file_errors++))
        fi
        
        # Check for division operations without explicit type context
        if echo "$line" | grep -qE '[0-9]+\s*/\s*[0-9]+' && ! echo "$line" | grep -qE '(Real|Rational|Int)'; then
            echo "ERROR in $filename: Line $line_num: Type annotation required for division operation"
            echo "  Type ambiguity: Division requires explicit numeric type (Real, Rational, or Int)"
            ((file_errors++))
        fi
        
        # Check for polymorphic function calls without type parameters
        if echo "$line" | grep -qE 'scale|dot|norm' && echo "$line" | grep -qE '[0-9]' && ! echo "$line" | grep -qE ':[A-Z]'; then
            echo "ERROR in $filename: Line $line_num: Type annotation required for polymorphic operation"
            echo "  Type ambiguity: Cannot infer concrete type for generic numeric operation"
            ((file_errors++))
        fi
        
        # Check for untyped zero/identity elements
        if echo "$line" | grep -qE '=\s*0\s*$' && ! echo "$line" | grep -qE '(:[A-Z]|//|axiom)'; then
            echo "ERROR in $filename: Line $line_num: Type annotation required for zero literal"
            echo "  Type ambiguity: Zero literal needs explicit type annotation"
            ((file_errors++))
        fi
        
    done < "$filename"
    
    if [ $file_errors -gt 0 ]; then
        ((FILES_WITH_ERRORS++))
        ((ERROR_COUNT += file_errors))
    fi
    
    ((FILES_CHECKED++))
}

# Check all .ae files
for file in "$SPECS_DIR"/*.ae; do
    if [ -f "$file" ]; then
        check_file_types "$file"
        echo ""
    fi
done

# Summary
echo "========================================="
echo "Type Checking Summary:"
echo "Files checked: $FILES_CHECKED"
echo "Files with errors: $FILES_WITH_ERRORS"
echo "Total errors: $ERROR_COUNT"
echo "========================================="

if [ $ERROR_COUNT -gt 0 ]; then
    echo "FAIL: Type checking failed with $ERROR_COUNT error(s)"
    exit 1
else
    echo "PASS: All files type-checked successfully"
    exit 0
fi