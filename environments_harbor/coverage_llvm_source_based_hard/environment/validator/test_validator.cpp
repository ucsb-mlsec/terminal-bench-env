#include "validator.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Running validation tests...\n\n";

    // ========================================
    // validateEmail - EXTENSIVE COVERAGE (90-95%)
    // ========================================
    std::cout << "Testing validateEmail...\n";
    
    // Valid emails
    assert(validateEmail("user@example.com") == true);
    assert(validateEmail("john.doe@company.co.uk") == true);
    assert(validateEmail("test123@test-domain.com") == true);
    assert(validateEmail("a@b.c") == true);
    assert(validateEmail("user+tag@example.com") == true);
    assert(validateEmail("user_name@example.com") == true);
    assert(validateEmail("123@number.com") == true);
    assert(validateEmail("test@subdomain.example.com") == true);
    
    // Invalid formats
    assert(validateEmail("notanemail") == false);
    assert(validateEmail("missing@domain") == false);
    assert(validateEmail("@nodomain.com") == false);
    assert(validateEmail("noat.com") == false);
    assert(validateEmail("double@@example.com") == false);
    assert(validateEmail("spaces in@example.com") == false);
    assert(validateEmail("") == false);
    assert(validateEmail("@") == false);
    assert(validateEmail("user@") == false);
    assert(validateEmail(".user@example.com") == false);
    assert(validateEmail("user..name@example.com") == false);
    
    std::cout << "validateEmail tests passed!\n\n";

    // ========================================
    // validatePhoneNumber - MODERATE COVERAGE (65-75%)
    // ========================================
    std::cout << "Testing validatePhoneNumber...\n";
    
    assert(validatePhoneNumber("1234567890") == true);
    assert(validatePhoneNumber("123-456-7890") == true);
    assert(validatePhoneNumber("(123) 456-7890") == true);
    assert(validatePhoneNumber("+1-123-456-7890") == true);
    assert(validatePhoneNumber("12345") == false);
    assert(validatePhoneNumber("abcd567890") == false);
    assert(validatePhoneNumber("") == false);
    assert(validatePhoneNumber("123-456-789") == false);
    
    std::cout << "validatePhoneNumber tests passed!\n\n";

    // ========================================
    // validateDate - GOOD COVERAGE (80-85%)
    // ========================================
    std::cout << "Testing validateDate...\n";
    
    // Valid dates
    assert(validateDate("2024-01-15") == true);
    assert(validateDate("2023-12-31") == true);
    assert(validateDate("2000-02-29") == true);
    assert(validateDate("1999-06-15") == true);
    assert(validateDate("2024-07-04") == true);
    
    // Invalid formats
    assert(validateDate("2024-13-01") == false);
    assert(validateDate("2024-00-15") == false);
    assert(validateDate("2024-01-32") == false);
    assert(validateDate("2023-02-29") == false);
    assert(validateDate("invalid-date") == false);
    assert(validateDate("") == false);
    assert(validateDate("24-01-15") == false);
    assert(validateDate("2024/01/15") == false);
    
    std::cout << "validateDate tests passed!\n\n";

    // ========================================
    // validateNumericRange - POOR COVERAGE (15-25%)
    // ========================================
    std::cout << "Testing validateNumericRange...\n";
    
    assert(validateNumericRange(50, 0, 100) == true);
    assert(validateNumericRange(150, 0, 100) == false);
    
    std::cout << "validateNumericRange tests passed!\n\n";

    // ========================================
    // validateURL - POOR COVERAGE (20-30%)
    // ========================================
    std::cout << "Testing validateURL...\n";
    
    assert(validateURL("http://www.example.com") == true);
    assert(validateURL("https://example.com") == true);
    assert(validateURL("notaurl") == false);
    
    std::cout << "validateURL tests passed!\n\n";

    // ========================================
    // validateCreditCard - MODERATE COVERAGE (60-70%)
    // ========================================
    std::cout << "Testing validateCreditCard...\n";
    
    assert(validateCreditCard("4532015112830366") == true);
    assert(validateCreditCard("6011514433546201") == true);
    assert(validateCreditCard("378282246310005") == true);
    assert(validateCreditCard("1234567890123456") == false);
    assert(validateCreditCard("1234") == false);
    assert(validateCreditCard("") == false);
    assert(validateCreditCard("abcd1234efgh5678") == false);
    assert(validateCreditCard("4532-0151-1283-0366") == false);
    
    std::cout << "validateCreditCard tests passed!\n\n";

    std::cout << "All tests completed successfully!\n";
    return 0;
}