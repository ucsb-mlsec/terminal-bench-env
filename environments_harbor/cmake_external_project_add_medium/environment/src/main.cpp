#include <iostream>
#include <string>
#include "stringutils.h"

int main() {
    std::string text1 = "hello world";
    std::string text2 = "  TRIM ME  ";
    std::string text3 = "reverse";

    std::cout << "Original: " << text1 << std::endl;
    std::string upper = toUpper(text1);
    std::cout << "Uppercase: " << upper << std::endl;

    std::string lower = toLower(text2);
    std::cout << "Lowercase: " << lower << std::endl;

    std::string trimmed = trim(text2);
    std::cout << "Trimmed: '" << trimmed << "'" << std::endl;

    std::string reversed = reverse(text3);
    std::cout << "Reversed: " << reversed << std::endl;

    std::cout << "String utilities test completed successfully!" << std::endl;

    return 0;
}