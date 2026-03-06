#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>

/* Math utility functions */
int calculate_sum(int a, int b);
int calculate_product(int a, int b);
double calculate_average(int* arr, int size);
int find_maximum(int* arr, int size);
int find_minimum(int* arr, int size);
double calculate_power(double base, int exponent);
int calculate_factorial(int n);

/* String utility functions */
char* format_string(const char* format, int value);
void reverse_string(char* str);
void to_uppercase(char* str);
void to_lowercase(char* str);
int count_chars(const char* str, char c);
char* duplicate_string(const char* str);

/* Data structure utilities */
void initialize_array(int* arr, int size, int value);
void sort_array(int* arr, int size);
int search_array(int* arr, int size, int target);
void copy_array(int* dest, int* src, int size);
void print_array(int* arr, int size);

/* File utilities */
char* read_file(const char* filename);
int write_file(const char* filename, const char* content);
int append_file(const char* filename, const char* content);

/* Miscellaneous utilities */
int generate_random(int min, int max);
int validate_input(int value, int min, int max);
double convert_units(double value, char from_unit, char to_unit);
void print_banner(const char* title);
int is_prime(int n);

#endif /* UTILS_H */