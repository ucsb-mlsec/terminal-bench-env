from libc.math cimport sqrt, pow as c_pow

def factorial(int n):
    """
    Compute the factorial of a non-negative integer n.
    
    Args:
        n: A non-negative integer
        
    Returns:
        The factorial of n
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    cdef int i
    cdef long long result = 1
    
    for i in range(1, n + 1):
        result *= i
    
    return result


def power(double base, double exponent):
    """
    Compute base raised to the power of exponent.
    
    Args:
        base: The base number
        exponent: The exponent
        
    Returns:
        base ** exponent
    """
    return c_pow(base, exponent)


def square_root(double x):
    """
    Compute the square root of a non-negative number.
    
    Args:
        x: A non-negative number
        
    Returns:
        The square root of x
    """
    if x < 0:
        raise ValueError("Cannot compute square root of negative number")
    
    return sqrt(x)


def compute_hypotenuse(double a, double b):
    """
    Compute the hypotenuse of a right triangle given two sides.
    
    Args:
        a: Length of first side
        b: Length of second side
        
    Returns:
        The length of the hypotenuse
    """
    return sqrt(a * a + b * b)