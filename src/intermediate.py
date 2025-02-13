def factorial(n: int) -> int:
    """Computes the factorial of a number recursively."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    return 1 if n == 0 else n * factorial(n - 1)


def gcd(a: int, b: int) -> int:
    """Computes the greatest common divisor (GCD) using the Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return abs(a)


def is_prime(n: int) -> bool:
    """Checks if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
