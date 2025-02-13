def reverse_string(s: str) -> str:
    """Reverses the input string."""
    return s[::-1]


def is_palindrome(s: str) -> bool:
    """Checks if a string is a palindrome."""
    return s == s[::-1]


def count_vowels(s: str) -> int:
    """Counts the number of vowels in a string."""
    return sum(1 for c in s.lower() if c in "aeiou")