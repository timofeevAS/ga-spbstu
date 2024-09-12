import math

def auto_precision(a: float, b: float, precision=0):
    length = (b - a + 1) * (10 ** precision)
    print(range)
    res = math.ceil(math.log2(length))
    return res

def binary_arithmetic_number(code: str, a: float, b: float, precision=0) -> float:
    n = auto_precision(a, b, precision)

    digit = int(code, 2)
    result = a + digit * ((b - a) / (2 ** n - 1))
    return result

def from_int_to_binary_string(digit: int):
    return bin(digit)[2:]