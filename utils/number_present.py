import math
from typing import Tuple


def auto_precision(a: float, b: float, precision=0):
    length = (b - a + 1) * (10 ** precision)
    res = math.ceil(math.log2(length))
    return res

def binary_arithmetic_number(code: str, a: float, b: float, precision=0) -> float:
    n = auto_precision(a, b, precision)

    digit = int(code, 2)
    result = a + digit * ((b - a) / (2 ** n - 1))
    return result

def from_int_to_binary_string(digit: int):
    return bin(digit)[2:]

def synchronize_binary_strings(code1: str, code2: str) -> (str, str):
    if len(code1) == len(code2):
        return code1, code2

    # Extend both str to similar length
    delta = abs(len(code1) - len(code2))
    zeros = str().zfill(delta)

    if len(code1) > len(code2):
        code2 = zeros + code2
    else:
        code1 = zeros + code1

    return code1, code2


def in_range(point: float, num_range: Tuple[float, float]) -> bool:
    if num_range[1] < num_range[0]:
        raise ValueError(f'Value error: range incorrect, cause {num_range[1]} greater than {num_range[0]}.')

    return num_range[0] <= point <= num_range[1]