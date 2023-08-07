def solution(a,b):
    from fractions import gcd
    a, b = int(a), int(b)

    if gcd(a, b) != 1:
        return "impossible"

    k = 0
    while a != b:

        if a == 1:
            k += b - 1
            b = 1

        elif b == 1:
            k += a - 1
            a = 1

        elif a > b:
            k += a // b
            a = a % b

        elif a < b:
            k += b // a
            b = b % a
        

    return str(k)

