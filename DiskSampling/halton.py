import numpy as np
import matplotlib.pyplot as plt

def next_prime():
    def is_prime(num):
        "Checks if num is a prime value"
        for i in range(2,int(num**0.5)+1):
            if(num % i)==0: return False
        return True

    prime = 3
    while(1):
        if is_prime(prime):
            yield prime
        prime += 2

def vdc(n, base=2):
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder/float(denom)
    return vdc

def halton_sequence(size, dim):
    seq = []
    primeGen = next_prime()
    next(primeGen)
    for d in range(dim):
        base = next(primeGen)
        seq.append([vdc(i, base) for i in range(size)])
    return seq

def halton_sequence2(size, primes):
    seq = []
    for base in primes:
        seq.append([vdc(i, base) for i in range(size)])
    return seq

def halton_rejection_disk(size):
    seq = halton_sequence(100, 2)
    seq = np.transpose(seq)
    resultList = np.empty((0, 2), float)
    counter = 0
    for pt in seq:
        if (pt[0] - 0.5)**2 + (pt[1] - 0.5)**2 < 0.5**2:
            resultList = np.append(resultList, np.array([pt]), axis=0)
            counter += 1
        if counter == size:
            break
    return resultList