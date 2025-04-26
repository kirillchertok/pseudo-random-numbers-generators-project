import secrets as sc
key = sc.token_bytes(32)
otp = "".join(str(sc.randbelow(10)) for _ in range(6))

def my_function():
    pass

import random
for _ in range(1000):
    x = random.randint(1, 100)
    assert my_function(x) >= 0

import numpy as np
data = np.random.normal(loc=100, scale=15, size=10000)


inside = 0
for _ in range(100000):
    x, y = random.random(), random.random()
    if x**2 + y**2 <= 1:
        inside += 1
pi_estimate = 4 * inside / 100000

