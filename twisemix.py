import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from statsmodels.tsa.stattools import acf

# Создаём папку для графиков
os.makedirs("plotsTwiseMix", exist_ok=True)

class TwiseMix:
    def __init__(self, seed=123456789):
        self.state = seed & 0xFFFFFFFFFFFFFFFF  # 64-битное состояние
        self.a = 6364136223846793005
        self.c = 1442695040888963407
        self.m = 2**64

    def next(self):
        self.state = (self.a * self.state + self.c) % self.m
        x = self.state
        x ^= (x >> 21)
        x ^= (x << 35) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 4)
        return (x & 0xFFFFFFFFFFFFFFFF) / self.m

# Создаём генератор и выборку
np.random.seed(42)
seed = np.random.randint(0, 2**31)
gen = TwiseMix(seed=42)
samples = [gen.next() for _ in range(1000)]

# Строим гистограмму и сохраняем
plt.figure(figsize=(10, 5))
plt.hist(samples, bins=20, edgecolor='black')
plt.title("Гистограмма значений TwiseMix")
plt.xlabel("Значения")
plt.ylabel("Частота")
plt.grid(True)
plt.savefig("plotsTwiseMix/twisemix_histogram.png")
plt.close()

# χ²-тест
hist, _ = np.histogram(samples, bins=10)
expected = [len(samples) / 10] * 10
chi2_stat, p_value = chisquare(hist, expected)
print(f"Chi-squared: {chi2_stat:.3f}, p-value: {p_value:.3f}")

# Автокорреляция
acf_vals = acf(samples, nlags=10)

print("Autocorrelation (lag 0–10):")
for lag in range(len(acf_vals)):
    print(f" lag_{lag}: {acf_vals[lag]:.4f}")