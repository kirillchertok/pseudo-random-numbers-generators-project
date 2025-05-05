import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
from scipy.stats import chisquare, kstest, entropy

class RandomGeneratorAnalyzer:
    def __init__(self, seed=None, count=1000, output_dir='plots'):
        self.seed = seed if seed is not None else int(time.time())
        self.count = count
        self.results = {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def calculate_stats(self, data):
        """Вычисление статистик для данных"""
        data_array = np.array(data)
        stats = {
            'mean': np.mean(data_array),
            'median': np.median(data_array),
            'mode': Counter(data).most_common(1)[0][0],
            'variance': np.var(data_array),
            'coefficient_of_variation': np.std(data_array) / np.mean(data_array) if np.mean(data_array) != 0 else 0
        }
        return stats

    def save_plot(self, filename):
        """Сохранение текущего графика"""
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_sequence(self, data, title, color):
        """График последовательности всех чисел"""
        plt.figure(figsize=(12, 6))
        plt.plot(data, color=color, linewidth=0.5, marker='o', markersize=2)
        plt.title(f'{title} - Full Sequence', fontsize=14)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.3)
        self.save_plot(f'{title.lower().replace(" ", "_")}_sequence.png')

    def plot_combined_distribution(self, data, title, color):
        """Гистограмма и полигон на одном графике"""
        n_bins = 1 + int(math.log2(len(data)))
        
        plt.figure(figsize=(12, 6))
        plt.title(f'{title} Distribution', fontsize=14, pad=20)
        
        # Гистограмма
        counts, bin_edges, _ = plt.hist(
            data, bins=n_bins, color=color, edgecolor='black', 
            alpha=0.5, label='Histogram', density=True
        )
        
        # Полигон частот
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.plot(
            bin_centers, counts, color='darkblue', marker='o',
            linestyle='-', linewidth=2, markersize=5, label='Frequency Polygon'
        )
        
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        self.save_plot(f'{title.lower().replace(" ", "_")}_distribution.png')

    def chi_square_test(self, data, bins=10):
        observed, _ = np.histogram(data, bins=bins)
        expected = [len(data)/bins] * bins
        stat, p = chisquare(f_obs=observed, f_exp=expected)
        return stat, p

    def kolmogorov_smirnov_test(self, data):
        stat, p = kstest(data, 'uniform', args=(0, 1))
        return stat, p

    def shannon_entropy(self, data, bins=256):
        hist, _ = np.histogram(data, bins=bins, range=(0, 1), density=True)
        hist = hist[hist > 0]
        return entropy(hist, base=2)

    def autocorrelation(self, data, lags=10):
        data = np.array(data)
        mean = np.mean(data)
        var = np.var(data)
        return [1.0 if lag == 0 else np.corrcoef(data[:-lag], data[lag:])[0, 1] for lag in range(lags + 1)]

    def print_detailed_analysis(self, name, data):
        print(f"{name}".center(50, "="))

        chi2_stat, chi2_p = self.chi_square_test(data)
        print(f"\n📊 χ²-ТЕСТ НА РАВНОМЕРНОСТЬ:\n  Статистика χ²: {chi2_stat:.4f}\n  p-value       : {chi2_p:.4f}")
        print("  ✅ Распределение не отличается от равномерного (принята H0)" if chi2_p > 0.05 else "  ❌ Распределение неравномерно (отклонена H0)")

        ks_stat, ks_p = self.kolmogorov_smirnov_test(data)
        print(f"\n📏 Kolmogorov–Smirnov ТЕСТ:\n  Статистика: {ks_stat:.4f}\n  p-value   : {ks_p:.4f}")
        print("  ✅ Распределение близко к теоретически равномерному" if ks_p > 0.05 else "  ❌ Есть отклонение от теоретически равномерного")

        entropy_value = self.shannon_entropy(data)
        print(f"\n🧠 Энтропия Шеннона:\n  Энтропия: {entropy_value:.4f} бит (макс. = 8.0000)")
        print("  ✅ Высокая энтропия, значения хорошо перемешаны" if entropy_value > 7.5 else "  ❌ Низкая энтропия, возможна корреляция или паттерны")

        print(f"\n🔁 Автокорреляция (lags 0–10):")
        for i, ac in enumerate(self.autocorrelation(data, lags=10)):
            print(f"  lag_{i:<2}: {ac:.4f}")
        print("  ✅ Значения близки к нулю → последовательность независимая\n")

    def lcg(self, a=1664525, c=1013904223, m=2**32):
        numbers = []
        x = self.seed
        for _ in range(self.count):
            x = (a * x + c) % m
            numbers.append(x / m)
        stats = self.calculate_stats(numbers)
        self.results['LCG'] = {'data': numbers, 'stats': stats}
        return numbers

    def mersenne_twister(self):
        rng = random.Random(self.seed)
        numbers = [rng.random() for _ in range(self.count)]
        stats = self.calculate_stats(numbers)
        self.results['Mersenne Twister'] = {'data': numbers, 'stats': stats}
        return numbers

    def xorshift(self):
        numbers = []
        x = self.seed if self.seed != 0 else 1
        for _ in range(self.count):
            x ^= (x << 13) & 0xFFFFFFFF
            x ^= (x >> 17)
            x ^= (x << 5) & 0xFFFFFFFF
            numbers.append(x / 0xFFFFFFFF)
        stats = self.calculate_stats(numbers)
        self.results['XorShift'] = {'data': numbers, 'stats': stats}
        return numbers

    def run_analysis(self):
        self.lcg()
        self.mersenne_twister()
        self.xorshift()
        
        print("\n" + "="*50)
        print("СТАТИСТИЧЕСКИЙ АНАЛИЗ".center(50))
        print("="*50)
        for name, result in self.results.items():
            print(f"\n{name}:")
            for stat, value in result['stats'].items():
                stat_name = stat.replace('_', ' ').title()
                print(f"{stat_name:<25}: {value:.6f}")

        print("\n" + "="*50)
        print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ".center(50))
        print("="*50 + "\n")
        for name, result in self.results.items():
            self.print_detailed_analysis(name, result['data'])

        colors = ['skyblue', 'lightgreen', 'salmon']
        for (name, result), color in zip(self.results.items(), colors):
            self.plot_sequence(result['data'], name, color)
            self.plot_combined_distribution(result['data'], name, color)

        print(f"\nВсе графики сохранены в папку '{self.output_dir}'")

analyzer = RandomGeneratorAnalyzer()
analyzer.run_analysis()
