import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

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

    def lcg(self, a=1664525, c=1013904223, m=2**32):
        """Линейный конгруэнтный генератор"""
        numbers = []
        x = self.seed
        for _ in range(self.count):
            x = (a * x + c) % m
            numbers.append(x / m)
        stats = self.calculate_stats(numbers)
        self.results['LCG'] = {'data': numbers, 'stats': stats}
        return numbers

    def mersenne_twister(self):
        """Встроенный Mersenne Twister"""
        rng = random.Random(self.seed)
        numbers = [rng.random() for _ in range(self.count)]
        stats = self.calculate_stats(numbers)
        self.results['Mersenne Twister'] = {'data': numbers, 'stats': stats}
        return numbers

    def xorshift(self):
        """XorShift (32-битная версия)"""
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
        """Запуск всех генераторов и анализ"""
        self.lcg()
        self.mersenne_twister()
        self.xorshift()
        
        # Вывод статистики
        print("\n" + "="*50)
        print("СТАТИСТИЧЕСКИЙ АНАЛИЗ".center(50))
        print("="*50)
        for name, result in self.results.items():
            print(f"\n{name}:")
            for stat, value in result['stats'].items():
                stat_name = stat.replace('_', ' ').title()
                print(f"{stat_name:<25}: {value:.6f}")
        
        # Построение графиков
        colors = ['skyblue', 'lightgreen', 'salmon']
        for (name, result), color in zip(self.results.items(), colors):
            self.plot_sequence(result['data'], name, color)
            self.plot_combined_distribution(result['data'], name, color)

        print(f"\nВсе графики сохранены в папку '{self.output_dir}'")

if __name__ == "__main__":
    analyzer = RandomGeneratorAnalyzer(count=100)
    analyzer.run_analysis()
