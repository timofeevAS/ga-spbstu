import os
import json
import numpy as np
import matplotlib.pyplot as plt

def read_data_from_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as file:
                json_data = json.load(file)
                data.append(json_data)
    return data

def calculate_statistics(data):
    iterations = max(len(item["steps"]) for item in data)
    time_stats = {"mean": [], "min": [], "max": []}
    best_stats = {"mean": [], "min": [], "max": []}

    for i in range(iterations):
        time_values = []
        best_values = []
        for item in data:
            if i < len(item["steps"]):
                time_values.append(item["steps"][i]["time"])
                best_values.append(item["steps"][i]["best"])

        if time_values and best_values:
            time_stats["mean"].append(np.mean(time_values))
            time_stats["min"].append(np.min(time_values))
            time_stats["max"].append(np.max(time_values))

            best_stats["mean"].append(np.mean(best_values))
            best_stats["min"].append(np.min(best_values))
            best_stats["max"].append(np.max(best_values))

    return time_stats, best_stats

def plot_statistics(time_stats, best_stats):
    iterations = range(1, len(time_stats["mean"]) + 1)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(iterations, time_stats["mean"], label='Среднее время', color='b')
    plt.fill_between(iterations, time_stats["min"], time_stats["max"], color='b', alpha=0.2)
    plt.xlabel('Итерация')
    plt.ylabel('Время (сек.)')
    plt.title('Статистика времени по итерациям')
    plt.legend()

    # График для лучшего значения
    plt.subplot(1, 2, 2)
    plt.plot(iterations, best_stats["mean"], label='Среднее значение "best"', color='g')
    plt.plot(iterations, best_stats["min"], label='Худшее значение "best"', color='r', linestyle='--')
    plt.plot(iterations, best_stats["max"], label='Лучшее значение "best"', color='orange', linestyle='--')
    plt.fill_between(iterations, best_stats["min"], best_stats["max"], color='g', alpha=0.2)
    plt.xlabel('Итерация')
    plt.ylabel('Значение "best"')
    plt.title('Статистика лучшего значения по итерациям')
    plt.legend()

    plt.tight_layout()
    plt.show()

directory_path = 'C:\\Users\\atimo\\PycharmProjects\\ga-spbstu\\evolution'

data = read_data_from_json_files(directory_path)
time_stats, best_stats = calculate_statistics(data)

plot_statistics(time_stats, best_stats)
