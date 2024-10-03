from math import sqrt
from prettytable import PrettyTable
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np

def main():
    file_path = 'data_lab1.xlsx'
    data = excel_to_array(file_path)

    sample_sizes = [10, 20, 50, 100, 200, len(data)]

    table = PrettyTable()
    table.field_names = ["Sample Size", "Mean", "Variance", "Standard Deviation", "Variation Coefficient",
                         "Interval 0.9",
                         "Interval 0.95", "Interval 0.99"]
    plot_data(data)
    for size in sample_sizes:
        sample_data = data[:size]
        mean = calс_mean(sample_data)
        variance = calc_variance(sample_data, mean)
        standard_deviation = calc_standard_deviation(variance)

        first_level = 0.9
        second_level = 0.95
        third_level = 0.99

        interval_for_90 = calc_confidence_interval(mean, standard_deviation, sample_data, first_level)
        interval_for_95 = calc_confidence_interval(mean, standard_deviation, sample_data, second_level)
        interval_for_99 = calc_confidence_interval(mean, standard_deviation, sample_data, third_level)

        variation_coefficient = calc_variation_coefficient(standard_deviation, mean)

        table.add_row(
            [size, mean, variance, standard_deviation, variation_coefficient, interval_for_90, interval_for_95,
             interval_for_99])

    print("Результаты анализа выборки:")
    print(table)
    autocorr_values = autocorrelation(data)
    print("Значения автокорреляции:", autocorr_values)
    plot_histogram(data)

    erlang_data = generate_erlang_data(k=2, scale=100, size=300)
    print("Сгененрированные данные по рспрделению Эрланга:", erlang_data)
    plot_histogram(erlang_data)


def excel_to_array(file_path):
    df = pd.read_excel(file_path, header=None)
    data = df.values.flatten().tolist()
    return data


def calс_mean(data):
    return sum(data) / len(data)


def calc_variance(data, mean):
    return sum((value - mean) ** 2 for value in data) / (len(data) - 1)


def calc_standard_deviation(variance):
    return sqrt(variance)


def calc_confidence_interval(mean, deviation, data, confidence_level):
    n = len(data)
    df = n - 1
    t_value = st.t.ppf((1 + confidence_level) / 2, df)
    lower_bound = mean - t_value * deviation / sqrt(n)
    upper_bound = mean + t_value * deviation / sqrt(n)
    interval = (upper_bound - lower_bound) / 2
    return interval


def calc_variation_coefficient(deviation, mean):
    return deviation / mean


def plot_data(data):
    plt.figure(figsize=(16, 10))
    plt.plot(data, marker='o', linestyle='-', color='blue')
    plt.title('График значений из файла Excel')
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    plt.grid(True)
    plt.show()


def autocorrelation(data):
    data_1 = data[1:].copy()
    data_2 = data[:-1].copy()
    autocorr_coefficients = []

    for _ in range(10):
        mean_1 = calс_mean(data_1)
        mean_2 = calс_mean(data_2)
        first = [(value - mean_1) for value in data_1]
        third = [(value - mean_1) ** 2 for value in data_1]
        second = [(value - mean_2) for value in data_2]
        fourth = [(value - mean_2) ** 2 for value in data_2]
        res = [f * s for f, s in zip(first, second)]
        autocorr_coefficients.append(sum(res) / sqrt(sum(third) * sum(fourth)))

        data_1 = data_1[1:]
        data_2 = data_2[:-1]

    return autocorr_coefficients


def plot_histogram(data, bins=10, title="Гистограмма распределения", xlabel="Значения", ylabel="Частота"):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='blue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def generate_erlang_data(k, scale, size):
    # Генерируем данные по Эрланговскому распределению
    data = st.erlang.rvs(a=k, scale=scale, size=size)

    # Нормализация данных в диапазоне [0, 1200], при этом большинство будет в [0, 250]
    min_value = 0
    max_value = 1200
    scaled_data = min_value + (max_value - min_value) * (data - np.min(data)) / (np.max(data) - np.min(data))

    # Ограничение данных: оставляем большинство в диапазоне [0, 250] с пиком около 200
    # Масштабирование для "смещения" данных так, чтобы они в основном находились до 250
    scaled_data = scaled_data[scaled_data <= 1200]

    return scaled_data


main()
