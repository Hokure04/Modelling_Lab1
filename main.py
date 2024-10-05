from math import sqrt
from prettytable import PrettyTable
import pandas as pd
import scipy.stats as st
import numpy as np
from graphics import draw_data, draw_histogram, draw_all_data, draw_two_histograms

def main():
    file_path = 'data_lab1.xlsx'
    excel_data = excel_to_array(file_path)
    autocorrelation_table = PrettyTable()
    autocorrelation_table.field_names = ["Сдвиг ЧП","1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    draw_data(excel_data, 'График значений из excel')
    calc_sequence_characteristics(excel_data)
    autocorr_values = autocorrelation(excel_data)
    autocorrelation_table.add_row(['К-т АК для задан. ЧП'] + autocorr_values)
    print("Значения автокорреляции:", autocorr_values)
    draw_histogram(excel_data, 15, 'Гистограмма распределения данных из excel')

    erlang_data = generate_erlang_data(k=2, scale=1, size=300)
    print("Сгененрированные данные по рспрделению Эрланга:", erlang_data)
    draw_data(erlang_data, 'График сгенерированных значений')
    calc_sequence_characteristics(erlang_data)
    draw_histogram(erlang_data, 15, 'Гистограмма распределения сгенерированных данных')
    autocorr_erlang = autocorrelation(erlang_data)
    autocorrelation_table.add_row(['К-т АК для сгенер. ЧП'] + autocorr_erlang)
    print("Коэффициенты автокореляции:")
    print(autocorrelation_table)
    draw_all_data(excel_data, erlang_data, "График значений обеих выборок")
    draw_two_histograms(excel_data, erlang_data, 15, 'Гистограммы распределения из excel и сгенерированная')



def excel_to_array(file_path):
    df = pd.read_excel(file_path, header=None)
    data = df.values.flatten().tolist()
    return data


def calc_mean(data):
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

def autocorrelation(data):
    data_1 = data[1:].copy()
    data_2 = data[:-1].copy()
    autocorr_coefficients = []

    for _ in range(10):
        mean_1 = calc_mean(data_1)
        mean_2 = calc_mean(data_2)
        first = [(value - mean_1) for value in data_1]
        third = [(value - mean_1) ** 2 for value in data_1]
        second = [(value - mean_2) for value in data_2]
        fourth = [(value - mean_2) ** 2 for value in data_2]
        res = [f * s for f, s in zip(first, second)]
        autocorr_coefficients.append(sum(res) / sqrt(sum(third) * sum(fourth)))

        data_1 = data_1[1:]
        data_2 = data_2[:-1]

    return autocorr_coefficients


def generate_erlang_data(k, scale, size):
    data = st.erlang.rvs(a=k, scale=scale, size=size)

    min_value = 20
    max_value = 1200
    scaled_data = min_value + (max_value - min_value) * (data - np.min(data)) / (np.max(data) - np.min(data))

    scaled_data = scaled_data[scaled_data <= max_value]

    return scaled_data

def calc_sequence_characteristics(data):
    sample_sizes = [10, 20, 50, 100, 200, len(data)]
    table = PrettyTable()
    table.field_names = ["Размер выборки", "Мат. ожидание", "Дисперсия", "С.к.о", "Коэффициент вариации",
                         "Доверит. интервал 0.9",
                         "Доверит. интервал 0.95", "Доверит. интервал 0.99"]
    for size in sample_sizes:
        sample_data = data[:size]
        mean = calc_mean(sample_data)
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

main()
