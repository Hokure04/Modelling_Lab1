from math import sqrt
from prettytable import PrettyTable
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt


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
    plt.figure(figsize=(15, 8))
    plt.plot(data, marker='o', linestyle='-', color='blue')
    plt.title('График значений из файла Excel')
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    plt.grid(True)
    plt.show()

# сделал через chat-gpt считает неправильно, нужно будет переделать
def autocorrelation(data, max_lag=10):

    n = len(data)
    mean = calс_mean(data)
    std_dev = calc_standard_deviation(calc_variance(data, mean))

    autocorr_values = []

    for lag in range(1, max_lag + 1):
        covar_sum = sum((data[i] - mean) * (data[i + lag] - mean) for i in range(n - lag))

        autocorr = covar_sum / ((n - lag) * std_dev ** 2)
        autocorr_values.append(autocorr)

    return autocorr_values


main()
