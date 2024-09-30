from math import sqrt
import scipy.stats as st
import pandas as pd


def main():
    file_path = 'data_lab1.xlsx'
    data = excel_to_array(file_path)

    sample_sizes = [10, 20, 50, 100, 200, len(data)]
    for size in sample_sizes:
        sample_data = data[:size]
        mean = cal_mean(sample_data)
        print(f"Мат ожидание: {mean}")
        variance = calc_variance(sample_data, mean)
        print(f"Дисперсия: {variance}")
        standard_deviation = calc_standard_deviation(variance)
        print(f"Среднеквадратическое отклонение: {standard_deviation}")
        first_level = 0.9
        second_level = 0.95
        third_level = 0.99
        print("Доверительные интервалы для разного показателя надёжности:")
        calc_confidence_interval(mean, standard_deviation, sample_data, first_level)
        calc_confidence_interval(mean, standard_deviation, sample_data, second_level)
        calc_confidence_interval(mean, standard_deviation, sample_data, third_level)
        variation_coefficient = calc_variation_coefficient(standard_deviation, mean)
        print(f"Коэффициент вариации: {variation_coefficient}")



def excel_to_array(file_path):
    df = pd.read_excel(file_path, header=None)
    data = df.values.flatten().tolist()
    return data


def cal_mean(data):
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
    print(interval)


def calc_variation_coefficient(deviation, mean):
    return deviation / mean


main()
