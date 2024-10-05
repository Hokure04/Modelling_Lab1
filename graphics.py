import matplotlib.pyplot as plt

def draw_data(data, title):
    plt.figure(figsize=(16, 10))
    plt.plot(data, marker='o', linestyle='-', color='blue')
    plt.title(title)
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    plt.grid(True)
    plt.show()


def draw_histogram(data, bins, title, xlabel="Значения", ylabel="Частота"):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='blue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def draw_all_data(excel_data, erlang_data, title, xlabel="Значения",ylabel="Частота"):
    plt.figure(figsize=(20, 12))
    plt.plot(excel_data, marker='o', linestyle='-', color='blue', linewidth=0.8)
    plt.plot(erlang_data, marker='o', linestyle='-', color='red', linewidth=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def draw_two_histograms(excel_data, erlang_data, bins, title, xlabel="Значения", ylabel="Частота"):
    plt.figure(figsize=(10, 6))
    plt.hist(excel_data, bins=bins, color='blue', edgecolor='black', alpha=0.5, label="Excel Data")
    plt.hist(erlang_data, bins=bins, color='red', edgecolor='black', alpha=0.5, label="Erlang Data")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()