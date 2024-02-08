import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor

start_time = time.perf_counter()


# Задана функция
def f(x):
    return float(3) ** x + np.sin(x)


# Відрізок та крок
start = 0
end = 1.5
h = 0.01

# Генерація значень x на відрізку з кроком h
x_values = np.arange(start, end + h, h)

# Генерація вузлових значень функції
x_nodes = np.arange(start, end + h, h)
y_nodes = f(x_nodes)


# Функція для обчислення кубічного полінома Лагранжа для даного вузлового відрізка
def cubic_lagrange_interpolation(x, x_nodes, y_nodes):
    n = len(x_nodes)
    for i in range(n - 1):
        if i + 2 < n and x_nodes[i] <= x <= x_nodes[i + 1]:
            if i == 0:  # Если x близко к началу диапазона
                x0, x1, x2 = x_nodes[i], x_nodes[i + 1], x_nodes[i + 2]
                y0, y1, y2 = y_nodes[i], y_nodes[i + 1], y_nodes[i + 2]
            elif i == n - 3:  # Если x близко к концу диапазона
                x0, x1, x2 = x_nodes[i - 1], x_nodes[i], x_nodes[i + 1]
                y0, y1, y2 = y_nodes[i - 1], y_nodes[i], y_nodes[i + 1]
            else:
                x0, x1, x2, x3 = x_nodes[i - 1], x_nodes[i], x_nodes[i + 1], x_nodes[i + 2]
                y0, y1, y2, y3 = y_nodes[i - 1], y_nodes[i], y_nodes[i + 1], y_nodes[i + 2]
            h = x1 - x0
            a = (x1 - x) / h
            b = (x - x0) / h
            c = (x2 - x) / h
            d = (x - x1) / h
            p = a * y0 + b * y1 + (a ** 2 - a) * ((h ** 2) / 6) * y0 + (b ** 2 - b) * ((h ** 2) / 6) * y1
            q = c * y1 + d * y2 + (c ** 2 - c) * ((h ** 2) / 6) * y1 + (d ** 2 - d) * ((h ** 2) / 6) * y2
            return p + (q - p) * ((x - x0) / h)
    if x == x_nodes[0]:  # Если x равно первому узлу
        x0, x1, x2 = x_nodes[0], x_nodes[1], x_nodes[2]
    elif x == x_nodes[-1]:  # Если x равно последнему узлу
        x0, x1, x2 = x_nodes[-3], x_nodes[-2], x_nodes[-1]
    y0, y1, y2 = f(x0), f(x1), f(x2)
    h = x1 - x0
    a = (x1 - x) / h
    b = (x - x0) / h
    c = (x2 - x) / h
    d = (x - x1) / h
    p = a * y0 + b * y1 + (a ** 2 - a) * ((h ** 2) / 6) * y0 + (b ** 2 - b) * ((h ** 2) / 6) * y1
    q = c * y1 + d * y2 + (c ** 2 - c) * ((h ** 2) / 6) * y1 + (d ** 2 - d) * ((h ** 2) / 6) * y2
    return p + (q - p) * ((x - x0) / h)


def global_lagrange_interpolation(x, x_nodes, y_nodes):
    n = len(x_nodes)
    y = 0
    for i in range(n):
        p = 1
        for j in range(n):
            if i != j:
                p *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        y += y_nodes[i] * p
    return y

global_lagrange_values = [global_lagrange_interpolation(x, x_nodes, y_nodes) for x in x_values]
# Код для параллельной программы на всех потоках процессора для ускорения выполнения при большом количестве вычислений
def compute_cubic_lagrange(x):
    return cubic_lagrange_interpolation(x, x_nodes, y_nodes)


# для обчислення малої кількості значень:
cubic_lagrange_values = [compute_cubic_lagrange(x) for x in x_values]
# для обчислення великої кількості значень:
# with ProcessPoolExecutor() as executor:
#     cubic_lagrange_values = list(executor.map(compute_cubic_lagrange, x_values))

# Построение графиков
plt.plot(x_values, f(x_values), color='black', label='Original Function')
plt.plot(x_values, cubic_lagrange_values, color='purple', label='Cubic Lagrange Interpolation')
plt.plot(x_values, global_lagrange_values, color='blue', label='Global Lagrange Interpolation')
plt.scatter(x_nodes, y_nodes, color='red', label='Nodes', s=5, alpha=0.5)
plt.legend()
plt.title('Cubic Lagrange Interpolation for f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

# Собираем результаты в массив numpy
cubic_lagrange_values = np.array(cubic_lagrange_values)

# Создаем таблицу данных
table_data = {'x': x_values, 'f(x)': f(x_values).round(4),
              'Cubic Lagrange Interpolation': cubic_lagrange_values.flatten().round(4), 'Global Lagrange Interpolation': global_lagrange_values}
table = pd.DataFrame(table_data)
pd.set_option('display.max_rows', None)
print(table)

# Вывод времени выполнения программы.
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
