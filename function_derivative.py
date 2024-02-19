import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Вхідні дані для обчислень
x_init = [-0.11, 0.11, 0.24, 0.36, 0.57, 0.66, 0.89, 1.1, 1.39, 1.6]
y_init = [0.7764, 1.2382, 1.5394, 1.8374, 2.4101, 2.6780, 3.4356, 4.2396, 5.5884, 6.7991]

# Генерація значень x для обчислень
x_values = np.arange(0, 1.5 + 0.01, 0.01)
h = 0.01
y_direct1, y_direct2, y_dif12, y_dif22 = [], [], [], []

# Обчислення похідних функції
for x0 in x_values:
    dif1 = np.log(3) * 3 ** x0 + np.cos(x0)
    dif2 = np.log(3) ** 2 * 3 ** x0 - np.sin(x0)
    y_direct1.append(dif1)
    y_direct2.append(dif2)


# Функція для знаходження найближчих значень x
def find_closest_values(x_init, x_input):
    for i in range(3, len(x_init)):
        if x_init[i] >= x_input:
            return x_init[i - 3], x_init[i - 2], x_init[i - 1], x_init[i]
    return x_init[-4], x_init[-3], x_init[-2], x_init[-1]


# Кубічна інтерполяція
def f(x):
    x0, x1, x2, x3 = find_closest_values(x_init, x)
    y0, y1, y2, y3 = y_init[x_init.index(x0)], y_init[x_init.index(x1)], y_init[x_init.index(x2)], y_init[
        x_init.index(x3)]
    y_cubic = y0 * ((x - x1) * (x - x2) * (x - x3)) / ((x0 - x1) * (x0 - x2) * (x0 - x3)) + \
              y1 * ((x - x0) * (x - x2) * (x - x3)) / ((x1 - x0) * (x1 - x2) * (x1 - x3)) + \
              y2 * ((x - x0) * (x - x1) * (x - x3)) / ((x2 - x0) * (x2 - x1) * (x2 - x3)) + \
              y3 * ((x - x0) * (x - x1) * (x - x2)) / ((x3 - x0) * (x3 - x1) * (x3 - x2))
    return y_cubic


# Обчислення чисельних похідних
for i in range(0, len(x_values)):
    if i == 0:
        diff1 = (f(x_values[i + 1]) - f(x_values[i])) / h
        y_dif12.append(diff1)
    elif i == len(x_values) - 1:
        diff1 = (f(x_values[i]) - f(x_values[i - 1])) / h
        y_dif12.append(diff1)
    else:
        diff1 = (f(x_values[i + 1]) - f(x_values[i - 1])) / (2 * h)
        y_dif12.append(diff1)

    if i == 0:
        diff2 = (f(x_values[i + 1]) + f(x_values[i] - h) - 2 * f(x_values[i])) / h ** 2
        y_dif22.append(diff2)
    elif i == len(x_values) - 1:
        diff2 = (f(x_values[i] + h) + f(x_values[i - 1]) - 2 * f(x_values[i])) / h ** 2
        y_dif22.append(diff2)
    else:
        diff2 = (f(x_values[i + 1]) + f(x_values[i - 1]) - 2 * f(x_values[i])) / h ** 2
        y_dif22.append(diff2)

# Побудова графіків
fig = plt.figure(figsize=(10, 10))
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Інтерпольоване та диференційне значення функції')
ax1.plot(x_values, y_direct1, label="f'(x) диференційний")
ax1.plot(x_values, y_dif12, label="f'(x) інтерпольоване")
ax1.legend()
ax2.plot(x_values, y_direct2, label="f''(x) диференційний")
ax2.plot(x_values, y_dif22, label="f''(x) інтерпольоване")
ax2.legend()
plt.show()

# Вивід результатів у вигляді таблиці
df = pd.DataFrame({
    "f'(x) диференційний": y_direct1,
    "f'(x) інтерпольоване": y_dif12,
    "f''(x) диференційний": y_direct2,
    "f''(x) інтерпольоване": y_dif22
})

pd.set_option('display.max_rows', None)

print(df.to_string(index=False))
