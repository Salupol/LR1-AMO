import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x_init = [-0.11, 0.11, 0.24, 0.36, 0.57, 0.66, 0.89, 1.1, 1.39, 1.6]
y_init = [0.7764, 1.2382, 1.5394, 1.8374, 2.4101, 2.6780, 3.4356, 4.2396, 5.5884, 6.7991]
x_args = np.arange(0, 1.5 + 0.01, 0.01)

y_args_formula = [3 ** i + np.sin(i) for i in x_args]


def cubic_interpolation(x, y, x_args):
    def find_closest_values(x_input):
        for i in range(3, len(x)):
            if x[i] >= x_input:
                return x[i - 3:i + 1]
        return x[-4:]

    y_args_cubic = []
    for x_input in x_args:
        x_values = find_closest_values(x_input)
        y_values = [y[x.index(x_val)] for x_val in x_values]
        y_cubic = sum(
            y_values[i] * np.prod([(x_input - x_values[j]) / (x_values[i] - x_values[j]) for j in range(4) if i != j])
            for i in range(4))
        y_args_cubic.append(y_cubic)
    return y_args_cubic


def lagrange_interpolation(x, y, x_args):
    y_args_lagrange = []
    for x_input in x_args:
        L = sum(
            y[i] * np.prod([(x_input - x[j]) / (x[i] - x[j]) for j in range(len(x)) if i != j]) for i in range(len(x)))
        y_args_lagrange.append(L)
    return y_args_lagrange


y_args_cubic = cubic_interpolation(x_init, y_init, x_args)
y_args_lagrange = lagrange_interpolation(x_init, y_init, x_args)
plt.figure(figsize=(10, 10), dpi=300)
plt.plot(x_args, y_args_formula, linewidth=3, label="initial formula")
plt.plot(x_args, y_args_cubic, linewidth=3, label="cubic interpolation")
plt.plot(x_args, y_args_lagrange, linewidth=3, label="lagrange interpolation")
plt.plot(x_init, y_init, 'ro', label="initial values")
plt.legend()
plt.show()

df = pd.DataFrame({"x": x_args, "y_formula": y_args_formula, "y_cubic": y_args_cubic, "y_lagrange": y_args_lagrange})
pd.set_option('display.max_rows', None)
print(df)
