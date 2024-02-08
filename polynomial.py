def lagrange_interpolation(x_values, y_values, x):
    result = 0

    for i in range(len(y_values)):
        term = y_values[i]
        for j in range(len(x_values)):
            if j != i:
                term = term * (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term

    return result

# Приклад використання:
x_values = [1, 2, 4]
y_values = [3, 1, 5]

interpolation_point = 5
result = lagrange_interpolation(x_values, y_values, interpolation_point)

print(f"Значення інтерполяційного поліному в точці {interpolation_point}: {result}")
