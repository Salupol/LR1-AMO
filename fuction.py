import math

def f(x):
    return 3**x + math.sin(x)

start_x = 0
end_x = 1.5
step = 0.01

table = []

current_x = start_x
while current_x <= end_x:
    table.append((current_x, f(current_x)))
    current_x += step

# Виведення таблиці
print("x \t| f(x)")
print("--------------------")
for entry in table:
    print(f"{entry[0]:.2f} \t| {entry[1]:.4f}")
print("--------------------")
def lagrange_interpolation(x_values, y_values, x):
    result = 0

    for i in range(len(y_values)):
        term = y_values[i]
        for j in range(len(x_values)):
            if j != i:
                term = term * (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term

    return result
def piecewise_lagrange_interpolation(start_x, end_x, step, degree):
    current_x = start_x
    x_values = []
    y_values = []

    while current_x <= end_x:
        x_values.append(current_x)
        y_values.append(f(current_x))
        current_x += step

    print("x \t| f(x)")
    print("--------------------")

    for i in range(len(x_values)):
        current_segment_x = x_values[i:i+degree+1]
        current_segment_y = y_values[i:i+degree+1]

        interpolation_point = (current_segment_x[0] + current_segment_x[-1]) / 2
        interpolation_result = lagrange_interpolation(current_segment_x, current_segment_y, interpolation_point)

        print(f"{interpolation_point:.2f} \t| {interpolation_result:.4f}")

piecewise_lagrange_interpolation(0, 1.5, 0.01, 2)