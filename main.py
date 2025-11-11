import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


a, b = 0.0, 1.0
A, B = -1.0, -2.0
h = 0.01
eps = 0.01


def cauchy(f, u0, x0, xn, h):
    x = x0
    u = np.array(u0, dtype=float)

    x_points = []
    y_points = []

    while x <= xn:
        x_points.append(x)
        y_points.append(u[0])

        step1 = f(x, u)
        u_pred = u + h * np.array(step1)

        step2 = f(x + h, u_pred)
        k_avg = (np.array(step1) + np.array(step2)) / 2

        u_next = u + h * k_avg
        u = u_next
        x += h

    return np.array(x_points), np.array(y_points)

def f(x, u):
    y, z = u[0], u[1]
    dy = z
    dz = z - x
    return np.array([dy, dz])

def Phi(eta):
    x_pts, y_pts = cauchy(f, [A, eta], a, b, h)
    return y_pts[-1] - B

def create_table(min_x, max_x, shift, func):
    x_values = np.arange(min_x, max_x + shift, shift)
    f_values = [func(x) for x in x_values]
    table = pd.DataFrame({'x': x_values, 'f(x)': f_values})
    return table

def localize(table_of_values):
    intervals = []
    for i in range(len(table_of_values) - 1):
        if table_of_values.iloc[i]['f(x)'] * table_of_values.iloc[i + 1]['f(x)'] < 0:
            intervals.append((float(table_of_values.iloc[i]['x']), float(table_of_values.iloc[i + 1]['x'])))
    return intervals

def bisection_method_with_curves(a, b, eps, func):
    curves = []
    while (b - a) > 2 * eps:
        mid = (a + b) / 2
        x_pts, y_pts = cauchy(f, [A, mid], 0, 1, h)
        phi_val = func(mid)
        curves.append((mid, phi_val, x_pts, y_pts))
        if func(mid) == 0:
            return float(mid), curves
        elif func(a) * func(mid) < 0:
            b = mid
        else:
            a = mid

    eta_star = float((a + b) / 2)
    x_pts, y_pts = cauchy(f, [A, eta_star], 0, 1, h)
    phi_val = func(eta_star)
    curves.append((eta_star, phi_val, x_pts, y_pts))

    return eta_star, curves


eta_min, eta_max = -1.0, 1.0
shift = 0.2

table_phi = create_table(eta_min, eta_max, shift, Phi)
# print("\nТаблица значений функции Φ(η):\n")
# print(table_phi)

intervals = localize(table_phi)
# print("\nЛокализованные интервалы для Φ(η):", intervals)

eta_star, trial_curves = bisection_method_with_curves(intervals[0][0], intervals[0][1], eps, Phi)
phi_star = Phi(eta_star)

print(f"\nНайденное значение η* = {eta_star:.6f}")
print(f"Φ(η*) = {phi_star:.6f}")
print(f"Количество пробных выстрелов: {len(trial_curves)}")


plt.figure(figsize=(8, 6))
for eta_val, phi_val, x_pts, y_pts in trial_curves[:-1]:
    plt.plot(x_pts, y_pts, '--', alpha=0.6, linewidth=1,
             label=f"η={eta_val:.4f}, Φ={phi_val:.4f}")

eta_final, phi_final, x_final, y_final = trial_curves[-1]
plt.plot(x_final, y_final, color='navy', linewidth=2.2,
         label=f"η*={eta_final:.4f}, Φ={phi_final:.4f}")

plt.scatter([b], [B], marker='x', s=80, color='red', label="Цель (x=1, y=-2)")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Метод стрельбы: все интегральные кривые (y'' = y' - x)")
plt.legend(fontsize=8)
plt.grid(True)
plt.show()