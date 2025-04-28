import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# A. Movimiento de un cometa

# Constantes
GM = 1.32712440018e11  # km^3 / s^2

# Condiciones iniciales
x0 = 4e9      # km
y0 = 0.0      # km
z0 = 0.0      # km
vx0 = 0.0     # km/s
vy0 = 0.5     # km/s
vz0 = 0.0     # km/s

y_init = np.array([x0, y0, z0, vx0, vy0, vz0])

# Periodo orbital estimado
a = x0
T = 2 * np.pi * np.sqrt(a**3 / GM)  # en segundos
print(f"Período estimado: {T/3600/24:.2f} días")

# Intervalo de tiempo (5 órbitas)
t0 = 0
tf = 5 * T
h = 50000  # s
n_steps = int((tf - t0) / h)

# Derivadas del sistema (ECM)
def derivadas(t, y):
    x, y_, z, vx, vy, vz = y
    r = np.sqrt(x**2 + y_**2 + z**2)
    ax = -GM * x / r**3
    ay = -GM * y_ / r**3
    az = -GM * z / r**3
    return np.array([vx, vy, vz, ax, ay, az])

# RK4 con paso fijo
def rk4(f, y0, t0, tf, h):
    t_values = [t0]
    y_values = [y0]
    t = t0
    y = y0.copy()

    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t += h
        t_values.append(t)
        y_values.append(y.copy())

    return np.array(t_values), np.array(y_values)

# Energía y momento angular específico
def calcular_conservadas(y_vals):
    energías = []
    momentos = []
    for y in y_vals:
        x, y_, z, vx, vy, vz = y
        r = np.sqrt(x**2 + y_**2 + z**2)
        v2 = vx**2 + vy**2 + vz**2
        energía = 0.5 * v2 - GM / r
        momento = np.cross([x, y_, z], [vx, vy, vz])
        energías.append(energía)
        momentos.append(np.linalg.norm(momento))
    return np.array(energías), np.array(momentos)

# Integración con RK4
t_rk4, y_rk4 = rk4(derivadas, y_init, t0, tf, h)
x_rk4 = y_rk4[:, 0]
y_rk4_plot = y_rk4[:, 1]
E_rk4, L_rk4 = calcular_conservadas(y_rk4)

# Gráfica de órbita
plt.figure(figsize=(8, 8))
plt.plot(x_rk4, y_rk4_plot, label="Órbita RK4")
plt.plot(0, 0, 'yo', label="Sol")
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.title("Punto 1: Órbita del cometa con RK4")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

# Gráfica de conservación del momento angular
plt.figure(figsize=(10, 4))
plt.plot(t_rk4 / (3600*24), L_rk4)
plt.xlabel("Tiempo (días)")
plt.ylabel("||L|| (km²/s)")
plt.title("Punto 1: Conservación del momento angular (RK4)")
plt.grid(True)
plt.show()

# Método RK45 adaptativo con solve_ivp
sol_adapt = solve_ivp(
    derivadas, (t0, tf), y_init, method='RK45',
    rtol=1e-9, atol=1e-9
)

t_adapt = sol_adapt.t
y_adapt = sol_adapt.y
x_adapt = y_adapt[0]
y_adapt_plot = y_adapt[1]

# Energía y momento angular para RK45
E_adapt, L_adapt = calcular_conservadas(y_adapt.T)

# Comparación x vs t
plt.figure(figsize=(10, 4))
plt.plot(t_rk4 / (3600*24), x_rk4, label="RK4 (paso fijo)")
plt.plot(t_adapt / (3600*24), x_adapt, '--', label="RK45 (adaptativo)")
plt.xlabel("Tiempo (días)")
plt.ylabel("x (km)")
plt.title("Punto 2: Comparación de x vs t")
plt.legend()
plt.grid(True)
plt.show()

# Comparación y vs t
plt.figure(figsize=(10, 4))
plt.plot(t_rk4 / (3600*24), y_rk4_plot, label="RK4 (paso fijo)")
plt.plot(t_adapt / (3600*24), y_adapt_plot, '--', label="RK45 (adaptativo)")
plt.xlabel("Tiempo (días)")
plt.ylabel("y (km)")
plt.title("Punto 2: Comparación de y vs t")
plt.legend()
plt.grid(True)
plt.show()

# Tamaño de paso adaptativo
step_sizes = np.diff(t_adapt)
plt.figure(figsize=(10, 4))
plt.plot(t_adapt[:-1] / (3600*24), step_sizes)
plt.xlabel("Tiempo (días)")
plt.ylabel("Tamaño del paso (s)")
plt.title("Punto 2: Tamaño de paso adaptativo (solve_ivp)")
plt.grid(True)
plt.show()

# Conservación de energía y momento angular con solve_ivp
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t_adapt / (3600*24), E_adapt)
plt.title("Energía específica (RK45)")
plt.xlabel("Tiempo (días)")
plt.ylabel("Energía (km²/s²)")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(t_adapt / (3600*24), L_adapt)
plt.title("Momento angular específico (RK45)")
plt.xlabel("Tiempo (días)")
plt.ylabel("Momento angular (km²/s)")
plt.grid()

plt.tight_layout()
plt.show()
