import numpy as np
import matplotlib.pyplot as plt

# 실수 x 구간
x = np.linspace(-5, 5, 400)

center = -0.5 + 0j
angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
v_list = np.concatenate([
    center + 0.4 * np.exp(1j * angles),
    center + 0.3 * np.exp(1j * angles),
    center + 0.2 * np.exp(1j * angles),
])

# 그래프
plt.figure(figsize=(10, 6))
for v in v_list:
    y = np.abs(1 + v * x)
    label = f"|1 + ({v.real:.1f} {'+' if v.imag >= 0 else '-'} {abs(v.imag):.1f}i)x|"
    plt.plot(x, y, label=label)

plt.title("Graphs of f(x) = |1 + alpha·lambda| for various complex v with |lambda| < 1")
plt.xlabel("alpha")
plt.ylabel("|1 + alpha·lambda|")
plt.grid(True)
plt.legend()
plt.show()
