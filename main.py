import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

np.random.seed(42)

# We want to solve
# (1/2) x^T Q x + q^T x
# subject to Ax <= c.
# Here we add Bx = b. For this we consider Bx <= b, Bx >= b

n = 10
m = 4
p = 2

Perturb = 0.2 * np.random.randn(n, n)
Q = 15 * np.eye(n) + Perturb + Perturb.T
q = np.random.randn(n, 1)

A = np.random.randn(m, n)
c = np.random.randn(m, 1)

B = np.random.randn(p, n)
b = np.random.randn(p, 1)

A = np.concatenate((A, B, -B), axis=0)  # A_new
c = np.concatenate((c, b, -b), axis=0)  # c_new

# f(x) = (1/2) x^T Q x + q^T x
# g(z) = \delta_{R^+} (z)

x_init = np.random.randn(n, 1)
z_init = np.random.randn(m + 2 * p, 1)
u_init = np.random.randn(m + 2 * p, 1)

T = 500

# rho
rho_list = [0.5, 1.0, 1.5, 2.0]
labels = ['rho = 0.5', 'rho = 1.0', 'rho = 1.5', 'rho = 2.0']

residuals_dict = {}

for idx, rho in enumerate(rho_list):
    # init
    x_old = np.copy(x_init)
    z_old = np.copy(z_init)
    u_old = np.copy(u_init)

    primal_residuals = np.zeros(T)
    dual_residuals = np.zeros(T)

    for it in range(T):
        # ADMM
        x_new = -np.linalg.inv(Q + rho * A.T.dot(A)).dot(q + rho * A.T.dot(z_old + u_old - c))
        z_new = np.maximum(np.zeros((m + 2 * p, 1)), -A.dot(x_new) - u_old + c)
        u_new = u_old + (A.dot(x_new) - c + z_new)

        # residual 계산
        r = A.dot(x_new) + z_new - c                               # primal residual
        s = rho * A.T.dot(z_new - z_old)                           # dual residual

        primal_residuals[it] = np.linalg.norm(r)
        dual_residuals[it] = np.linalg.norm(s)

        x_old = x_new
        z_old = z_new
        u_old = u_new

        # 저장
        residuals_dict[labels[idx]] = (primal_residuals, dual_residuals)

# 고유한 rho 값 추출
rho_values = [label.split('=')[1].strip() for label in labels]
unique_rhos = sorted(set(rho_values))

# rho → color 매핑 (컬러맵 사용)
cmap = mpl.colormaps.get_cmap('tab10').resampled(len(unique_rhos))
rho_to_color = {rho: cmap(i) for i, rho in enumerate(unique_rhos)}

# 시각화
plt.figure(figsize=(10, 5))
for label in labels:
    rho = label.split('=')[1].strip()
    color = rho_to_color[rho]
    primal_residuals, dual_residuals = residuals_dict[label]

    plt.plot(np.log10(primal_residuals + 1e-15), color=color, label=f'{label} (Primal)')
    plt.plot(np.log10(dual_residuals + 1e-15), color=color, linestyle='--', label=f'{label} (Dual)')

plt.xlabel('Iteration')
plt.ylabel('Log Residual')
plt.title('ADMM Convergence for Different $\\rho$ Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()