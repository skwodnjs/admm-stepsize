import matplotlib.pyplot as plt

import numpy as np

# np.random.seed(42)

# We want to solve
# (1/2) x^T Q x + q^T x
# subject to Ax <= c.
# Here we add Bx = b. For this we consider Bx <= b, Bx >= b

n = 10
m = 4

Perturb = 0.2 * np.random.randn(n, n)
Q = 15 * np.eye(n) + Perturb + Perturb.T

q = np.random.randn(n, 1)

A = np.random.randn(m, n)
c = np.random.randn(m, 1)

# f(x) = (1/2) x^T Q x + q^T x
# g(z) = \delta_{R^+} (z)

x_init = np.random.randn(n, 1)
z_init = np.random.randn(m, 1)
u_init = np.random.randn(m, 1)

T = 500

rho_list = np.arange(0.1, 8.0, 0.05)

residuals_dict = {}

ep = 0.1e-10

for rho in rho_list:
    # init
    x_old = np.copy(x_init)
    z_old = np.copy(z_init)
    u_old = np.copy(u_init)

    primal_residuals = np.zeros(T)
    dual_residuals = np.zeros(T)
    l2_combined = np.zeros(T)

    for it in range(T):
        # ADMM
        x_new = -np.linalg.inv(Q + rho * A.T.dot(A)).dot(q + rho * A.T.dot(z_old + u_old - c))
        z_new = np.maximum(np.zeros((m, 1)), -A.dot(x_new) - u_old + c)
        u_new = u_old + (A.dot(x_new) - c + z_new)

        # residual 계산
        r = A.dot(x_new) + z_new - c                               # primal residual
        s = rho * A.T.dot(z_new - z_old)                           # dual residual

        if np.linalg.norm(r) < ep and np.linalg.norm(s) < ep:
            residuals_dict[rho] = it
            break
        elif it == T-1:
            residuals_dict[rho] = T

        x_old = x_new
        z_old = z_new
        u_old = u_new

# 시각화 준비
sorted_rhos = sorted(residuals_dict.keys())
x_rho = sorted_rhos
y_iters = [residuals_dict[rho] for rho in x_rho]

# 시각화
plt.figure(figsize=(8, 5))
plt.plot(x_rho, y_iters, marker='o')

# 값 표시
for x, y in zip(x_rho, y_iters):
    plt.text(x, y + 5, str(y), ha='center', va='bottom', fontsize=9)

plt.xlabel('ρ (rho)')
plt.ylabel('Number of iterations to converge')
plt.title('Iterations required for each $\\rho$')
plt.grid(True)
plt.tight_layout()
plt.show()