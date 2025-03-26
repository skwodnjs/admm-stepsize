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
rho = 1

primal_residuals = np.zeros(T)
dual_residuals = np.zeros(T)

x_old = x_init
z_old = z_init
u_old = u_init

for it in range(T):
    # ADMM 업데이트
    x_new = -np.linalg.inv(Q + rho * A.T.dot(A)).dot(q + rho * A.T.dot(z_old + u_old - c))
    z_new = np.maximum(np.zeros((m + 2 * p, 1)), -A.dot(x_new) - u_old + c)
    u_new = u_old + rho * (A.dot(x_new) - c + z_new)

    # ▶ 수렴 속도 측정용 residual 계산
    r = A.dot(x_new) + z_new - c                               # primal residual
    s = rho * A.T.dot(z_new - z_old)                           # dual residual

    primal_residuals[it] = np.linalg.norm(r)
    dual_residuals[it] = np.linalg.norm(s)

    # 업데이트
    x_old = x_new
    z_old = z_new
    u_old = u_new

# ▶ 로그 스케일로 수렴 속도 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(np.log10(primal_residuals + .1e-15), label='log10(Primal Residual)')
plt.plot(np.log10(dual_residuals + .1e-15), label='log10(Dual Residual)')
plt.xlabel('Iteration')
plt.ylabel('Log Residual')
plt.title('ADMM Convergence Residuals')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
