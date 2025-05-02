import matplotlib.pyplot as plt

from g_admm_qp import *

rhos = np.arange(0, 100, 1) + 0.1
iterations = []

# seed = 35600
seed = np.random.randint(0, 100000)
np.random.seed(seed)
print(f"seed: {seed}")

param = generate_random_param_of_QP(m=3, n=5)

for rho in rhos:
    iter_count = admm_QP(param=param, rho=rho)
    iterations.append(iter_count)

rho_opt = optimal_rho(param)

# print
min_iter = min(iterations)
iter_at_rho_opt = admm_QP(param=param, rho=rho_opt)

print(f"최소 반복 횟수: {min_iter}")
print(f"최적 rho에서의 반복 횟수 (rho = {rho_opt:.3f}): {iter_at_rho_opt}")

# plot
plt.figure(figsize=(10, 5))
plt.plot(rhos, iterations, marker='o')

for x, y in zip(rhos, iterations):
    plt.text(x, y + 0.5, str(y), ha='center', fontsize=8)

plt.axvline(rho_opt, color='red', linestyle='--')
y_max = max(iterations)
plt.text(rho_opt + 1, y_max + 3, f"rho = {rho_opt:.3f}", color='red', ha='left', fontsize=10)

plt.title("ADMM Convergence")
plt.xlabel("rho")
plt.ylabel("Number of Iterations")
plt.grid(True)
plt.tight_layout()
plt.show()
