import matplotlib.pyplot as plt

from QPs.admm_qp import *

# 네가 작성한 코드들을 모두 불러왔다고 가정
# from your_module import generate_random_param_of_QP, admm_QP, optimal_theta

rhos = np.arange(0.1, 100.1, 1)
iterations = []

# seed = 42
# np.random.seed(seed)

param = generate_random_param_of_QP(m=3, n=5)

for rho in rhos:
    iter_count = admm_QP(param=param, rho=rho, tol=1e-5, max_iter=100000, verbose=False)
    iterations.append(iter_count)

theta_opt = optimal_theta(param)

# plot
plt.figure(figsize=(10, 5))
plt.plot(rhos, iterations, marker='o')

for x, y in zip(rhos, iterations):
    plt.text(x, y + 0.5, str(y), ha='center', fontsize=8)

plt.axvline(theta_opt, color='red', linestyle='--')
y_max = max(iterations)
plt.text(theta_opt + 1, y_max + 3, f"θ = {theta_opt:.3f}", color='red', ha='left', fontsize=10)

plt.title("ADMM Convergence vs Theta (QP)")
plt.xlabel("Theta")
plt.ylabel("Number of Iterations")
plt.grid(True)
plt.tight_layout()
plt.show()
