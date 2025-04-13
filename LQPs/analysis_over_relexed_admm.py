import matplotlib.pyplot as plt

from over_relexed_admm_lqp import *

thetas = np.arange(0.1, 100.1, 1)
iterations = []

seed = 42
np.random.seed(seed)

param = generate_random_param_of_LQP(m=3, n=5)

for theta in thetas:
    iter_count = over_relexed_admm_LQP(param=param, theta=theta, tol=1e-8, max_iter=1000, verbose=False)
    iterations.append(iter_count)

theta_opt = optimal_theta(param, max_iter=100000, lr=2, tol=1e-4)

# plot
plt.figure(figsize=(10, 5))
plt.plot(thetas, iterations, marker='o')

for x, y in zip(thetas, iterations):
    plt.text(x, y + 0.5, str(y), ha='center', fontsize=8)

plt.axvline(theta_opt, color='red', linestyle='--')
y_max = max(iterations)
plt.text(theta_opt + 1, y_max + 3, f"θ = {theta_opt:.3f}", color='red', ha='left', fontsize=10)

plt.title("over relexed ADMM Convergence")
plt.xlabel("Theta")
plt.ylabel("Number of Iterations")
plt.grid(True)
plt.tight_layout()
plt.show()

# 문제 1. eigenvalue 가 복소수일 수 있음.
# 문제 2. gradient descent가 불안정함. step size 가 작으면 수렴 속도가 엄청 느리고(10만번 돌려도 수렴안함), 너무 크면 특정 조건에서 발산해버림