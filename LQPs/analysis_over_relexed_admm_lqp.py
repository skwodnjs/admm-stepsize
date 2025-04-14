import matplotlib.pyplot as plt

from over_relexed_admm_lqp import *

thetas = np.arange(0.1, 100.1, 1)
iterations_theta = []

seed = 42
np.random.seed(seed)

param = generate_random_param_of_LQP(m=3, n=5)

for theta in thetas:
    iter_count = over_relexed_admm_LQP(param=param, theta=theta, alpha=2, tol=1e-8, max_iter=1000, verbose=False)
    iterations_theta.append(iter_count)

theta_opt = optimal_theta(param, max_iter=100000, lr=2, tol=1e-4)

alphas = np.arange(1.1, 2.1, 1e-2)
iterations_alpha = []

for alpha in alphas:
    iter_count = over_relexed_admm_LQP(param=param, theta=theta_opt, alpha=alpha, tol=1e-8, max_iter=1000, verbose=False)
    iterations_alpha.append(iter_count)

# 두 개의 그래프를 수직으로 배치
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharey=False)

# (1) Theta에 따른 반복 횟수
axes[0].plot(thetas, iterations_theta, marker='o')
for x, y in zip(thetas, iterations_theta):
    axes[0].text(x, y + 0.5, str(y), ha='center', fontsize=8)

axes[0].axvline(theta_opt, color='red', linestyle='--')
y_max = max(iterations_theta)
axes[0].text(theta_opt + 1, y_max + 3, f"θ = {theta_opt:.3f}", color='red', ha='left', fontsize=10)

axes[0].set_title("Over-relaxed ADMM: Iterations vs θ (α = 2)")
axes[0].set_xlabel("Theta")
axes[0].set_ylabel("Number of Iterations")
axes[0].grid(True)

# (2) Alpha에 따른 반복 횟수 (최적 θ 고정)
axes[1].plot(alphas, iterations_alpha, marker='o')
for x, y in zip(alphas, iterations_alpha):
    axes[1].text(x, y + 0.5, str(y), ha='center', fontsize=8)

axes[1].axvline(2.0, color='red', linestyle='--')
y_max_alpha = max(iterations_alpha)
axes[1].text(2.01, y_max_alpha + 3, "α = 2.0", color='red', ha='left', fontsize=10)

axes[1].set_title(f"Over-relaxed ADMM: Iterations vs α (θ = {theta_opt:.3f})")
axes[1].set_xlabel("Alpha")
axes[1].set_ylabel("Number of Iterations")
axes[1].grid(True)

plt.tight_layout()
plt.show()

# 문제 1. eigenvalue 가 복소수일 수 있음.
# 문제 2. gradient descent가 불안정함. step size 가 작으면 수렴 속도가 엄청 느리고(10만번 돌려도 수렴안함), 너무 크면 특정 조건에서 발산해버림