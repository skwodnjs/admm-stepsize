import matplotlib.pyplot as plt

from d_over_relexed_admm_lqp import *

thetas = np.arange(0, 100, 1) + 0.1
iterations = []

# seed = 42
seed = np.random.randint(0, 100000)
np.random.seed(seed)
print(f"seed: {seed}")

param = generate_random_param_of_LQP(m=3, n=5)

for theta in thetas:
    iter_count = over_relexed_admm_LQP(param=param, theta=theta)
    iterations.append(iter_count)

theta_opt = optimal_theta(param)

alphas = np.arange(0.9, 2.1, 1e-2)
iterations_alpha = []

for alpha in alphas:
    iter_count = over_relexed_admm_LQP(param=param, theta=theta_opt, alpha=alpha)
    iterations_alpha.append(iter_count)

# 두 개의 그래프를 수직으로 배치
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharey=False)

# (1) Theta에 따른 반복 횟수
axes[0].plot(thetas, iterations, marker='o')
for x, y in zip(thetas, iterations):
    axes[0].text(x, y + 0.5, str(y), ha='center', fontsize=8)

axes[0].axvline(theta_opt, color='red', linestyle='--')
y_max = max(iterations)
axes[0].text(theta_opt + 1, y_max + 3, f"θ = {theta_opt:.3f}", color='red', ha='left', fontsize=10)

axes[0].set_title("Over-relaxed ADMM: Iterations vs θ (α = 2)")
axes[0].set_xlabel("Theta")
axes[0].set_ylabel("Number of Iterations")
axes[0].grid(True)

# (2) Alpha에 따른 반복 횟수 (최적 θ 고정)
axes[1].plot(alphas, iterations_alpha, marker='o')
for x, y in zip(alphas, iterations_alpha):
    axes[1].text(x, y + 0.5, str(y), ha='center', fontsize=8)

# axes[1].axvline(2.0, color='red', linestyle='--')
# y_max_alpha = max(iterations_alpha)
# axes[1].text(2.01, y_max_alpha + 3, "α = 2.0", color='red', ha='left', fontsize=10)

axes[1].set_title(f"Over-relaxed ADMM: Iterations vs α (θ = {theta_opt:.3f})")
axes[1].set_xlabel("Alpha")
axes[1].set_ylabel("Number of Iterations")
axes[1].grid(True)

plt.tight_layout()
plt.show()