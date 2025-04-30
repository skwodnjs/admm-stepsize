import matplotlib.pyplot as plt

from a_admm_lqp import *

thetas = np.arange(0, 100, 1) + 0.1
iterations = []

# seed = 42
seed = np.random.randint(0, 100000)
np.random.seed(seed)
print(f"seed: {seed}")

param = generate_random_param_of_LQP(m=3, n=5)

for theta in thetas:
    iter_count = admm_LQP(param=param, theta=theta)
    iterations.append(iter_count)

theta_opt = optimal_theta(param)

# print
min_iter = min(iterations)
iter_at_theta_opt = admm_LQP(param=param, theta=theta_opt)

print(f"최소 반복 횟수: {min_iter}")
print(f"최적 θ에서의 반복 횟수 (θ = {theta_opt:.3f}): {iter_at_theta_opt}")

# plot
plt.figure(figsize=(10, 5))
plt.plot(thetas, iterations, marker='o')

for x, y in zip(thetas, iterations):
    plt.text(x, y + 0.5, str(y), ha='center', fontsize=8)

plt.axvline(theta_opt, color='red', linestyle='--')
y_max = max(iterations)
plt.text(theta_opt + 1, y_max + 3, f"θ = {theta_opt:.3f}", color='red', ha='left', fontsize=10)

plt.title("ADMM Convergence")
plt.xlabel("Theta")
plt.ylabel("Number of Iterations")
plt.grid(True)
plt.tight_layout()
plt.show()
