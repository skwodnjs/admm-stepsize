import matplotlib.pyplot as plt

from admm_lqp import *

thetas = np.arange(0.1, 100.1, 1)
iterations = []

seed = 42
np.random.seed(seed)

param = generate_random_param_of_LQP(m=3, n=5)

for theta in thetas:
    iter_count = admm_LQP(param=param, theta=theta, tol=1e-8, max_iter=1000, verbose=False)
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

plt.title("ADMM Convergence")
plt.xlabel("Theta")
plt.ylabel("Number of Iterations")
plt.grid(True)
plt.tight_layout()
plt.show()
