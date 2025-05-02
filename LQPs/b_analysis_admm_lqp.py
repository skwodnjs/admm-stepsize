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
    _, _, iter_count, _ = admm_LQP(param=param, theta=theta)
    iterations.append(iter_count)

theta_opt, _ = optimal_theta(param)

# print
min_iter = min([i for i in iterations if i is not None])
min_index = iterations.index(min_iter)
theta_min = thetas[min_index]
_, _, iter_at_theta_opt, time_theta_opt = admm_LQP(param=param, theta=theta_opt)

print("\n📌 ADMM 반복 횟수 요약")
print(f"▶ 최소 반복 횟수      \t: {min_iter} (θ = {theta_min:.3f})")
print(f"▶ optimal θ         \t: {theta_opt:.6f}")
print(f"  ↪ 해당 θ의 반복 횟수 \t: {iter_at_theta_opt}")
print(f"  ↪ admm 시간        \t: {time_theta_opt:.6f} sec")

# plot
plt.figure(figsize=(10, 5))
plt.plot(thetas, iterations, marker='o')

for x, y in zip(thetas, iterations):
    if y is not None:
        plt.text(x, y + 0.5, str(y), ha='center', fontsize=8)

plt.axvline(theta_opt, color='red', linestyle='--')
y_max = max([i for i in iterations if i is not None])
plt.text(theta_opt + 1, y_max + 3, f"θ = {theta_opt:.3f}", color='red', ha='left', fontsize=10)

plt.title("ADMM Convergence")
plt.xlabel("Theta")
plt.ylabel("Number of Iterations")
plt.grid(True)
plt.tight_layout()
plt.show()
