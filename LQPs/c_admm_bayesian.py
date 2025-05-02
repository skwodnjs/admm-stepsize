import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings, simplefilter

from a_admm_lqp import generate_random_param_of_LQP, admm_LQP, optimal_theta

# 설정
# seed = 42     # 21758, 25465
seed = np.random.randint(0, 100000)
np.random.seed(seed)
print(f"seed: {seed}")

param = generate_random_param_of_LQP(m=3, n=5)
max_iter = 1000

theta_max = 30

# 목적 함수
def objective(theta):
    _, _, iter_count, _ = admm_LQP(param=param, theta=theta, max_iter=max_iter)
    if iter_count is not None:
        return -iter_count

# GP 예측 함수
def surrogate(model, X):
    with catch_warnings():
        simplefilter("ignore")
        return model.predict(X, return_std=True)

# Acquisition Function (Probability of Improvement)
# PI
# def acquisition(X, Xsamples, model):
#     yhat, _ = surrogate(model, X)
#     best = max(yhat)
#     mu, std = surrogate(model, Xsamples)
#     probs = norm.cdf((mu - best) / (std + 1e-9))
#     return probs
# EI
def acquisition(X, Xsamples, model):
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    mu, std = surrogate(model, Xsamples)
    mu = mu.flatten()
    std = std.flatten()
    imp = mu - best
    Z = imp / (std + 1e-9)
    ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
    return ei

# Acquisition 최적화 (n개 랜덤한 지점 중 다음 탐색 지점 결정)
def opt_acquisition(X, model, n=50):
    Xsamples = np.random.random(n) * theta_max + 0.1
    Xsamples = Xsamples.reshape(-1, 1)
    scores = acquisition(X, Xsamples, model)
    ix = np.argmax(scores)
    return Xsamples[ix, 0]

# surrogate 시각화
def plot(X, y, model):
    plt.scatter(X, y, label="observed")
    Xsamples = np.arange(0.1, theta_max, 0.05).reshape(-1, 1)
    ysamples, _ = surrogate(model, Xsamples)
    plt.plot(Xsamples, ysamples, label="surrogate")
    plt.xlabel("theta")
    plt.ylabel("-iteration count (objective)")
    plt.title("Surrogate model of ADMM iteration")
    plt.grid(True)
    plt.legend()
    plt.show()

# 초기 샘플 수집
X_raw = np.random.random(50) * theta_max + 0.1
# X_raw = np.random.random(30) * theta_max + 0.1
X, y = [], []
for x in X_raw:
    val = objective(x)
    if val is not None:
        X.append([x])
        y.append([val])
X = np.asarray(X)
y = np.asarray(y)

# surrogate 모델 정의 및 초기 학습
model = GaussianProcessRegressor()
model.fit(X, y)
plot(X, y, model)

# 최적화 루프
for i in range(50):
    x = opt_acquisition(X, model)
    actual = objective(x)
    if actual is None:
        print(f"> Iter {i+1:02d}: theta={x:.4f}, skipped (diverged or max_iter reached)")   # 탐색한 지점에서 ADMM이 MAX_ITER에 도달함
        continue
    est, _ = surrogate(model, [[x]])
    print(f"> Iter {i+1:02d}: theta={x:.4f}, predicted={est[0]:.3f}, actual={actual:.3f}")  # 탐색 결과: 예측값 & 실제값
    X = np.vstack((X, [[x]]))
    y = np.vstack((y, [[actual]]))
    model.fit(X, y)     # 실제 값으로 GP 모델 학습

ix = np.argmax(y)
theta_bayesian = X[ix][0]
theta_opt, _ = optimal_theta(param=param)

# ADMM 실행 (Bayesian 결과)
(u_bayes, _), obj_bayes, iter_bayes, time_bayes = admm_LQP(param=param, theta=theta_bayesian)

# ADMM 실행 (Gradient Descent 결과)
(u_gd, _), obj_gd, iter_gd, time_gd = admm_LQP(param=param, theta=theta_opt)

print("\n📌 ADMM θ 탐색 결과 요약")
print(f"▶ Bayesian Optimization θ \t: {theta_bayesian:.6f}")
print(f"  ↪ 반복 횟수              \t: {iter_bayes}")
print(f"  ↪ u                     \t: {u_bayes}")
print(f"  ↪ objective value       \t: {obj_bayes:.9f}")
print(f"  ↪ 수행 시간              \t: {time_bayes:.6f} sec")
print()
print(f"▶ Gradient Descent θ*     \t: {theta_opt:.6f}")
print(f"  ↪ 반복 횟수              \t: {iter_gd}")
print(f"  ↪ u                     \t: {u_gd}")
print(f"  ↪ objective value       \t: {obj_gd:.9f}")
print(f"  ↪ 수행 시간              \t: {time_gd:.6f} sec")

# 최종 결과 시각화
plot(X, y, model)
