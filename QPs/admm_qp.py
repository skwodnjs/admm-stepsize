import numpy as np

def generate_random_param_of_QP(m=100, n=150):
    R = np.random.randn(n, n)
    Q_mat = R @ R.T + n * np.eye(n)

    q = np.random.randn(n)

    if m >= n:
        Q, _ = np.linalg.qr(np.random.randn(m, n))
        A = Q[:, :n]
    else:
        Q, _ = np.linalg.qr(np.random.randn(n, m))
        A = Q[:, :m].T

    c = np.random.randn(m)

    return {"Q": Q_mat, "q": q, "A": A, "c": c}

def generate_random_QP(param):
    Q, q = param["Q"], param["q"]

    def objective(x):
        return 0.5 * x.T @ Q @ x + q.T @ x

    return objective

def admm_QP(param, rho=0.5, tol=1e-8, max_iter=1000, verbose=True):
    Q, q, A, c = param["Q"], param["q"], param["A"], param["c"]
    m, n = A.shape

    f_obj = generate_random_QP(param)

    # Initialize
    x = np.random.randn(n)
    z = np.random.randn(m)
    u = np.random.randn(m)

    # Precompute
    At = A.T
    Ax = A @ x
    AtA = At @ A
    K = Q + rho * AtA
    K_inv = np.linalg.inv(K)

    if verbose:
        print("Initial values:")
        print("x =", x)
        print("z =", z)
        print("u =", u)

    # ADMM
    for k in range(max_iter):
        rhs = -q + rho * At @ (z + u - c)
        x = -K_inv @ rhs
        z = np.maximum(0, -Ax - u + c)
        u = u + Ax - c + z

        # Residuals
        primal_res = np.linalg.norm(Ax + z - c)
        dual_res = np.linalg.norm(rho * At @ (z - np.maximum(0, -Ax - u + c)))

        if verbose:
            print(f"Iter {k:4d}: primal_res = {primal_res:.2e}, dual_res = {dual_res:.2e}, obj = {f_obj(x):.6f}")

        if primal_res < tol and dual_res < tol:
            if verbose:
                print(f"✅ Converged at iteration {k + 1}")
            return k + 1

    return max_iter

import numpy as np

def optimal_theta(param):
    Q, q, A, c = param["Q"], param["q"], param["A"], param["c"]

    Q_inv = np.linalg.inv(Q)
    AQAt = A @ Q_inv @ A.T

    eigvals = np.linalg.eigvalsh(AQAt)
    eigvals = np.sort(eigvals)

    lambda_min = eigvals[0]
    lambda_max = eigvals[-1]

    theta_star = np.sqrt(lambda_min * lambda_max)
    return theta_star

if __name__ == "__main__":
    seed = 42

    np.random.seed(seed)
    param = generate_random_param_of_QP(m=3, n=5)
    admm_QP(param, rho=1, max_iter=5000)
    print(optimal_theta(param))