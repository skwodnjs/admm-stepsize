import numpy as np

def generate_random_param_of_QP(m=100, n=150):
    random_matrix = np.random.randn(n, n)
    Q = random_matrix @ random_matrix.T + np.eye(n)         # Q: random symmetric positive definite matrix

    q = np.random.randn(n)

    def generate_full_rank_A(m, n):
        U, _, Vt = np.linalg.svd(np.random.randn(m, n), full_matrices=False)
        S = np.diag(np.linspace(1, 2, min(m, n)))
        return U @ S @ Vt           # Full-rank with SVD

    A = generate_full_rank_A(m, n)  # Full-rank A

    c = np.random.randn(m)

    return {"Q": Q, "q": q, "A": A, "c": c}

def generate_random_QP(param):
    Q, q, A, c = param["Q"], param["q"], param["A"], param["c"]

    def objective(x, z):
        if np.any(z < 0):
            return np.inf  # 지표 함수 I_+(z): z_i < 0 이면 무한대
        return 0.5 * x.T @ Q @ x + q.T @ x  # + 0 when z ≥ 0

    return objective

def admm_QP(param, rho=1, tol=1e-6, max_iter=1000):
    Q, q, A, c = param["Q"], param["q"], param["A"], param["c"]

    f_x = generate_random_QP(param)

    # Precompute
    n = Q.shape[0]
    m = A.shape[0]

    inv_LHS_x = np.linalg.inv(Q + rho * A.T @ A)

    # Initialize
    z = np.random.randn(m)
    b = np.random.randn(m)

    # print("Initial values:")
    # print("x =", x)
    # print("z =", z)
    # print("b =", b)

    # ADMM
    prev_z = np.copy(z)
    for k in range(max_iter):
        # x-update (minimize quadratic)
        rhs = rho * A.T @ (z - b + c) - q
        x = inv_LHS_x @ rhs

        # z-update (projection onto nonnegative orthant)
        z = np.maximum(0, A @ x - c + b)

        # dual update
        b = b + A @ x - c - z

        # Residuals
        r_primal = A @ x - c + z
        r_dual = rho * A.T @ (z - prev_z)
        prev_z = np.copy(z)

        r_norm = np.linalg.norm(r_primal)
        s_norm = np.linalg.norm(r_dual)

        # print(f"[{k + 1:03d}] f(x, z) = {f_x(x, z):.9f}, ||Ax - c + z|| = {np.linalg.norm(r_primal):.9f}")

        if r_norm < tol and s_norm < tol:
            # print(f"✅ Converged at iteration {k + 1}")
            return k + 1

    return max_iter

def optimal_rho(param):
    Q, q, A, c = param["Q"], param["q"], param["A"], param["c"]

    # Compute M = A Q^{-1} A^T
    Q_inv = np.linalg.inv(Q)
    M = A @ Q_inv @ A.T

    # Compute eigenvalues of M
    eigvals = np.linalg.eigvalsh(M)  # Hermitian eigen solver (symmetric)
    lambda_min = np.min(eigvals)
    lambda_max = np.max(eigvals)

    # Compute rho*
    rho_star = 1.0 / np.sqrt(lambda_min * lambda_max)

    return rho_star

if __name__ == "__main__":
    # seed = 42
    seed = np.random.randint(0, 100000)
    np.random.seed(seed)
    print(f"seed: {seed}")
    param = generate_random_param_of_QP(m=3, n=5)

    iter_count = admm_QP(param)
    rho_opt = optimal_rho(param)
    iter_count_opt = admm_QP(param, rho=rho_opt)

    print(f"ADMM 반복 횟수(θ = 1): {iter_count}")
    print(f"최적의 rho: {rho_opt:.6f}")
    print(f"ADMM 반복 횟수(rho = optimal): {iter_count_opt}")