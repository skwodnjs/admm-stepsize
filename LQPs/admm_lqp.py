def generate_random_param_of_LQP(m=100, n=150):

    # µ in R: regularization parameter (positive scalar)
    mu = np.random.uniform(0.1, 10.0)

    # A in R^{m×n}: encoding matrix
    A = np.random.randn(m, n)

    # f in R^m: input vector
    f = np.random.randn(m)

    # L in R^{n×n}: regularization matrix (e.g., finite difference operator)
    # Here: identity or random symmetric positive definite matrix
    random_matrix = np.random.randn(n, n)
    L = random_matrix @ random_matrix.T + np.eye(n)  # makes it symmetric positive definite

    return {"mu": mu, "A": A, "f": f, "L": L}

def generate_random_LQP(param):
    mu, A, f, L = param["mu"], param["A"], param["f"], param["L"]

    def objective(u, w):
        residual1 = A @ u - f
        residual2 = L @ w
        return ((mu / 2) * np.dot(residual1, residual1)
                + 0.5 * np.dot(residual2, residual2))

    return objective

def admm_LQP(param, theta=0.5, tol=1e-8, max_iter=1000, verbose=True):
    mu, A, f, L = param["mu"], param["A"], param["f"], param["L"]

    f_u = generate_random_LQP(param)

    # Precompute
    n = A.shape[1]

    I = np.eye(n)
    AtA = A.T @ A
    Atf = A.T @ f
    LtL = L.T @ L

    inv_LHS_w = np.linalg.inv(LtL + theta * I)
    inv_LHS_u = np.linalg.inv(mu * AtA + theta * I)

    # Initialize
    u = np.random.randn(n)
    w = np.random.randn(n)
    b = np.random.randn(n)

    if verbose:
        print("Initial values:")
        print("u =", u)
        print("w =", w)
        print("b =", b)

    # ADMM
    prev_w = np.copy(w)
    for k in range(max_iter):
        w = inv_LHS_w @ (theta * u + theta * b)
        u = inv_LHS_u @ (theta * w - theta * b + mu * Atf)
        b = b + u - w

        # Residuals
        r_primal = w - u
        r_dual = theta * (w - prev_w)
        prev_w = np.copy(w)

        r_norm = np.linalg.norm(r_primal)
        s_norm = np.linalg.norm(r_dual)

        if verbose:
            print(f"[{k + 1:03d}] f(u, w) = {f_u(u, w):.9f}, ||w - u|| = {np.linalg.norm(w - u):.9f}")

        if r_norm < tol and s_norm < tol:
            if verbose:
                print(f"✅ Converged at iteration {k + 1}")
            return k + 1

    return max_iter

import numpy as np

def optimal_theta(param, max_iter=1000, lr=1, tol=1e-4):
    mu, A, f, L = param["mu"], param["A"], param["f"], param["L"]
    n = A.shape[1]
    I = np.eye(n)

    def compute_Q(theta):
        AtA = A.T @ A
        LtL = L.T @ L
        term1 = theta * np.linalg.inv(mu * AtA + theta * I)
        term2 = np.linalg.inv(LtL + theta * I) @ (theta * I - mu * AtA)
        return term1 @ (term2 - I)

    def spectral_radius(Q_mat):
        eigvals = np.linalg.eigvals(Q_mat)
        return np.max(np.abs(eigvals))

    # Gradient descent for minimizing the smallest eigenvalue of Q(θ)
    theta = 1.0

    for k in range(max_iter):
        eps = 1e-5
        Q_plus = compute_Q(theta + eps)
        Q_minus = compute_Q(theta - eps)

        λ_plus = spectral_radius(Q_plus)
        λ_minus = spectral_radius(Q_minus)

        grad = (λ_plus - λ_minus) / (2 * eps)
        theta_new = theta - lr * grad

        # Ensure theta remains positive
        if theta_new < 1e-4:
            theta_new = 1e-4

        if abs(theta_new - theta) < tol:
            break

        theta = theta_new

    print(f"[✔] Converged in {k + 1} iterations")
    return theta

if __name__ == "__main__":
    seed = 42

    np.random.seed(seed)
    param = generate_random_param_of_LQP(m=3, n=5)
    admm_LQP(param)
    print(optimal_theta(param))