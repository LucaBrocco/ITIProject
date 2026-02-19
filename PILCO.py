"""
PILCO (Probabilistic Inference for Learning Control) for a Discrete Gridworld
==============================================================================
Adapted for a configurable gridworld with continuous-state PILCO machinery:
  - Grid layout defined by a 2D numpy array (shape determines grid size)
  - forbidden_cells: list of (row, col) tuples the agent cannot enter
  - terminal_cells:  list of (row, col) tuples that end the episode (goals)
  - States are continuous 2D vectors [row, col]
  - Actions are discrete: 0=Up, 1=Down, 2=Left, 3=Right
  - GP dynamics model with RBF kernel (one GP per state dimension)
  - Moment matching for analytic uncertainty propagation
  - RBF network policy with softmax output
  - Policy optimized via analytic gradients + L-BFGS-B

References:
  Deisenroth & Rasmussen, "PILCO: A Model-Based and Data-Efficient
  Approach to Policy Search", ICML 2011.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 1. GRIDWORLD ENVIRONMENT
# =============================================================================

class GridWorld:
    """
    Configurable gridworld. State = [row, col] as floats.

    Args:
        grid (np.ndarray): 2D array whose shape defines the grid dimensions.
                           Cell values are not used internally (layout comes
                           from forbidden_cells / terminal_cells), but they
                           can encode domain-specific meaning for the caller.
        forbidden_cells (list[tuple]): List of (row, col) cells the agent
                           cannot enter. Attempting to move there keeps the
                           agent in place and incurs a -5 penalty.
        terminal_cells  (list[tuple]): List of (row, col) goal cells. Reaching
                           any of them ends the episode with +10 reward.

    Actions: 0=Up(-row), 1=Down(+row), 2=Left(-col), 3=Right(+col)
    """
    def __init__(self, grid: np.ndarray,
                 forbidden_cells: list,
                 terminal_cells: list):
        if grid.ndim != 2:
            raise ValueError("grid must be a 2D numpy array.")
        if not terminal_cells:
            raise ValueError("terminal_cells must contain at least one cell.")

        self.grid        = grid
        self.n_rows, self.n_cols = grid.shape
        self.forbidden   = set(map(tuple, forbidden_cells))
        self.terminals   = set(map(tuple, terminal_cells))

        # Validate all supplied cells are within bounds
        for cell in self.forbidden | self.terminals:
            r, c = cell
            if not (0 <= r < self.n_rows and 0 <= c < self.n_cols):
                raise ValueError(f"Cell {cell} is out of bounds for grid shape {grid.shape}.")

        # Pre-compute valid (walkable, non-terminal) starting cells
        self._free_cells = [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if (r, c) not in self.forbidden and (r, c) not in self.terminals
        ]
        if not self._free_cells:
            raise ValueError("No free cells available for the agent to start in.")

        # Terminal cell coordinates as float arrays (for cost computation)
        self.terminal_coords = [np.array(t, dtype=float) for t in self.terminals]

        self.action_deltas = {
            0: np.array([-1., 0.]),
            1: np.array([1.,  0.]),
            2: np.array([0., -1.]),
            3: np.array([0.,  1.]),
        }
        self.n_actions  = 4
        self.state_dim  = 2
        self.state      = None

    # ------------------------------------------------------------------
    def _is_forbidden(self, state: np.ndarray) -> bool:
        """Check if a continuous state rounds to a forbidden cell."""
        return (int(round(state[0])), int(round(state[1]))) in self.forbidden

    def _is_terminal(self, state: np.ndarray) -> bool:
        """Check if a continuous state rounds to a terminal cell."""
        return (int(round(state[0])), int(round(state[1]))) in self.terminals

    # ------------------------------------------------------------------
    def reset(self, start_cell: tuple = None) -> np.ndarray:
        """
        Reset the environment.

        Args:
            start_cell: Optional (row, col) to start from. If None, the agent
                        starts from the first free cell (top-left-most).
        Returns:
            Initial state as a float array [row, col].
        """
        if start_cell is not None:
            r, c = start_cell
            if (r, c) in self.forbidden:
                raise ValueError(f"start_cell {start_cell} is forbidden.")
            if (r, c) in self.terminals:
                raise ValueError(f"start_cell {start_cell} is a terminal cell.")
            self.state = np.array([float(r), float(c)])
        else:
            r, c = self._free_cells[0]
            self.state = np.array([float(r), float(c)])
        return self.state.copy()

    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        Execute one action.

        Forbidden-cell collisions: agent stays in place, reward = -5.
        Out-of-bounds moves:       agent stays in place, reward = -1.
        Normal step:               reward = -1.
        Terminal cell reached:     reward = +10, done = True.

        Returns:
            next_state (np.ndarray), reward (float), done (bool)
        """
        delta      = self.action_deltas[int(action)]
        candidate  = self.state + delta

        # Check grid bounds
        candidate = np.array([
            np.clip(candidate[0], 0, self.n_rows - 1),
            np.clip(candidate[1], 0, self.n_cols - 1),
        ])

        # Check forbidden
        if self._is_forbidden(candidate):
            reward = -100.0
            # state unchanged
        else:
            self.state = candidate
            if self._is_terminal(self.state):
                return self.state.copy(), 10.0, True
            reward = -1.0

        return self.state.copy(), reward, False

    # ------------------------------------------------------------------
    def cost_and_grad(self, state: np.ndarray):
        """
        Differentiable cost for PILCO rollouts.

        Uses a mixture of saturating costs — one per terminal cell — so the
        agent is attracted to ALL terminals simultaneously.  Forbidden cells
        add a repulsive bump cost.

        c_terminal(x) = prod_i [ 1 - exp(-||x - g_i||^2 / (2*l^2)) ]
        c_forbidden(x) = sum_f  w * exp(-||x - f||^2 / (2*lf^2))
        total cost     = c_terminal + c_forbidden   (clamped to [0,1])

        Returns:
            cost (float), dc/dstate (np.ndarray of shape (2,))
        """
        l  = 1.0   # attraction length scale
        lf = 0.8   # repulsion length scale
        w  = 0.5   # forbidden repulsion weight

        # --- Attraction: product of per-terminal saturating costs ---
        # Each factor f_i = 1 - exp(-d_i^2 / 2l^2) in [0,1]
        # Product is 0 when at any terminal, 1 when far from all.
        factors    = []
        df_dstates = []
        for g in self.terminal_coords:
            diff  = state - g
            d2    = np.dot(diff, diff)
            exp_i = np.exp(-0.5 * d2 / l**2)
            f_i   = 1.0 - exp_i
            factors.append(f_i)
            # df_i/dstate = (diff / l^2) * exp_i
            df_dstates.append((diff / l**2) * exp_i)

        c_attract = float(np.prod(factors))
        # Gradient via product rule: d(prod)/dx = sum_i [ df_i/dx * prod_{j!=i} f_j ]
        dc_attract = np.zeros(2)
        for i in range(len(factors)):
            prod_others = 1.0
            for j, fj in enumerate(factors):
                if j != i:
                    prod_others *= fj
            dc_attract += df_dstates[i] * prod_others

        # --- Repulsion: Gaussian bumps at forbidden cells ---
        c_forbid  = 0.0
        dc_forbid = np.zeros(2)
        for fr, fc in self.forbidden:
            f_coord = np.array([float(fr), float(fc)])
            diff    = state - f_coord
            d2      = np.dot(diff, diff)
            bump    = np.exp(-0.5 * d2 / lf**2)
            c_forbid  += w * bump
            dc_forbid += w * (-diff / lf**2) * bump

        cost = np.clip(c_attract + c_forbid, 0.0, 2.0)
        dc   = dc_attract + dc_forbid
        return float(cost), dc


# =============================================================================
# 2. GP DYNAMICS MODEL (RBF Kernel)
# =============================================================================

class GPModel:
    """
    Gaussian Process with RBF (squared exponential) kernel.
    Learns to predict one dimension of the state delta: Δs_d = s'_d - s_d.

    Kernel: k(x,x') = sf^2 * exp(-0.5 * sum((x-x')^2 / l^2)) + sn^2 * delta
    Log-hyperparameters: theta = [log(l_1),...,log(l_D), log(sf), log(sn)]
    """
    def __init__(self, input_dim):
        self.D = input_dim
        # Initialize log-hyperparameters: [log_l x D, log_sf, log_sn]
        self.log_hyp = np.zeros(input_dim + 2)
        self.log_hyp[-1] = np.log(0.1)  # small noise
        self.X_train = None
        self.Y_train = None
        self.alpha = None   # K^{-1} y
        self.L = None       # Cholesky of K

    @property
    def l(self):
        return np.exp(self.log_hyp[:self.D])

    @property
    def sf2(self):
        return np.exp(2 * self.log_hyp[self.D])

    @property
    def sn2(self):
        return np.exp(2 * self.log_hyp[self.D + 1])

    def kernel(self, X1, X2):
        """Compute RBF kernel matrix K(X1, X2). Shape: (n1, n2)."""
        # Scale inputs by length scales
        X1s = X1 / self.l[None, :]
        X2s = X2 / self.l[None, :]
        # Squared distances
        diff = X1s[:, None, :] - X2s[None, :, :]  # (n1, n2, D)
        sq_dist = np.sum(diff**2, axis=2)           # (n1, n2)
        return self.sf2 * np.exp(-0.5 * sq_dist)

    def fit(self, X, Y):
        """Fit GP to data X (n, D), Y (n,). Optimize hyperparameters."""
        self.X_train = X.copy()
        self.Y_train = Y.copy()
        self._optimize_hyperparams()
        self._compute_cholesky()

    def _compute_cholesky(self):
        n = self.X_train.shape[0]
        K = self.kernel(self.X_train, self.X_train)
        K += self.sn2 * np.eye(n)
        K += 1e-6 * np.eye(n)  # jitter
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Y_train))

    def neg_log_likelihood(self, log_hyp):
        """Negative log marginal likelihood for hyperparameter optimization."""
        self.log_hyp = log_hyp
        n = self.X_train.shape[0]
        try:
            K = self.kernel(self.X_train, self.X_train)
            K += np.exp(2 * log_hyp[-1]) * np.eye(n)
            K += 1e-6 * np.eye(n)
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.Y_train))
        nll = (0.5 * self.Y_train @ alpha
               + np.sum(np.log(np.diag(L)))
               + 0.5 * n * np.log(2 * np.pi))
        return nll

    def _optimize_hyperparams(self):
        result = minimize(
            self.neg_log_likelihood,
            self.log_hyp,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        self.log_hyp = result.x
        self._compute_cholesky()

    def predict(self, x_star):
        """
        Predict mean and variance at a single test point x_star (D,).
        Returns: mu (scalar), sigma2 (scalar)
        """
        k_star = self.kernel(self.X_train, x_star[None, :]).flatten()  # (n,)
        mu = k_star @ self.alpha
        v = np.linalg.solve(self.L, k_star)
        sigma2 = self.sf2 - v @ v + self.sn2
        sigma2 = max(sigma2, 1e-8)
        return mu, sigma2

    def predict_with_grad(self, x_star):
        """
        Returns (mu, sigma2, dmu/dx, dsigma2/dx) for gradient computation.
        """
        k_star = self.kernel(self.X_train, x_star[None, :]).flatten()
        mu = k_star @ self.alpha

        v = np.linalg.solve(self.L, k_star)
        sigma2 = max(self.sf2 - v @ v + self.sn2, 1e-8)

        # Gradient of k_star w.r.t. x_star: shape (n, D)
        diff = (self.X_train - x_star[None, :]) / (self.l[None, :]**2)  # (n, D)
        dk_dx = k_star[:, None] * diff  # (n, D)  -- note sign: d/dx* k(xi, x*) = (xi - x*)/l^2 * k

        dmu_dx = dk_dx.T @ self.alpha          # (D,)
        # dsigma2/dx = -2 * (L^{-1} k_star)^T @ L^{-1} dk_dx
        Linv_k = v  # (n,)
        Linv_dk = np.linalg.solve(self.L, dk_dx)  # (n, D)
        dsigma2_dx = -2.0 * Linv_k @ Linv_dk      # (D,)

        return mu, sigma2, dmu_dx, dsigma2_dx


# =============================================================================
# 3. POLICY: RBF NETWORK WITH SOFTMAX
# =============================================================================

class RBFPolicy:
    """
    Policy: pi(s) = softmax(W * phi(s))
    where phi(s) are RBF features centered at valid (non-forbidden) grid cells.
    Actions: 0=Up, 1=Down, 2=Left, 3=Right.

    Parameters: W of shape (n_actions, n_features) -- flattened for optimization.
    """
    def __init__(self, n_actions=4, n_rows=4, n_cols=4, forbidden_cells=None):
        self.n_actions = n_actions
        forbidden = set(map(tuple, forbidden_cells)) if forbidden_cells else set()
        # Centers: all walkable grid cells (forbidden cells excluded)
        centers = [
            [float(r), float(c)]
            for r in range(n_rows)
            for c in range(n_cols)
            if (r, c) not in forbidden
        ]
        self.centers = np.array(centers)          # (n_features, 2)
        self.n_features = len(self.centers)
        self.bandwidth = 1.0
        # Initialize weights
        self.W = np.zeros((n_actions, self.n_features))
        self.n_params = self.W.size

    def get_params(self):
        return self.W.flatten()

    def set_params(self, params):
        self.W = params.reshape(self.n_actions, self.n_features)

    def rbf_features(self, s):
        """Compute RBF features phi(s), shape (n_features,)."""
        diff = self.centers - s[None, :]  # (n_features, 2)
        sq_dist = np.sum(diff**2, axis=1)
        return np.exp(-sq_dist / (2 * self.bandwidth**2))

    def rbf_features_grad(self, s):
        """Returns phi(s) and d(phi)/ds. Shapes: (n_feat,), (n_feat, 2)."""
        diff = self.centers - s[None, :]  # (n_features, 2)
        sq_dist = np.sum(diff**2, axis=1)
        phi = np.exp(-sq_dist / (2 * self.bandwidth**2))
        # d phi_i / d s = phi_i * (centers_i - s) / bandwidth^2  [sign: -diff/bw^2]
        dphi_ds = phi[:, None] * (-diff) / self.bandwidth**2  # wait: d/ds exp(-||c-s||^2/2b^2)
        # = exp(...) * (c_i - s) / b^2   [chain rule: inner deriv of -||c-s||^2/2b^2 w.r.t s = (c-s)/b^2]
        # Actually d/ds (-||c-s||^2) = 2(c-s), so d phi/ds = phi * (c-s)/b^2
        dphi_ds = phi[:, None] * diff / self.bandwidth**2  # (n_feat, 2)  -- diff = c - s
        return phi, dphi_ds

    def softmax(self, logits):
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def action_probs(self, s):
        """Returns action probabilities pi(a|s), shape (n_actions,)."""
        phi = self.rbf_features(s)
        logits = self.W @ phi  # (n_actions,)
        return self.softmax(logits)

    def action_probs_and_grad(self, s):
        """
        Returns pi(a|s) and d(pi)/dW (flattened gradient w.r.t. policy params).
        Also returns d(pi)/ds for moment matching / state gradient.
        """
        phi, dphi_ds = self.rbf_features_grad(s)
        logits = self.W @ phi
        pi = self.softmax(logits)

        # Jacobian d(pi_a)/d(logits_b) = pi_a*(delta_ab - pi_b)
        J_pi_logits = np.diag(pi) - np.outer(pi, pi)  # (n_act, n_act)

        # d(logits_a)/d(W_a,j) = phi_j, so d(pi)/d(W) is (n_act, n_act*n_feat)
        # For gradient w.r.t. W (flattened):
        # d(pi_a)/d(W_b,j) = J_pi_logits[a,b] * phi[j]
        dpi_dW = np.einsum('ab,j->abj', J_pi_logits, phi).reshape(self.n_actions, -1)

        # d(pi)/d(s): chain through logits -> phi -> s
        # d(logits_a)/d(s) = W_a @ d(phi)/d(s) = (n_act, 2)
        dlogits_ds = self.W @ dphi_ds  # (n_act, 2)
        dpi_ds = J_pi_logits @ dlogits_ds  # (n_act, 2)

        return pi, dpi_dW, dpi_ds

    def sample_action(self, s):
        """Sample action from policy."""
        probs = self.action_probs(s)
        return np.random.choice(self.n_actions, p=probs)

    def greedy_action(self, s):
        """Return greedy (most probable) action."""
        probs = self.action_probs(s)
        return np.argmax(probs)


# =============================================================================
# 4. MOMENT MATCHING: Propagate Gaussian belief through GP + policy
# =============================================================================

def propagate_moments(mu_s, sigma2_s, gp_models, policy, env):
    """
    Given a Gaussian belief p(s) = N(mu_s, diag(sigma2_s)) over state s,
    propagate it through the policy and GP dynamics to get the next-state
    belief and expected cost.

    This is a simplified (diagonal covariance) moment matching approximation.

    Args:
        mu_s:    (state_dim,) mean of current state belief
        sigma2_s: (state_dim,) variance of current state belief (diagonal)
        gp_models: list of GPModel, one per state dimension
        policy:  RBFPolicy
        env:     GridWorld (for cost function)

    Returns:
        mu_next:    (state_dim,) mean of next state
        sigma2_next: (state_dim,) variance of next state
        E_cost:     expected cost at this step
        grads: dict with gradient info for policy optimization
    """
    state_dim = len(mu_s)
    n_actions = policy.n_actions

    # --- Step 1: Compute expected action (soft, using mean state) ---
    # For simplicity we evaluate policy at the mean state (linearization approx)
    pi, dpi_dW, dpi_ds = policy.action_probs_and_grad(mu_s)

    # Action deltas (deterministic given action)
    action_deltas = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])

    # --- Step 2: Expected delta via GP predictions at mean state ---
    # Input to GP: [s, a_one_hot] -- but for simplicity, we use [s] directly
    # and weight GP predictions by action probabilities.
    # We create augmented input x = [mu_s] (GPs trained on [s, a_one_hot])
    # Here we use a single GP per dimension trained on (s, a) -> delta_s

    # For the simplified version: we treat the expected action as soft,
    # and use the GP prediction at (mu_s, expected_action_embedding).
    # The GP input is [s_row, s_col, a0, a1, a2, a3] where a is one-hot.

    E_delta = np.zeros(state_dim)
    V_delta = np.zeros(state_dim)
    dE_delta_dpi = np.zeros((state_dim, n_actions))  # gradient of E[delta] w.r.t. pi
    dE_delta_ds  = np.zeros((state_dim, state_dim))  # gradient of E[delta] w.r.t. mu_s

    for d in range(state_dim):
        gp = gp_models[d]
        for a in range(n_actions):
            # Build GP input: [s_row, s_col, one_hot(a)]
            a_oh = np.zeros(n_actions)
            a_oh[a] = 1.0
            x_in = np.concatenate([mu_s, a_oh])
            mu_gp, var_gp, dmu_dx, _ = gp.predict_with_grad(x_in)

            E_delta[d] += pi[a] * mu_gp
            V_delta[d] += pi[a] * (var_gp + mu_gp**2)
            dE_delta_dpi[d, a] = mu_gp
            dE_delta_ds[d]  += pi[a] * dmu_dx[:state_dim]  # gradient w.r.t. s part

        V_delta[d] -= E_delta[d]**2   # Var[delta] = E[delta^2] - E[delta]^2
        V_delta[d] = max(V_delta[d], 1e-6)

    # --- Step 3: Next state moments ---
    mu_next    = mu_s + E_delta
    sigma2_next = sigma2_s + V_delta

    # Clip to grid
    mu_next = np.clip(mu_next, [0, 0], [env.n_rows - 1, env.n_cols - 1])

    # --- Step 4: Expected cost ---
    E_cost, dc_ds = env.cost_and_grad(mu_next)

    # --- Step 5: Gradient of E_cost w.r.t. policy params W ---
    # Chain: E_cost -> mu_next -> E_delta -> pi -> W
    # dc/dW = dc/dmu_next * dmu_next/dE_delta * dE_delta/dpi * dpi/dW
    #       = dc_ds * 1 * dE_delta_dpi * dpi_dW  (summed appropriately)
    # shape notes:
    #   dc_ds: (D,)
    #   dE_delta_dpi: (D, n_act)
    #   dpi_dW: (n_act, n_params)
    dE_cost_dpi = dc_ds @ dE_delta_dpi   # (n_act,)
    dE_cost_dW  = dE_cost_dpi @ dpi_dW  # (n_params,)

    # Also through dpi/ds and dE_delta/ds
    dE_cost_ds_via_delta = dc_ds @ dE_delta_ds   # (D,)
    dE_cost_dW_via_s = (dc_ds @ dE_delta_dpi) @ (dpi_ds @ np.eye(state_dim))

    grads = {
        'dE_cost_dW': dE_cost_dW,
        'dc_ds': dc_ds,
        'dE_delta_ds': dE_delta_ds,
        'dpi_dW': dpi_dW,
        'dpi_ds': dpi_ds,
    }

    return mu_next, sigma2_next, E_cost, grads


def compute_q(s, a, gp_models, policy, env, T=15, gamma=0.95):
    """
    Estimate Q^pi(s, a) using PILCO's GP dynamics and moment matching.

    Definition:
        Q^pi(s, a) = E[ sum_{t=0}^{T-1} gamma^t * c(s_t)
                       | s_0 = s, a_0 = a, a_{t>=1} ~ pi ]

    The first transition is forced (action a is taken deterministically),
    then the policy pi drives all subsequent steps via moment matching.

    Args:
        s         (np.ndarray): Starting state [row, col], shape (2,).
        a         (int):        Forced action at t=0 (0=Up,1=Down,2=Left,3=Right).
        gp_models (list):       Fitted GP models, one per state dimension.
        policy    (RBFPolicy):  Current policy.
        env       (GridWorld):  Environment (used for cost and clipping).
        T         (int):        Rollout horizon.
        gamma     (float):      Discount factor.

    Returns:
        q_value   (float):      Discounted expected cumulative cost under Q^pi(s,a).
    """
    state_dim = env.state_dim
    n_actions  = env.n_actions
    sigma2_s   = 0.01 * np.ones(state_dim)   # small initial uncertainty

    # ------------------------------------------------------------------
    # Step t=0: forced action a -- query GP with one-hot encoding of a
    # ------------------------------------------------------------------
    a_oh = np.zeros(n_actions)
    a_oh[a] = 1.0

    mu_delta  = np.zeros(state_dim)
    var_delta = np.zeros(state_dim)
    for d in range(state_dim):
        x_in = np.concatenate([s, a_oh])
        mu_gp, var_gp = gp_models[d].predict(x_in)
        mu_delta[d]  = mu_gp
        var_delta[d] = var_gp

    mu_s1     = np.clip(s + mu_delta,
                        [0, 0], [env.n_rows - 1, env.n_cols - 1])
    sigma2_s1 = sigma2_s + var_delta

    # Cost at the state reached after the forced action
    c0, _ = env.cost_and_grad(mu_s1)
    q_value = (gamma ** 0) * c0

    # ------------------------------------------------------------------
    # Steps t=1 ... T-1: policy-driven moment matching
    # ------------------------------------------------------------------
    mu_s     = mu_s1
    sigma2_s = sigma2_s1
    for t in range(1, T):
        mu_s, sigma2_s, E_cost, _ = propagate_moments(
            mu_s, sigma2_s, gp_models, policy, env
        )
        q_value += (gamma ** t) * E_cost

    return q_value
# =============================================================================
# 5. POLICY OPTIMIZATION VIA ROLLOUT
# =============================================================================

def rollout_and_cost(params, policy, gp_models, env, T=15, gamma=0.99):
    """
    Simulate T-step rollout using GP dynamics and moment matching.
    Returns total expected cost and gradient w.r.t. policy params.
    """
    policy.set_params(params)

    mu_s = env.reset().copy()
    sigma2_s = 0.01 * np.ones(env.state_dim)

    total_cost = 0.0
    total_grad = np.zeros_like(params)

    # We propagate gradients backwards through time (simplified: additive)
    for t in range(T):
        mu_next, sigma2_next, E_cost, grads = propagate_moments(
            mu_s, sigma2_s, gp_models, policy, env
        )
        w = gamma**t
        total_cost += w * E_cost
        total_grad += w * grads['dE_cost_dW']

        mu_s = mu_next
        sigma2_s = sigma2_next

    return total_cost, total_grad


def optimize_policy(policy, gp_models, env, T=15, gamma=0.99, n_restarts=3):
    """Optimize policy parameters using L-BFGS-B."""
    best_cost = np.inf
    best_params = policy.get_params().copy()

    for restart in range(n_restarts):
        if restart == 0:
            x0 = policy.get_params().copy()
        else:
            x0 = np.random.randn(policy.n_params) * 0.1

        def objective(params):
            cost, grad = rollout_and_cost(params, policy, gp_models, env, T, gamma)
            return cost, grad

        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 50, 'ftol': 1e-5, 'disp': False}
        )

        if result.fun < best_cost:
            best_cost = result.fun
            best_params = result.x.copy()

    policy.set_params(best_params)
    print(f"  Policy optimized. Expected cost: {best_cost:.4f}")
    return best_cost


# =============================================================================
# 6. DATA COLLECTION
# =============================================================================

def _bfs_distances(env):
    """
    Compute BFS shortest-path distances from every free cell to the nearest
    terminal cell, ignoring forbidden cells.  Returns a dict {(r,c): dist}.
    Cells unreachable from any terminal get distance = infinity.
    """
    from collections import deque
    dist = {}
    queue = deque()
    for t in env.terminals:
        dist[t] = 0
        queue.append(t)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        r, c = queue.popleft()
        for dr, dc in deltas:
            nb = (r + dr, c + dc)
            nr, nc = nb
            if (0 <= nr < env.n_rows and 0 <= nc < env.n_cols
                    and nb not in env.forbidden
                    and nb not in dist):
                dist[nb] = dist[(r, c)] + 1
                queue.append(nb)
    return dist


def _action_toward_terminal(env, state, bfs_dist, epsilon=0.2):
    """
    Greedy BFS-guided action: pick the neighbour with smallest BFS distance.
    With probability epsilon, choose randomly (for exploration).
    """
    if np.random.rand() < epsilon:
        return np.random.randint(env.n_actions)
    r, c = int(round(state[0])), int(round(state[1]))
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    best_a, best_d = None, float('inf')
    for a, (dr, dc) in enumerate(deltas):
        nb = (r + dr, c + dc)
        nr, nc = nb
        if not (0 <= nr < env.n_rows and 0 <= nc < env.n_cols):
            continue
        if nb in env.forbidden:
            continue
        d = bfs_dist.get(nb, float('inf'))
        if d < best_d:
            best_d, best_a = d, a
    return best_a if best_a is not None else np.random.randint(env.n_actions)


def collect_data_random(env, n_episodes=5, max_steps=20):
    """
    Collect transition data using a BFS-guided epsilon-greedy exploration
    policy so that terminal cells are reliably visited even in mazes.
    """
    bfs_dist = _bfs_distances(env)
    X_data = []  # inputs: [s_row, s_col, a_oh0, a_oh1, a_oh2, a_oh3]
    Y_data = []  # targets: delta_s per dimension
    total_reward = 0

    for _ in range(n_episodes):
        s = env.reset()
        for _ in range(max_steps):
            a = _action_toward_terminal(env, s, bfs_dist, epsilon=0.3)
            s_next, r, done = env.step(a)
            a_oh = np.zeros(env.n_actions)
            a_oh[a] = 1.0
            x = np.concatenate([s, a_oh])
            y = s_next - s
            X_data.append(x)
            Y_data.append(y)
            total_reward += r
            s = s_next
            if done:
                break

    return np.array(X_data), np.array(Y_data), total_reward


def collect_data_policy(env, policy, n_episodes=3, max_steps=30):
    """Collect transition data using current policy."""
    X_data = []
    Y_data = []
    total_reward = 0

    for _ in range(n_episodes):
        s = env.reset()
        for _ in range(max_steps):
            a = policy.sample_action(s)
            s_next, r, done = env.step(a)
            a_oh = np.zeros(env.n_actions)
            a_oh[a] = 1.0
            x = np.concatenate([s, a_oh])
            y = s_next - s
            X_data.append(x)
            Y_data.append(y)
            total_reward += r
            s = s_next
            if done:
                break

    return np.array(X_data), np.array(Y_data), total_reward


# =============================================================================
# 7. GP FITTING
# =============================================================================

def fit_gp_models(X, Y, state_dim=2):
    """
    Fit one GP per state dimension.
    X: (n, input_dim), Y: (n, state_dim) -- Y contains deltas.
    Returns list of fitted GPModel.
    """
    input_dim = X.shape[1]
    gp_models = []
    for d in range(state_dim):
        print(f"  Fitting GP for dimension {d}...")
        gp = GPModel(input_dim=input_dim)
        gp.fit(X, Y[:, d])
        gp_models.append(gp)
    return gp_models


# =============================================================================
# 8. EVALUATION
# =============================================================================

def evaluate_policy(env, policy, n_episodes=10, max_steps=30):
    """
    Run greedy policy and report success rate, avg reward, and avg steps.

    Returns:
        success_rate (float): fraction of episodes that reached a terminal cell.
        avg_reward   (float): mean episode reward across all episodes.
        avg_steps    (float): mean number of steps taken in successful episodes
                              (None if no episode succeeded).
    """
    successes = 0
    total_reward = 0
    total_steps_successful = 0
    for _ in range(n_episodes):
        s = env.reset()
        ep_reward = 0
        for step in range(max_steps):
            a = policy.greedy_action(s)
            s, r, done = env.step(a)
            ep_reward += r
            if done:
                successes += 1
                total_steps_successful += step + 1  # +1 because step is 0-indexed
                break
        total_reward += ep_reward
    avg_steps = (total_steps_successful / successes) if successes > 0 else None
    return successes / n_episodes, total_reward / n_episodes, avg_steps


def visualize_policy(env, policy):
    """
    Print a grid showing the greedy action at each free cell.
    Legend:  arrows = greedy action   X = forbidden   T = terminal
    """
    arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    print("\n  Greedy policy (T=terminal, X=forbidden):")
    for r in range(env.n_rows):
        row_str = "  "
        for c in range(env.n_cols):
            if (r, c) in env.terminals:
                row_str += ' T '
            elif (r, c) in env.forbidden:
                row_str += ' X '
            else:
                s = np.array([float(r), float(c)])
                a = policy.greedy_action(s)
                row_str += f' {arrows[a]} '
        print(row_str)
    print()


# =============================================================================
# 9. POLICY WARM-START (BFS-based)
# =============================================================================

def _warm_start_policy(policy, env):
    """
    Initialise policy weights so that at each cell the action pointing toward
    the nearest terminal (by BFS) gets a higher logit.  This gives the
    gradient-based optimiser a head-start near a reasonable solution.
    """
    bfs_dist = _bfs_distances(env)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    W = np.zeros_like(policy.W)

    for idx, center in enumerate(policy.centers):
        r, c = int(round(center[0])), int(round(center[1]))
        for a, (dr, dc) in enumerate(deltas):
            nb = (r + dr, c + dc)
            nr, nc = nb
            if not (0 <= nr < env.n_rows and 0 <= nc < env.n_cols):
                continue
            if nb in env.forbidden:
                continue
            d_nb  = bfs_dist.get(nb, float('inf'))
            d_cur = bfs_dist.get((r, c), float('inf'))
            if d_nb < d_cur:
                W[a, idx] += 2.0   # prefer actions that reduce BFS distance

    policy.W = W


# =============================================================================
# 10. MAIN PILCO LOOP
# =============================================================================

def main():
    np.random.seed(42)
    print("=" * 60)
    print("  PILCO for Custom Gridworld")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Define your gridworld here:
    #   grid          -- 2D numpy array (shape = grid dimensions; values unused)
    #   forbidden_cells -- cells the agent cannot enter (walls / obstacles)
    #   terminal_cells  -- goal cells that end the episode
    #
    # Example: 5x5 grid with a wall column and two goals
    #
    #   . . . . .
    #   . X X . .
    #   . X . . .
    #   . X . . T
    #   . . . . T
    # -----------------------------------------------------------------------
    grid = np.zeros((5, 5))
    forbidden_cells = [(1, 1), (1, 2), (2, 1), (3, 1)]
    terminal_cells  = [(3, 4), (4, 4)]

    env = GridWorld(
        grid=grid,
        forbidden_cells=forbidden_cells,
        terminal_cells=terminal_cells,
    )
    policy = RBFPolicy(
        n_actions=env.n_actions,
        n_rows=env.n_rows,
        n_cols=env.n_cols,
        forbidden_cells=forbidden_cells,
    )

    # Warm-start policy weights using BFS distances so the optimizer
    # starts near a sensible solution rather than a flat zero.
    _warm_start_policy(policy, env)

    # --- Phase 1: Initial random data collection ---
    print("\n[Phase 0] Collecting initial random data...")
    X_all, Y_all, _ = collect_data_random(env, n_episodes=15, max_steps=30)
    print(f"  Collected {len(X_all)} transitions.")

    n_pilco_iterations = 5

    for iteration in range(n_pilco_iterations):
        print(f"\n{'=' * 60}")
        print(f"  PILCO Iteration {iteration + 1}/{n_pilco_iterations}")
        print(f"{'=' * 60}")

        # --- Step 1: Fit GP dynamics model ---
        print("\n[Step 1] Fitting GP dynamics models...")
        gp_models = fit_gp_models(X_all, Y_all, state_dim=env.state_dim)

        # --- Step 2: Optimize policy ---
        print("\n[Step 2] Optimizing policy...")
        optimize_policy(policy, gp_models, env, T=15, gamma=0.95, n_restarts=2)

        # --- Step 3: Evaluate current policy ---
        print("\n[Step 3] Evaluating policy on real environment...")
        success_rate, avg_reward = evaluate_policy(env, policy, n_episodes=20)
        print(f"  Success rate: {success_rate * 100:.1f}%  |  Avg reward: {avg_reward:.2f}")
        visualize_policy(env, policy)

        # --- Step 4: Collect new data with current policy ---
        print("[Step 4] Collecting new data with current policy...")
        X_new, Y_new, ep_reward = collect_data_policy(env, policy, n_episodes=5)
        print(f"  Collected {len(X_new)} new transitions. Episode reward: {ep_reward:.2f}")

        # Aggregate data
        X_all = np.vstack([X_all, X_new])
        Y_all = np.vstack([Y_all, Y_new])
        print(f"  Total dataset size: {len(X_all)} transitions.")

        if success_rate >= 0.9:
            print("\n  *** Policy converged! Success rate >= 90% ***")
            break

    print("\n" + "=" * 60)
    print("  Final Evaluation")
    print("=" * 60)
    success_rate, avg_reward = evaluate_policy(env, policy, n_episodes=50)
    print(f"  Final success rate: {success_rate * 100:.1f}%  |  Avg reward: {avg_reward:.2f}")
    visualize_policy(env, policy)


if __name__ == "__main__":
    main()
