import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import r2_score
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class BARTModel:
    """
    Prospect Theory + Prelec weighting for BART
    """

    def __init__(self, data_df):
        self.data = data_df.copy()

    # ---------- utility ----------
    @staticmethod
    def utility(x, rho, lam):
        x = np.asarray(x, float)
        u = np.abs(x) ** rho
        u[x < 0] *= -lam
        return u

    @staticmethod
    def prelec_weight(p, delta, eta, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)
        return np.exp(-delta * (-np.log(p)) ** eta)

    # ---------- probability in BART ----------
    @staticmethod
    def compute_p_loss(pump_number, max_pump=64):
        """
        Вероятность взрыва на текущем шаге:
        равномерное распределение break_point ∈ [1, max_pump]
        """
        remaining = max_pump - (pump_number - 1)
        remaining = np.maximum(remaining, 1)
        return 1.0 / remaining

    # ---------- NLL ----------
    def nll(self, params, df):
        rho, lam, delta, eta, beta = params
        eps = 1e-12

        pump = df["pump_number"].to_numpy()
        gain = df["gain_amount"].to_numpy()
        loss = df["loss_amount"].to_numpy()
        choices = df["choice"].to_numpy()
        max_pump = df["max_pump"].to_numpy()

        p_loss = self.compute_p_loss(pump, max_pump)
        p_gain = 1 - p_loss

        u_gain = self.utility(gain, rho, lam)
        u_loss = self.utility(-np.abs(loss), rho, lam)

        w_gain = self.prelec_weight(p_gain, delta, eta)
        w_loss = self.prelec_weight(p_loss, delta, eta)

        EU = w_gain * u_gain + w_loss * u_loss

        p_continue = expit(EU / beta)
        p_continue = np.clip(p_continue, eps, 1 - eps)

        return -np.sum(
            choices * np.log(p_continue) +
            (1 - choices) * np.log(1 - p_continue)
        )

    # ---------- FIT ----------
    def fit(self, df, bounds=None, n_starts=50, n_jobs=8, random_state=None):

        if bounds is None:
            bounds = [
                (0.01, 3.0),   # rho
                (0.01, 10.0),  # lambda
                (0.01, 5.0),   # delta
                (0.01, 3.0),   # eta
                (0.01, 1.0)    # beta
            ]

        rng = np.random.default_rng(random_state)

        def one_start(_):
            x0 = [rng.uniform(b[0], b[1]) for b in bounds]
            try:
                res = minimize(self.nll, x0, args=(df,), bounds=bounds, method="L-BFGS-B")
                if res.success:
                    return res.fun, res.x
            except:
                return np.inf, None
            return np.inf, None

        best_fun, best_x = np.inf, None

        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(one_start, i) for i in range(n_starts)]
            for f in tqdm(as_completed(futures), total=n_starts, leave=False):
                fun, x = f.result()
                if fun < best_fun:
                    best_fun, best_x = fun, x

        return best_x if best_x is not None else [np.nan] * 5

    # ---------- SIMULATE ----------
    def simulate(self, params, df, seed=None):
        rho, lam, delta, eta, beta = params
        rng = np.random.default_rng(seed)

        sim = df.drop(columns=["choice"], errors="ignore").copy()

        pump = sim["pump_number"].to_numpy()
        gain = sim["gain_amount"].to_numpy()
        loss = sim["loss_amount"].to_numpy()
        max_pump = sim["max_pump"].to_numpy()

        p_loss = self.compute_p_loss(pump, max_pump)
        p_gain = 1 - p_loss

        u_gain = self.utility(gain, rho, lam)
        u_loss = self.utility(-np.abs(loss), rho, lam)

        w_gain = self.prelec_weight(p_gain, delta, eta)
        w_loss = self.prelec_weight(p_loss, delta, eta)

        EU = w_gain * u_gain + w_loss * u_loss

        p_continue = expit(EU / beta)

        sim["choice"] = rng.binomial(1, np.clip(p_continue, 1e-12, 1 - 1e-12))

        return sim
    
    def predictive_check(self, df, n_sim=200, seed=42):
        params = self.fit(df)

        rng = np.random.default_rng(seed)

        real = df["choice"].values
        real_mean = real.mean()
        real_var = real_mean * (1 - real_mean)

        sim_means = []

        for _ in range(n_sim):
            sim = self.simulate(params, df, seed=rng.integers(1e9))
            sim_means.append(sim["choice"].mean())

        sim_means = np.array(sim_means)

        p_value = np.mean(sim_means >= real_mean)

        if real_var > 0:
            mse = np.mean((sim_means - real_mean) ** 2)
            r2 = 1 - mse / real_var
        else:
            r2 = 1.0

        return {
            "real_mean": real_mean,
            "sim_mean": sim_means.mean(),
            "p_value": p_value,
            "r2_ppc": r2
        }
    
    def parameter_recovery(self, template_df, n_subjects=20, random_state=123):
        rng = np.random.default_rng(random_state)

        bounds = [(0.01,3),(0.01,10),(0.01,5),(0.01,3),(0.01,1)]

        true_params = np.column_stack([
            rng.uniform(b[0], b[1], n_subjects) for b in bounds
        ])

        def one(i):
            tp = true_params[i]
            sim = self.simulate(tp, template_df, seed=100+i)
            fit = self.fit(sim)
            return np.hstack([tp, fit])

        results = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(one, i) for i in range(n_subjects)]
            for f in tqdm(as_completed(futures), total=n_subjects, leave=False):
                results.append(f.result())

        df = pd.DataFrame(np.vstack(results), columns=[
            "true_rho","true_lambda","true_delta","true_eta","true_beta",
            "fit_rho","fit_lambda","fit_delta","fit_eta","fit_beta"
        ])

        print("\n=== Recovery ===")
        for p in ["rho","lambda","delta","eta","beta"]:
            r = np.corrcoef(df[f"true_{p}"], df[f"fit_{p}"])[0,1]
            print(f"{p}: r={r:.3f}")

        return df
    
    def compute_aic_bic(self, params, df):
        n = len(df)
        k = len(params)
        ll = -self.nll(params, df)

        aic = 2*k - 2*ll
        bic = k*np.log(n) - 2*ll

        return aic, bic, ll