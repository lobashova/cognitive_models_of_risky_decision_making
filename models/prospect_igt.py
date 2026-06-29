import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import r2_score


class IGTModelPT:
    """
    Prospect Theory + Prelec weighting for Iowa Gambling Task

    Параметры:
    ----------
    rho (ρ): curvature of utility
    lam (λ): loss aversion
    delta (δ): Prelec elevation
    eta (η): Prelec shape
    beta (β): inverse temperature (decision noise)
    """

    def __init__(self, data_df):
        self.data = data_df.copy()

        # фиксированные параметры колод (IGT дизайн)
        self.deck_info = {
            0: {"gain": 100, "loss": 250, "p_loss": 0.5},  # A
            1: {"gain": 100, "loss": 625, "p_loss": 0.2},  # B
            2: {"gain": 50,  "loss": 50,  "p_loss": 0.5},  # C
            3: {"gain": 50,  "loss": 125, "p_loss": 0.2},  # D
        }

    # ---------- utility ----------
    @staticmethod
    def utility(x, rho, lam):
        x = np.asarray(x, float)

        u = np.empty_like(x, dtype=float)

        pos = x >= 0
        neg = ~pos

        u[pos] = np.power(x[pos], rho)
        u[neg] = -lam * np.power(-x[neg], rho)

        return u

    # ---------- Prelec ----------
    @staticmethod
    def prelec_weight(p, delta, eta, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)
        return np.exp(-delta * (-np.log(p)) ** eta)

    # ---------- expected utility for all decks ----------
    def compute_EU_all_decks(self, rho, lam, delta, eta):
        EU = []

        for d in range(4):
            g = self.deck_info[d]["gain"]
            l = self.deck_info[d]["loss"]
            p_loss = self.deck_info[d]["p_loss"]
            p_gain = 1 - p_loss

            u_gain = self.utility(g, rho, lam)
            u_loss = self.utility(-l, rho, lam)

            w_gain = self.prelec_weight(p_gain, delta, eta)
            w_loss = self.prelec_weight(p_loss, delta, eta)

            EU.append(w_gain * u_gain + w_loss * u_loss)

        return np.array(EU)

    # ---------- NLL ----------
    def nll(self, params, df):
        rho, lam, delta, eta, beta = params
        eps = 1e-12

        choices = df["deck_num"].to_numpy()

        EU = self.compute_EU_all_decks(rho, lam, delta, eta)

        # softmax probabilities
        probs = softmax(EU / beta)

        probs = np.clip(probs, eps, 1 - eps)

        loglik = 0.0
        for c in choices:
            loglik += np.log(probs[c])

        return -loglik

    # ---------- fit ----------
    def fit(self, df, bounds=None, n_starts=30, n_jobs=8, random_state=None):
        if bounds is None:
            bounds = [
                (0.01, 3.0),   # rho
                (0.01, 10.0),  # lambda
                (0.01, 5.0),   # delta
                (0.01, 3.0),   # eta
                (0.01, 1.0)    # beta (больше чем в CCT)
            ]

        rng = np.random.default_rng(random_state)

        def one_start(_):
            x0 = [rng.uniform(b[0], b[1]) for b in bounds]
            try:
                res = minimize(self.nll, x0, args=(df,), bounds=bounds, method="L-BFGS-B")
                if res.success:
                    return res.fun, res.x
            except:
                pass
            return np.inf, None

        best_fun, best_x = np.inf, None

        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(one_start, i) for i in range(n_starts)]
            for f in tqdm(as_completed(futures), total=n_starts, desc="IGT fit", leave=False):
                fun, x = f.result()
                if fun < best_fun:
                    best_fun, best_x = fun, x

        return best_x if best_x is not None else [np.nan]*5

    # ---------- simulate ----------
    def simulate(self, params, df, n_trials=110, seed=None):
        rho, lam, delta, eta, beta = params
        rng = np.random.default_rng(seed)

        EU = self.compute_EU_all_decks(rho, lam, delta, eta)
        probs = softmax(EU / beta)

        sim_choices = rng.choice(4, size=n_trials, p=probs)

        return sim_choices
   

    def predictive_check_with_r2(self, df, n_sim=200, seed=42):
        params = self.fit(df)
        rng = np.random.default_rng(seed)

        real = df["deck_num"].values
        real_good = np.isin(real, [2,3]).astype(float)  # 1 для "хороших", 0 для "плохих"

        sim_matrix = np.zeros((n_sim, len(real)))

        for i in range(n_sim):
            sim = self.simulate(params, df, n_trials=len(real), seed=rng.integers(1e9))
            sim_good = np.isin(sim, [2,3]).astype(float)
            sim_matrix[i] = sim_good

        sim_mean = sim_matrix.mean(axis=0)  # среднее по симуляциям для каждого trial

        # R² между реальными и средними симуляциями
        r2 = r2_score(real_good, sim_mean)

        # средняя доля "хороших" колод
        return {
            "real_good_rate": real_good.mean(),
            "sim_good_rate": sim_mean.mean(),
            "p_value": np.mean(sim_mean >= real_good.mean()),
            "r2": r2
        }


    def parameter_recovery(self, df, n_subjects=20, random_state=123):
        """
        Parameter recovery, используя разные шаблоны для разных субъектов.
        df должен содержать реальные данные всех субъектов.
        """
        rng = np.random.default_rng(random_state)

        # границы параметров
        bounds = [(0.01,3),(0.01,10),(0.01,5),(0.01,3),(0.01,1)]

        # генерируем истинные параметры для каждого субъекта
        true_params = np.column_stack([
            rng.uniform(b[0], b[1], n_subjects) for b in bounds
        ])

        results = []

        # получаем уникальных пользователей
        uids = df["user_id"].unique()[:n_subjects]

        for i, uid in tqdm(enumerate(uids), desc="Recovery", total=n_subjects):
            tp = true_params[i]

            # шаблон этого субъекта
            subj_df = df[df["user_id"] == uid].copy().reset_index(drop=True)

            # если trials больше 110, берем первые 110
            base_df = subj_df.iloc[:110].copy().reset_index(drop=True)

            # --- симуляция 110 trials ---
            sim_choices = self.simulate(tp, base_df, n_trials=110, seed=100+i)

            sim_df = base_df.copy()
            sim_df["deck_num"] = sim_choices

            # --- подгонка ---
            fit = self.fit(sim_df)

            results.append(np.hstack([tp, fit]))

        df_res = pd.DataFrame(results, columns=[
            "true_rho","true_lambda","true_delta","true_eta","true_beta",
            "fit_rho","fit_lambda","fit_delta","fit_eta","fit_beta"
        ])

        print("\n=== Parameter Recovery (per subject template) ===")
        for p in ["rho","lambda","delta","eta","beta"]:
            r = np.corrcoef(df_res[f"true_{p}"], df_res[f"fit_{p}"])[0,1]
            print(f"{p}: r = {r:.3f}")

        return df_res

    # ---------- AIC/BIC ----------
    def compute_aic_bic(self, params, df):
        n = len(df)
        k = len(params)
        ll = -self.nll(params, df)

        aic = 2*k - 2*ll
        bic = k*np.log(n) - 2*ll

        return aic, bic, ll