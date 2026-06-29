import pymc as pm
import pytensor.tensor as pt
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import lgamma
from scipy.stats import pearsonr, mode
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class HaffkeColdModel_HBA:
    """
    Haffke Model 3 адаптированная для CCT-Cold (HBA).
    Оптимизирована для суперкомпьютера (HPC) и стандартов Q1/Q2.
    """
    def __init__(self, df, scale_factor=100.0):
        self.df = df.copy().reset_index(drop=True)
        self.scale_factor = scale_factor
        self.n_trials = len(self.df)
        self._prepare_data()

    def _prepare_data(self):
        """Предрасчет матриц для мгновенной работы PyTensor."""
        self.user_ids = self.df['user_id'].unique()
        self.n_users = len(self.user_ids)
        self.user2idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.subj_idx = self.df['user_id'].map(self.user2idx).values

        self.max_opts = 33 # От 0 до 32 карт
        
        self.P_succ_mat = np.zeros((self.n_trials, self.max_opts))
        self.Gain_mat = np.zeros((self.n_trials, self.max_opts))
        self.Loss_mat = np.zeros((self.n_trials, self.max_opts))
        self.Mask_mat = np.zeros((self.n_trials, self.max_opts)) 
        self.choice_idx = np.zeros(self.n_trials, dtype=int)

        for t in range(self.n_trials):
            row = self.df.iloc[t]
            c = int(row['cards_left'])
            g = int(row['gains_left'])
            
            # Масштабирование
            gain_amt = float(row['gain_amount']) / self.scale_factor
            loss_amt = abs(float(row['loss_amount'])) / self.scale_factor
            
            opts = row['options'] if isinstance(row['options'], (list, np.ndarray)) else eval(row['options'])
            self.choice_idx[t] = int(row['choice'])
            
            for m in opts:
                if m <= c:
                    self.Mask_mat[t, m] = 1.0
                    self.Gain_mat[t, m] = m * gain_amt
                    self.Loss_mat[t, m] = loss_amt
                    
                    if m == 0:
                        p = 1.0
                    elif m > g:
                        p = 0.0
                    else:
                        logp = lgamma(g+1) - lgamma(m+1) - lgamma(g-m+1) \
                             - (lgamma(c+1) - lgamma(m+1) - lgamma(c-m+1))
                        p = np.exp(logp)
                    self.P_succ_mat[t, m] = p

    def build_model(self):
        coords = {"subject": range(self.n_users), "obs_idx": range(self.n_trials)}
        with pm.Model(coords=coords) as model:
            # 1. Групповые гиперпараметры
            mu_alpha = pm.Normal('mu_alpha', mu=0.0, sigma=1.5)
            mu_lam   = pm.Normal('mu_lam', mu=0.0, sigma=1.5)
            mu_delta = pm.Normal('mu_delta', mu=0.0, sigma=1.5)
            mu_eta   = pm.Normal('mu_eta', mu=0.0, sigma=1.5)
            mu_theta = pm.Normal('mu_theta', mu=0.0, sigma=1.5)
            mu_eps   = pm.Normal('mu_eps', mu=-2.0, sigma=1.5) # Ближе к 0 для lapse

            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1.0)
            sigma_lam   = pm.HalfNormal('sigma_lam', sigma=1.0)
            sigma_delta = pm.HalfNormal('sigma_delta', sigma=1.0)
            sigma_eta   = pm.HalfNormal('sigma_eta', sigma=1.0)
            sigma_theta = pm.HalfNormal('sigma_theta', sigma=1.0)
            sigma_eps   = pm.HalfNormal('sigma_eps', sigma=1.0)

            # 2. Индивидуальные сдвиги (Non-centered)
            z_alpha = pm.Normal('z_alpha', mu=0, sigma=1, dims='subject')
            z_lam   = pm.Normal('z_lam', mu=0, sigma=1, dims='subject')
            z_delta = pm.Normal('z_delta', mu=0, sigma=1, dims='subject')
            z_eta   = pm.Normal('z_eta', mu=0, sigma=1, dims='subject')
            z_theta = pm.Normal('z_theta', mu=0, sigma=1, dims='subject')
            z_eps   = pm.Normal('z_eps', mu=0, sigma=1, dims='subject')

            # 3. Трансформация в строгие границы (invlogit)
            # alpha (0, 3), lam (0, 10), delta (0, 3), eta (0, 3), theta (0, 10), eps (0, 0.2)
            alpha = pm.Deterministic('alpha', 3.0 * pm.math.invlogit(mu_alpha + z_alpha * sigma_alpha), dims='subject')
            lam   = pm.Deterministic('lambda', 10.0 * pm.math.invlogit(mu_lam + z_lam * sigma_lam), dims='subject')
            delta = pm.Deterministic('delta', 3.0 * pm.math.invlogit(mu_delta + z_delta * sigma_delta), dims='subject')
            eta   = pm.Deterministic('eta', 3.0 * pm.math.invlogit(mu_eta + z_eta * sigma_eta), dims='subject')
            theta = pm.Deterministic('theta', 10.0 * pm.math.invlogit(mu_theta + z_theta * sigma_theta), dims='subject')
            epsilon = pm.Deterministic('epsilon', 0.2 * pm.math.invlogit(mu_eps + z_eps * sigma_eps), dims='subject')

            alpha_t = alpha[self.subj_idx][:, None]
            lam_t   = lam[self.subj_idx][:, None]
            delta_t = delta[self.subj_idx][:, None]
            eta_t   = eta[self.subj_idx][:, None]
            theta_t = theta[self.subj_idx][:, None]
            eps_t   = epsilon[self.subj_idx][:, None]

            # 4. Вероятности Prelec
            P_safe = pt.clip(pt.as_tensor(self.P_succ_mat), 1e-6, 1.0 - 1e-6)
            
            w_p_succ = pt.exp(-delta_t * (-pt.log(P_safe))**eta_t)
            w_p_loss = pt.exp(-delta_t * (-pt.log(1.0 - P_safe))**eta_t)

            # 5. Полезность
            V_gain = pt.power(pt.as_tensor(self.Gain_mat) + 1e-12, alpha_t)
            V_loss = -lam_t * pt.power(pt.as_tensor(self.Loss_mat), alpha_t)

            EU = w_p_succ * V_gain + w_p_loss * V_loss
            
            Mask_tensor = pt.as_tensor(self.Mask_mat)
            EU_masked = pt.switch(Mask_tensor > 0, EU, -np.inf)
            EU_masked = pt.set_subtensor(EU_masked[:, 0], 0.0) # Полезность 0 карт всегда 0

            # 6. Softmax
            EU_max = pt.max(EU_masked, axis=1, keepdims=True)
            exp_EU = pt.exp(theta_t * (EU_masked - EU_max))
            exp_EU = pt.switch(Mask_tensor > 0, exp_EU, 0.0)
            probs_softmax = exp_EU / pt.sum(exp_EU, axis=1, keepdims=True)

            # 7. Lapse
            n_opts = pt.sum(Mask_tensor, axis=1, keepdims=True)
            probs = (1 - eps_t) * probs_softmax + eps_t * pt.switch(Mask_tensor > 0, 1.0 / n_opts, 0.0)

            pm.Categorical('choice_obs', p=probs, observed=self.choice_idx, dims="obs_idx")
            
        return model

    def fit(self, draws=1500, tune=1000, chains=4, target_accept=0.95):
        print(f"[*] Fitting HBA Cold Haffke Model: N={self.n_users}, Trials={self.n_trials}")
        model = self.build_model()
        with model:
            idata = pm.sample(draws=draws, tune=tune, chains=chains, 
                              target_accept=target_accept, progressbar=False, 
                              return_inferencedata=True)
            
            # Убедитесь, что ArviZ видит лог-правдоподобие
            # Если оно уже определено как Deterministic 'log_lik', 
            # ArviZ должен подхватить его автоматически, если указать var_name
            loo = az.loo(idata, var_name="log_lik")
            print(f"[FIT METRICS] Model LOOIC: {loo.elpd_loo:.2f}")
        return idata, model, loo

    def posterior_predictive_check(self, idata, model, save_plot_path="cold_haffke_ppc.png"):
        print("\n[*] Запуск True HBA PPC (Cold Haffke)...")
        with model:
            ppc = pm.sample_posterior_predictive(idata, extend_inferencedata=False, progressbar=False)

        y_sim = ppc.posterior_predictive['choice_obs'].values
        y_sim_flat = y_sim.reshape(-1, self.n_trials)
        choice_real = self.choice_idx

        metrics = []
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        plot_idx = 0

        for uid_code, uid_label in enumerate(self.user_ids):
            user_mask = (self.subj_idx == uid_code)
            y_real_u = choice_real[user_mask]
            y_sim_u = y_sim_flat[:, user_mask]
            y_sim_mean = y_sim_u.mean(axis=0)
            
            if np.std(y_real_u) == 0 or np.std(y_sim_mean) == 0:
                r2 = np.nan
            else:
                r2 = r2_score(y_real_u, y_sim_mean)
                
            rmse = np.sqrt(mean_squared_error(y_real_u, y_sim_mean))
            msd = mean_squared_error(y_real_u, y_sim_mean)
            mae = mean_absolute_error(y_real_u, y_sim_mean)
            
            sim_modes = mode(y_sim_u, axis=0, keepdims=False).mode
            hit_rate = np.mean(sim_modes == y_real_u)
            
            sim_user_means = y_sim_u.mean(axis=1)
            ppp = np.mean(sim_user_means >= y_real_u.mean())
            ppp_two_sided = 2 * min(ppp, 1 - ppp)

            metrics.append({
                'user_id': uid_label, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 
                'MSD': msd, 'Hit_Rate': hit_rate, 'ppp': ppp_two_sided,
                'real_mean_k': y_real_u.mean(), 'sim_mean_k': sim_user_means.mean()
            })

            # Timecourse
            if plot_idx < 4:
                window = min(10, len(y_real_u))
                real_rolling = pd.Series(y_real_u).rolling(window, min_periods=1).mean()
                sim_rollings = np.array([pd.Series(row).rolling(window, min_periods=1).mean() for row in y_sim_u])
                hdi_lower = np.percentile(sim_rollings, 2.5, axis=0)
                hdi_upper = np.percentile(sim_rollings, 97.5, axis=0)
                sim_mean_rolling = sim_rollings.mean(axis=0)

                ax = axes[plot_idx]
                ax.plot(real_rolling.values, label='Real (MA)', color='crimson', lw=2)
                ax.plot(sim_mean_rolling, label='Simulated Mean', color='teal', lw=2)
                ax.fill_between(range(len(real_rolling)), hdi_lower, hdi_upper, color='teal', alpha=0.3, label='95% HDI')
                ax.set_title(f'PPC Timecourse (Haffke Cold): User {uid_label}')
                ax.set_xlabel('Trial Sequence')
                ax.set_ylabel('Cards Turned (k)')
                ax.legend()
                plot_idx += 1

        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=300)
        plt.close()

        return pd.DataFrame(metrics)

    def _numpy_simulate(self, df_trials, params_dict):
        choices = []
        for t, row in df_trials.iterrows():
            c = int(row['cards_left'])
            g = int(row['gains_left'])
            gain_amt = float(row['gain_amount']) / self.scale_factor
            loss_amt = abs(float(row['loss_amount'])) / self.scale_factor
            
            opts = row['options'] if isinstance(row['options'], (list, np.ndarray)) else eval(row['options'])
            n_opts = len(opts)
            
            a, l, d, e = params_dict['alpha'], params_dict['lambda'], params_dict['delta'], params_dict['eta']
            t_inv, eps = params_dict['theta'], params_dict['epsilon']
            
            EU = np.zeros(n_opts)
            for i, m in enumerate(opts):
                if m == 0: p = 1.0
                elif m > g: p = 0.0
                else:
                    logp = lgamma(g+1) - lgamma(m+1) - lgamma(g-m+1) - (lgamma(c+1) - lgamma(m+1) - lgamma(c-m+1))
                    p = np.exp(logp)
                
                p_safe = np.clip(p, 1e-6, 1.0 - 1e-6)
                w_p = np.exp(-d * (-np.log(p_safe))**e)
                w_p_loss = np.exp(-d * (-np.log(1 - p_safe))**e)
                
                v_gain = (m * gain_amt)**a
                v_loss = -l * (loss_amt)**a
                
                EU[i] = w_p * v_gain + w_p_loss * v_loss
            
            EU[0] = 0.0 # Обязательно 0 для k=0
            max_eu = np.max(EU)
            exp_eu = np.exp(t_inv * (EU - max_eu))
            probs = exp_eu / np.sum(exp_eu)
            probs = (1 - eps) * probs + eps / n_opts
            
            choices.append(np.random.choice(opts, p=probs))
            
        return choices

    def parameter_recovery(self, n_subjects=50, save_plot_path="cold_haffke_recovery.png"):
        print(f"\n[*] True HBA Parameter Recovery Cold Haffke (N={n_subjects})...")
        rng = np.random.default_rng(42)
        
        true_alpha = rng.uniform(0.1, 2.5, n_subjects)
        true_lam = rng.uniform(0.1, 8.0, n_subjects)
        true_delta = rng.uniform(0.1, 2.5, n_subjects)
        true_eta = rng.uniform(0.1, 2.5, n_subjects)
        true_theta = rng.uniform(0.1, 8.0, n_subjects)
        true_eps = rng.uniform(0.0, 0.15, n_subjects)

        template_df = self.df[self.df['user_id'] == self.user_ids[0]].copy()
        
        sim_data_frames = []
        for i in range(n_subjects):
            user_df = template_df.copy()
            p_dict = {'alpha': true_alpha[i], 'lambda': true_lam[i], 'delta': true_delta[i], 
                      'eta': true_eta[i], 'theta': true_theta[i], 'epsilon': true_eps[i]}
            
            user_df['choice'] = self._numpy_simulate(user_df, p_dict)
            user_df['user_id'] = f'syn_{i}'
            sim_data_frames.append(user_df)

        sim_df = pd.concat(sim_data_frames, ignore_index=True)
        
        rec_model = HaffkeColdModel_HBA(sim_df, scale_factor=self.scale_factor)
        rec_idata, _, _ = rec_model.fit(draws=800, tune=800, chains=4, target_accept=0.90)
        
        post = rec_idata.posterior
        fit_params = {p: post[p].median(dim=("chain", "draw")).values for p in ['alpha', 'lambda', 'delta', 'eta', 'theta', 'epsilon']}

        df_results = pd.DataFrame({
            "user_id": [f'syn_{i}' for i in range(n_subjects)],
            "true_alpha": true_alpha, "fit_alpha": fit_params['alpha'],
            "true_lam": true_lam, "fit_lam": fit_params['lambda'],
            "true_delta": true_delta, "fit_delta": fit_params['delta'],
            "true_eta": true_eta, "fit_eta": fit_params['eta'],
            "true_theta": true_theta, "fit_theta": fit_params['theta'],
            "true_eps": true_eps, "fit_eps": fit_params['epsilon']
        })

        metrics_list = []
        params = [("alpha", true_alpha, fit_params['alpha'], 'alpha'), 
                  ("lam", true_lam, fit_params['lambda'], 'lambda'), 
                  ("delta", true_delta, fit_params['delta'], 'delta'), 
                  ("eta", true_eta, fit_params['eta'], 'eta'),
                  ("theta", true_theta, fit_params['theta'], 'theta'),
                  ("epsilon", true_eps, fit_params['epsilon'], 'epsilon')]

        fig, axes = plt.subplots(1, 6, figsize=(25, 4))
        fig.suptitle('HBA Parameter Recovery (Cold Haffke)', fontsize=16)

        for ax, (p_name, true_v, fit_v, trace_name) in zip(axes, params):
            r_val = np.corrcoef(true_v, fit_v)[0, 1]
            r2_val = r2_score(true_v, fit_v)
            rmse = np.sqrt(mean_squared_error(true_v, fit_v))
            bias = np.mean(fit_v - true_v)
            
            hdi = az.hdi(post[trace_name])
            lower = hdi[trace_name].sel(hdi='lower').values
            upper = hdi[trace_name].sel(hdi='higher').values
            coverage = np.mean((true_v >= lower) & (true_v <= upper))
            
            metrics_list.append({'Parameter': p_name, 'Pearson_r': r_val, 'R2': r2_val, 'RMSE': rmse, 'Bias': bias, 'Coverage_95': coverage})
            
            sns.scatterplot(x=true_v, y=fit_v, ax=ax, s=60, color='indigo', alpha=0.7)
            min_v, max_v = min(true_v.min(), fit_v.min()), max(true_v.max(), fit_v.max())
            ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='y=x')
            ax.set_title(f'{p_name} (r: {r_val:.2f}, Cov: {coverage:.2f})')
            ax.set_xlabel('True')
            ax.set_ylabel('Recovered')

        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=300)
        plt.close()

        return df_results, pd.DataFrame(metrics_list)