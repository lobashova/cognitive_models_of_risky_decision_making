import pymc as pm
import pytensor.tensor as pt
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gammaln, expit
from scipy.stats import mode
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

class COLDModel2_HBA:
    def __init__(self, data_df, max_N=32, scale_factor=100.0):
        self.max_N = int(max_N)
        self.scale_factor = scale_factor
        self.data = data_df.copy().reset_index(drop=True)
        self.data['gain_scaled'] = self.data['gain_amount'].astype(float) / self.scale_factor
        self.data['loss_scaled'] = self.data['loss_amount'].abs().astype(float) / self.scale_factor
        self.data['subj_idx'], self.subj_labels = pd.factorize(self.data['user_id'])
        self.n_subj = len(self.subj_labels)
        if 'round_id' not in self.data.columns:
            self.data['round_id'] = self.data.groupby(['user_id', 'trial_number']).ngroup()

    @staticmethod
    def log_comb(n, k):
        if (k < 0) or (k > n) or (n < 0): return -np.inf
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

    def precompute_matrices(self, df):
        T, K = len(df), self.max_N + 1
        BaseGain, BaseLoss, P_no_loss, Mask = np.zeros((T, K)), np.zeros(T), np.zeros((T, K)), np.zeros((T, K))
        Ls, Gs, Losses = df['loss_cards'].values.astype(int), df['gain_scaled'].values, df['loss_scaled'].values
        N_trials = df.get('total_cards', pd.Series(self.max_N, index=df.index)).values.astype(int)

        for t in range(T):
            BaseLoss[t] = abs(Losses[t])
            N_t, L_t = N_trials[t], Ls[t]
            for k in range(K):
                if k > N_t:
                    Mask[t, k] = -1e10
                    continue
                BaseGain[t, k] = k * Gs[t]
                if k > (N_t - L_t):
                    P_no_loss[t, k] = 0.0
                else:
                    P_no_loss[t, k] = np.exp(self.log_comb(N_t - L_t, k) - self.log_comb(N_t, k))
        return BaseGain, BaseLoss, P_no_loss, Mask

    def build_model(self):
        subj_idx = self.data['subj_idx'].values
        observed_k = self.data['num_cards'].values.astype(int)
        BaseGain, BaseLoss, P_no_loss, Mask = self.precompute_matrices(self.data)

        with pm.Model() as model:
            mu_rho = pm.Normal('mu_rho', mu=0, sigma=1.5)
            mu_lam = pm.Normal('mu_lam', mu=0, sigma=1.5)
            mu_delta = pm.Normal('mu_delta', mu=0, sigma=1.5)
            mu_eta = pm.Normal('mu_eta', mu=0, sigma=1.5)
            mu_beta = pm.Normal('mu_beta', mu=0, sigma=1.5)

            sigma_rho = pm.HalfNormal('sigma_rho', sigma=1)
            sigma_lam = pm.HalfNormal('sigma_lam', sigma=1)
            sigma_delta = pm.HalfNormal('sigma_delta', sigma=1)
            sigma_eta = pm.HalfNormal('sigma_eta', sigma=1)
            sigma_beta = pm.HalfNormal('sigma_beta', sigma=1)

            rho_raw = pm.Normal('rho_raw', mu=0, sigma=1, shape=self.n_subj)
            lam_raw = pm.Normal('lam_raw', mu=0, sigma=1, shape=self.n_subj)
            delta_raw = pm.Normal('delta_raw', mu=0, sigma=1, shape=self.n_subj)
            eta_raw = pm.Normal('eta_raw', mu=0, sigma=1, shape=self.n_subj)
            beta_raw = pm.Normal('beta_raw', mu=0, sigma=1, shape=self.n_subj)

            rho = pm.Deterministic('rho', 3.0 * pm.math.invlogit(mu_rho + sigma_rho * rho_raw))
            lambd = pm.Deterministic('lambd', 10.0 * pm.math.invlogit(mu_lam + sigma_lam * lam_raw))
            delta = pm.Deterministic('delta', 3.0 * pm.math.invlogit(mu_delta + sigma_delta * delta_raw))
            eta = pm.Deterministic('eta', 3.0 * pm.math.invlogit(mu_eta + sigma_eta * eta_raw))
            beta = pm.Deterministic('beta', 10.0 * pm.math.invlogit(mu_beta + sigma_beta * beta_raw))

            rho_t = rho[subj_idx]
            lambd_t = lambd[subj_idx]
            delta_t = delta[subj_idx]
            eta_t = eta[subj_idx]
            beta_t = beta[subj_idx]

            pg = pt.clip(pt.as_tensor_variable(P_no_loss), 1e-6, 1.0 - 1e-6)
            pl = pt.clip(1.0 - pt.as_tensor_variable(P_no_loss), 1e-6, 1.0 - 1e-6)
            
            wg = pt.exp(-delta_t[:, None] * (-pt.log(pg)) ** eta_t[:, None])
            wl = pt.exp(-delta_t[:, None] * (-pt.log(pl)) ** eta_t[:, None])

            u_gain = pt.power(pt.as_tensor_variable(BaseGain) + 1e-12, rho_t[:, None])
            u_loss = -lambd_t * pt.power(pt.as_tensor_variable(BaseLoss), rho_t)

            eu = wg * u_gain + wl * u_loss[:, None]
            eu = pt.set_subtensor(eu[:, 0], 0.0)
            
            logits = eu / beta_t[:, None] + pt.as_tensor_variable(Mask)
            pm.Categorical('choices', logit_p=logits, observed=observed_k)
        return model

    def fit(self, draws=1500, tune=1000, target_accept=0.95, chains=4):
        print(f"[*] HBA Fit CCT-COLD Model 2: Subj={self.n_subj}, Trials={len(self.data)}")
        model = self.build_model()
        with model:
            idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept, 
                              progressbar=False, return_inferencedata=True)
            pm.compute_log_likelihood(idata)
            loo = az.loo(idata, pointwise=True)
        return idata, model, loo

    def predictive_check(self, idata, model, save_plot_path="cold_wullhorst2_ppc_timecourse.png"):
        print("\n[*] Запуск True HBA Posterior Predictive Check (Model 2)...")
        with model:
            ppc = pm.sample_posterior_predictive(idata, extend_inferencedata=False, progressbar=False)
            
        y_sim = ppc.posterior_predictive['choices'].values
        y_sim_flat = y_sim.reshape(-1, y_sim.shape[-1])
        choice_real = self.data['num_cards'].values
        subj_idx = self.data['subj_idx'].values

        metrics = []
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        plot_idx = 0

        for uid_code, uid_label in enumerate(self.subj_labels):
            user_mask = (subj_idx == uid_code)
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
                ax.set_title(f'PPC Timecourse (Model 2): User {uid_label}')
                ax.set_xlabel('Trial Sequence')
                ax.set_ylabel('Cards Turned (k)')
                ax.legend()
                plot_idx += 1

        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=300)
        plt.close()

        return pd.DataFrame(metrics)

    def parameter_recovery(self, n_subjects=50, save_plot_path="cold_wullhorst2_recovery.png"):
        print(f"\n[*] True HBA Parameter Recovery Model 2 (N={n_subjects})...")
        rng = np.random.default_rng(42)
        
        true_rho = rng.uniform(0.1, 2.5, n_subjects)
        true_lambd = rng.uniform(0.1, 8.0, n_subjects)
        true_delta = rng.uniform(0.1, 2.5, n_subjects)
        true_eta = rng.uniform(0.1, 2.5, n_subjects)
        true_beta = rng.uniform(0.1, 8.0, n_subjects)

        extract_cols = ['loss_cards', 'gain_amount', 'loss_amount']
        if 'total_cards' in self.data.columns:
            extract_cols.append('total_cards')
        unique_trials = self.data[extract_cols].drop_duplicates().copy()
        if 'total_cards' not in unique_trials.columns:
            unique_trials['total_cards'] = self.max_N

        sim_data = []
        for i in range(n_subjects):
            subj_df = unique_trials.copy()
            subj_df['user_id'] = f'syn_{i}'
            subj_df['trial_number'] = np.arange(1, len(subj_df) + 1)
            
            N_t = subj_df['total_cards'].values.astype(int)
            L_t = subj_df['loss_cards'].values.astype(int)
            G_t = subj_df['gain_amount'].values.astype(float) / self.scale_factor
            Loss_t = subj_df['loss_amount'].abs().values.astype(float) / self.scale_factor
            
            choices = []
            for t in range(len(subj_df)):
                ks = np.arange(N_t[t] + 1)
                u_gain = (ks * G_t[t]) ** true_rho[i]
                u_gain[0] = 0.0
                u_loss = -true_lambd[i] * (Loss_t[t] ** true_rho[i])
                
                p_no_loss = np.zeros(N_t[t] + 1)
                for k in ks:
                    if k <= (N_t[t] - L_t[t]):
                        p_no_loss[k] = np.exp(self.log_comb(N_t[t] - L_t[t], k) - self.log_comb(N_t[t], k))
                        
                pg_c = np.clip(p_no_loss, 1e-6, 1.0 - 1e-6)
                pl_c = np.clip(1.0 - p_no_loss, 1e-6, 1.0 - 1e-6)
                wg = np.exp(-true_delta[i] * ((-np.log(pg_c)) ** true_eta[i]))
                wl = np.exp(-true_delta[i] * ((-np.log(pl_c)) ** true_eta[i]))

                eu = wg * u_gain + wl * u_loss
                eu[0] = 0.0
                logits = eu / true_beta[i]
                
                # Маскируем невозможные значения k > N
                mask = np.zeros(len(logits))
                mask[N_t[t]+1:] = -1e9
                logits += mask
                
                logits -= np.max(logits)
                probs = np.exp(logits) / np.sum(np.exp(logits))
                
                choices.append(rng.choice(ks, p=probs))
                
            subj_df['num_cards'] = choices
            sim_data.append(subj_df)
            
        syn_df = pd.concat(sim_data, ignore_index=True)

        rec_model = COLDModel2_HBA(syn_df, max_N=self.max_N, scale_factor=self.scale_factor)
        rec_idata, _, _ = rec_model.fit(draws=800, tune=800, chains=4, target_accept=0.90)
        
        post = rec_idata.posterior
        fit_rho = post['rho'].median(dim=("chain", "draw")).values
        fit_lambd = post['lambd'].median(dim=("chain", "draw")).values
        fit_delta = post['delta'].median(dim=("chain", "draw")).values
        fit_eta = post['eta'].median(dim=("chain", "draw")).values
        fit_beta = post['beta'].median(dim=("chain", "draw")).values

        df_results = pd.DataFrame({
            "user_id": [f'syn_{i}' for i in range(n_subjects)],
            "true_rho": true_rho, "fit_rho": fit_rho,
            "true_lambd": true_lambd, "fit_lambd": fit_lambd,
            "true_delta": true_delta, "fit_delta": fit_delta,
            "true_eta": true_eta, "fit_eta": fit_eta,
            "true_beta": true_beta, "fit_beta": fit_beta
        })

        metrics_list = []
        params = [("rho", true_rho, fit_rho, 'rho'), ("lambda", true_lambd, fit_lambd, 'lambd'), 
                  ("delta", true_delta, fit_delta, 'delta'), ("eta", true_eta, fit_eta, 'eta'),
                  ("beta", true_beta, fit_beta, 'beta')]

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle('HBA Parameter Recovery (Cold CCT Model 2)', fontsize=16)

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