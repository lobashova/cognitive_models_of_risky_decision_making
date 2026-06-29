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

class COLDModelHBA:
    """
    Иерархическая байесовская реализация CCT-cold (Prospect Theory, Model 1).
    Полностью оптимизирована под суперкомпьютер (HPC) и стандарты Q1/Q2.
    """
    def __init__(self, data_df, max_N=32, scale_factor=100.0):
        self.max_N = int(max_N)
        self.scale_factor = scale_factor
        self.data = data_df.copy().reset_index(drop=True)
        
        # Масштабирование (Критично для избегания NaN в Softmax)
        self.data['gain_scaled'] = self.data['gain_amount'].astype(float) / self.scale_factor
        self.data['loss_scaled'] = self.data['loss_amount'].abs().astype(float) / self.scale_factor
        
        self.data['subj_idx'], self.subj_labels = pd.factorize(self.data['user_id'])
        self.n_subj = len(self.subj_labels)
        
        if 'round_id' not in self.data.columns:
            self.data['round_id'] = self.data.groupby(['user_id', 'trial_number']).ngroup()

    @staticmethod
    def log_comb(n, k):
        if (k < 0) or (k > n) or (n < 0):
            return -np.inf
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

    def precompute_matrices(self, df):
        T = len(df)
        K = self.max_N + 1
        
        BaseGain = np.zeros((T, K))
        BaseLoss = np.zeros(T)
        P_no_loss = np.zeros((T, K))
        Mask = np.zeros((T, K)) 

        Ls = df['loss_cards'].values.astype(int)
        Gs = df['gain_scaled'].values
        Losses = df['loss_scaled'].values
        N_trials = df.get('total_cards', pd.Series(self.max_N, index=df.index)).values.astype(int)

        for t in range(T):
            BaseLoss[t] = abs(Losses[t])
            N_t = N_trials[t]
            L_t = Ls[t]

            for k in range(K):
                if k > N_t:
                    Mask[t, k] = -1e10 
                    continue
                
                BaseGain[t, k] = k * Gs[t]
                
                if k > (N_t - L_t):
                    P_no_loss[t, k] = 0.0
                else:
                    log_p = self.log_comb(N_t - L_t, k) - self.log_comb(N_t, k)
                    P_no_loss[t, k] = np.exp(log_p)
                    
        return BaseGain, BaseLoss, P_no_loss, Mask

    def build_model(self):
        subj_idx = self.data['subj_idx'].values
        observed_k = self.data['num_cards'].values.astype(int)
        BaseGain, BaseLoss, P_no_loss, Mask = self.precompute_matrices(self.data)

        with pm.Model() as model:
            # 1. Групповые гиперпараметры
            mu_rho = pm.Normal('mu_rho', mu=0, sigma=1.5)
            mu_lam = pm.Normal('mu_lam', mu=0, sigma=1.5)
            mu_beta = pm.Normal('mu_beta', mu=0, sigma=1.5)

            sigma_rho = pm.HalfNormal('sigma_rho', sigma=1)
            sigma_lam = pm.HalfNormal('sigma_lam', sigma=1)
            sigma_beta = pm.HalfNormal('sigma_beta', sigma=1)

            # 2. Индивидуальный уровень (Non-centered)
            rho_raw = pm.Normal('rho_raw', mu=0, sigma=1, shape=self.n_subj)
            lam_raw = pm.Normal('lam_raw', mu=0, sigma=1, shape=self.n_subj)
            beta_raw = pm.Normal('beta_raw', mu=0, sigma=1, shape=self.n_subj)

            # 3. Гладкая трансформация границ (вместо clip)
            # rho: (0, 3), lam: (0, 10), beta: (0, 10)
            rho = pm.Deterministic('rho', 3.0 * pm.math.invlogit(mu_rho + sigma_rho * rho_raw))
            lambd = pm.Deterministic('lambd', 10.0 * pm.math.invlogit(mu_lam + sigma_lam * lam_raw))
            beta = pm.Deterministic('beta', 10.0 * pm.math.invlogit(mu_beta + sigma_beta * beta_raw))

            rho_t = rho[subj_idx]
            lambd_t = lambd[subj_idx]
            beta_t = beta[subj_idx]

            # 4. Векторизованная математика (T x K)
            u_gain = pt.power(BaseGain + 1e-12, rho_t[:, None])
            u_loss = -lambd_t * pt.power(BaseLoss, rho_t)

            eu = P_no_loss * u_gain + (1 - P_no_loss) * u_loss[:, None]
            logits = eu / beta_t[:, None]
            logits = logits + Mask 

            # 5. Правдоподобие
            pm.Categorical('choices', logit_p=logits, observed=observed_k)
            
        return model

    def fit(self, draws=1500, tune=1000, target_accept=0.95, chains=4):
        print(f"[*] HBA Fit CCT-COLD: Subj={self.n_subj}, Trials={len(self.data)}")
        model = self.build_model()
        with model:
            idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept, 
                              progressbar=False, return_inferencedata=True)
            
            # ИСПРАВЛЕНИЕ: Правильная функция расчета log-likelihood
            pm.compute_log_likelihood(idata)
            
            loo = az.loo(idata, pointwise=True)
        return idata, model, loo

    def predictive_check(self, idata, model, save_plot_path="cold_wullhorst1_ppc_timecourse.png"):
        print("\n[*] Запуск True HBA Posterior Predictive Check...")
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
                ax.set_title(f'PPC Timecourse (k cards): User {uid_label}')
                ax.set_xlabel('Trial Sequence')
                ax.set_ylabel('Cards Turned (k)')
                ax.legend()
                plot_idx += 1

        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=300)
        plt.close()

        return pd.DataFrame(metrics)

    def parameter_recovery(self, n_subjects=50, save_plot_path="cold_wullhorst1_recovery.png"):
        print(f"\n[*] True HBA Parameter Recovery (N={n_subjects})...")
        rng = np.random.default_rng(42)
        
        true_rho = rng.uniform(0.1, 2.5, n_subjects)
        true_lambd = rng.uniform(0.1, 8.0, n_subjects)
        true_beta = rng.uniform(0.1, 8.0, n_subjects)

        # Шаблон раундов из реальных данных
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
            # Важно: симулируем с учетом масштабирования
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
                        
                eu = p_no_loss * u_gain + (1 - p_no_loss) * u_loss
                logits = eu / true_beta[i]
                logits -= np.max(logits) # защита от переполнения
                probs = np.exp(logits) / np.sum(np.exp(logits))
                
                choices.append(rng.choice(ks, p=probs))
                
            subj_df['num_cards'] = choices
            sim_data.append(subj_df)
            
        syn_df = pd.concat(sim_data, ignore_index=True)

        # Подгонка
        rec_model = COLDModelHBA(syn_df, max_N=self.max_N, scale_factor=self.scale_factor)
        rec_idata, _, _ = rec_model.fit(draws=800, tune=800, chains=4, target_accept=0.90)
        
        post = rec_idata.posterior
        fit_rho = post['rho'].median(dim=("chain", "draw")).values
        fit_lambd = post['lambd'].median(dim=("chain", "draw")).values
        fit_beta = post['beta'].median(dim=("chain", "draw")).values

        df_results = pd.DataFrame({
            "user_id": [f'syn_{i}' for i in range(n_subjects)],
            "true_rho": true_rho, "fit_rho": fit_rho,
            "true_lambd": true_lambd, "fit_lambd": fit_lambd,
            "true_beta": true_beta, "fit_beta": fit_beta
        })

        metrics_list = []
        params = [("rho", true_rho, fit_rho, 'rho'), 
                  ("lambda", true_lambd, fit_lambd, 'lambd'), 
                  ("beta", true_beta, fit_beta, 'beta')]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('HBA Parameter Recovery (Cold CCT)', fontsize=16)

        for ax, (p_name, true_v, fit_v, trace_name) in zip(axes, params):
            r_val = np.corrcoef(true_v, fit_v)[0, 1]
            r2_val = r2_score(true_v, fit_v)
            rmse = np.sqrt(mean_squared_error(true_v, fit_v))
            bias = np.mean(fit_v - true_v)
            
            # Расчет Coverage
            hdi = az.hdi(post[trace_name])
            lower = hdi[trace_name].sel(hdi='lower').values
            upper = hdi[trace_name].sel(hdi='higher').values
            coverage = np.mean((true_v >= lower) & (true_v <= upper))
            
            metrics_list.append({
                'Parameter': p_name, 'Pearson_r': r_val, 'R2': r2_val, 
                'RMSE': rmse, 'Bias': bias, 'Coverage_95': coverage
            })
            
            sns.scatterplot(x=true_v, y=fit_v, ax=ax, s=60, color='indigo', alpha=0.7)
            min_v, max_v = min(true_v.min(), fit_v.min()), max(true_v.max(), fit_v.max())
            ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='y=x')
            ax.set_title(f'{p_name} (r: {r_val:.2f}, Cov: {coverage:.2f})')
            ax.set_xlabel('True values')
            ax.set_ylabel('Recovered values (Median)')
            ax.legend()

        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=300)
        plt.close()

        return df_results, pd.DataFrame(metrics_list)