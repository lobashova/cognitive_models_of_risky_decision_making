import pymc as pm
import pytensor.tensor as pt
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
from scipy.stats import mode
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

class HOTModel2_HBA:
    """
    Иерархическая Байесовская Модель (HBA) для CCT-hot (Wüllhorst Model 2: PT + Prelec).
    Полностью оптимизировано под суперкомпьютер (HPC) и стандарты Q1/Q2.
    """
    def __init__(self, data_df, scale_factor=100.0):
        self.data = data_df.copy()
        
        # Масштабирование наград и штрафов для стабильности Softmax
        self.scale_factor = scale_factor
        self.data['gain_scaled'] = self.data['gain_amount'] / self.scale_factor
        self.data['loss_scaled'] = self.data['loss_amount'].abs() / self.scale_factor
        
        # Индексация пользователей для HBA
        self.data['subj_idx'], self.subj_labels = pd.factorize(self.data['user_id'])
        self.n_subj = len(self.subj_labels)
        
        if 'round_id' not in self.data.columns:
            self.data['round_id'] = self.data.groupby(['user_id', 'trial_number']).ngroup()

    def build_model(self):
        subj_idx = self.data['subj_idx'].values
        gain_amt = self.data['gain_scaled'].values
        loss_amt = self.data['loss_scaled'].values
        loss_cards = self.data['loss_cards'].values
        flip_no = self.data['flip_number'].values
        choice = self.data['choice'].values

        # Получаем общее количество карт
        total = self.data["total_cards"].values if "total_cards" in self.data.columns else np.full_like(flip_no, 32.0)
        denom = np.maximum(total - (flip_no - 1.0), 1.0)
        
        p_loss_val = loss_cards / denom
        p_gain_val = 1.0 - p_loss_val

        with pm.Model() as model:
            # 1. Гиперпараметры (Групповой уровень)
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

            # 2. Индивидуальный уровень (Non-centered parameterization)
            rho_raw = pm.Normal('rho_raw', mu=0, sigma=1, shape=self.n_subj)
            lam_raw = pm.Normal('lam_raw', mu=0, sigma=1, shape=self.n_subj)
            delta_raw = pm.Normal('delta_raw', mu=0, sigma=1, shape=self.n_subj)
            eta_raw = pm.Normal('eta_raw', mu=0, sigma=1, shape=self.n_subj)
            beta_raw = pm.Normal('beta_raw', mu=0, sigma=1, shape=self.n_subj)

            # 3. Трансформация (Гладкие границы вместо clip для NUTS)
            rho = pm.Deterministic('rho', 3.0 * pm.math.invlogit(mu_rho + sigma_rho * rho_raw))
            lam = pm.Deterministic('lam', 10.0 * pm.math.invlogit(mu_lam + sigma_lam * lam_raw))
            delta = pm.Deterministic('delta', 3.0 * pm.math.invlogit(mu_delta + sigma_delta * delta_raw))
            eta = pm.Deterministic('eta', 3.0 * pm.math.invlogit(mu_eta + sigma_eta * eta_raw))
            beta = pm.Deterministic('beta', 1.0 * pm.math.invlogit(mu_beta + sigma_beta * beta_raw))

            # 4. Полезность и Prelec Weights
            rho_t = rho[subj_idx]
            max_pumps = lam[subj_idx]
            delta_t = delta[subj_idx]
            eta_t = eta[subj_idx]
            beta_t = beta[subj_idx]

            u_gain = gain_amt ** rho_t
            u_loss = -max_pumps * (loss_amt ** rho_t) 
            
            # Защита от log(0) и log(1)
            pg = pt.clip(p_gain_val, 1e-6, 1.0 - 1e-6)
            pl = pt.clip(p_loss_val, 1e-6, 1.0 - 1e-6)
            
            w_gain = pt.exp(-delta_t * (-pt.log(pg)) ** eta_t)
            w_loss = pt.exp(-delta_t * (-pt.log(pl)) ** eta_t)

            EU = w_gain * u_gain + w_loss * u_loss
            p_turn = pm.math.invlogit(EU / beta_t)

            # 5. Функция правдоподобия
            pm.Bernoulli('choice_obs', p=p_turn, observed=choice)
            
        return model

    def fit(self, draws=1500, tune=1000, chains=4, target_accept=0.95):
        print(f"[*] HBA Fit Model 2: Subj={self.n_subj}, Trials={len(self.data)}")
        model = self.build_model()
        with model:
            idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept, 
                              progressbar=False, return_inferencedata=True)
            # ИСПРАВЛЕНО: Правильная функция расчета лог-правдоподобия для LOOIC
            pm.compute_log_likelihood(idata)
            
            # ИСПРАВЛЕНО: Обращение к .elpd_loo вместо .loo
            loo = az.loo(idata, pointwise=True)
            print(f"[FIT METRICS] Model 2 LOOIC: {loo.elpd_loo:.2f}")
        return idata, model, loo

    def posterior_predictive_check(self, idata, model, save_plot_path="hot_wullhorst2_ppc.png"):
        print("\n[*] Запуск True HBA PPC (Model 2)...")
        with model:
            ppc = pm.sample_posterior_predictive(idata, extend_inferencedata=False, progressbar=False)
            
        y_sim = ppc.posterior_predictive['choice_obs'].values
        y_sim_flat = y_sim.reshape(-1, y_sim.shape[-1])
        choice_real = self.data['choice'].values
        subj_idx = self.data['subj_idx'].values
        round_ids = self.data['round_id'].values

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
            
            user_rounds, user_rounds_mapped = np.unique(round_ids[user_mask], return_inverse=True)
            real_cards_per_round = np.bincount(user_rounds_mapped, weights=y_real_u).mean()
            
            sim_cards_per_round = np.array([np.bincount(user_rounds_mapped, weights=row).mean() for row in y_sim_u])
            ppp = np.mean(sim_cards_per_round >= real_cards_per_round)
            ppp_two_sided = 2 * min(ppp, 1 - ppp)

            metrics.append({
                'user_id': uid_label, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 
                'MSD': msd, 'Hit_Rate': hit_rate, 'ppp': ppp_two_sided,
                'real_cards_avg': real_cards_per_round, 'sim_cards_avg': sim_cards_per_round.mean()
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
                ax.set_title(f'PPC Timecourse: User {uid_label}')
                ax.set_xlabel('Trial Sequence')
                ax.set_ylabel('Turn Rate')
                ax.legend()
                plot_idx += 1

        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=300)
        plt.close()

        return pd.DataFrame(metrics)

    @staticmethod
    def simulate_data(template_df, rho, lam, delta, eta, beta, scale_factor=100.0):
        sim = template_df.copy()
        sim['choice'] = 0
        eps = 1e-6

        gain_amt = sim['gain_amount'].values / scale_factor
        loss_amt = np.abs(sim['loss_amount'].apply(lambda x: x if x < 0 else -x).values.astype(float)) / scale_factor
        
        flip_no = sim['flip_number'].values
        loss_cards = sim['loss_cards'].values
        total = sim["total_cards"].values if "total_cards" in sim.columns else np.full_like(flip_no, 32.0)
        
        denom = np.maximum(total - (flip_no - 1.0), 1.0)
        p_loss = np.clip(loss_cards / denom, eps, 1 - eps)
        p_gain = np.clip(1.0 - p_loss, eps, 1 - eps)

        u_gain = gain_amt ** rho
        u_loss = -lam * (loss_amt ** rho)
        
        w_gain = np.exp(-delta * (-np.log(p_gain)) ** eta)
        w_loss = np.exp(-delta * (-np.log(p_loss)) ** eta)
        
        EU = w_gain * u_gain + w_loss * u_loss
        
        p_turn = expit(EU / beta)
        p_turn = np.clip(p_turn, eps, 1 - eps)
        
        sim['choice'] = np.random.binomial(1, p_turn)
        return sim

    # ИСПРАВЛЕНО: Метод теперь принимает idata первым аргументом, как в main()
    def parameter_recovery(self, idata, n_subjects=50, save_plot_path="hot_wullhorst2_recovery.png"):
        print(f"\n[*] True HBA Parameter Recovery Model 2 (N={n_subjects})...")
        
        uids = self.data['user_id'].unique()[:n_subjects]
        template = pd.concat([self.data[self.data['user_id'] == uid] for uid in uids]).copy()
        template['subj_idx'], subj_labels = pd.factorize(template['user_id'])
        
        rng = np.random.default_rng(42)
        true_rho = rng.uniform(0.1, 2.5, n_subjects)
        true_lam = rng.uniform(0.1, 8.0, n_subjects)
        true_delta = rng.uniform(0.1, 2.5, n_subjects)
        true_eta = rng.uniform(0.1, 2.5, n_subjects)
        true_beta = rng.uniform(0.1, 0.9, n_subjects)
        
        simulated_dfs = []
        for idx, uid in enumerate(subj_labels):
            subj_data = template[template['user_id'] == uid]
            sim_df = self.simulate_data(subj_data, true_rho[idx], true_lam[idx], true_delta[idx], true_eta[idx], true_beta[idx], self.scale_factor)
            simulated_dfs.append(sim_df)
            
        recovery_data = pd.concat(simulated_dfs)
        
        rec_model = HOTModel2_HBA(recovery_data, scale_factor=self.scale_factor)
        rec_idata, _, _ = rec_model.fit(draws=800, tune=800, chains=4, target_accept=0.90)
        
        post = rec_idata.posterior
        fit_rho = post['rho'].median(dim=("chain", "draw")).values
        fit_lam = post['lam'].median(dim=("chain", "draw")).values
        fit_delta = post['delta'].median(dim=("chain", "draw")).values
        fit_eta = post['eta'].median(dim=("chain", "draw")).values
        fit_beta = post['beta'].median(dim=("chain", "draw")).values

        df_results = pd.DataFrame({
            "user_id": subj_labels,
            "true_rho": true_rho, "fit_rho": fit_rho,
            "true_lam": true_lam, "fit_lam": fit_lam,
            "true_delta": true_delta, "fit_delta": fit_delta,
            "true_eta": true_eta, "fit_eta": fit_eta,
            "true_beta": true_beta, "fit_beta": fit_beta
        })

        metrics_list = []
        params = [("rho", true_rho, fit_rho, 'rho'), ("lam", true_lam, fit_lam, 'lam'), 
                  ("delta", true_delta, fit_delta, 'delta'), ("eta", true_eta, fit_eta, 'eta'),
                  ("beta", true_beta, fit_beta, 'beta')]

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle('HBA Parameter Recovery Model 2 (True vs Fit)', fontsize=16)

        for ax, (p_name, true_v, fit_v, trace_name) in zip(axes, params):
            r_val = np.corrcoef(true_v, fit_v)[0, 1]
            r2_val = r2_score(true_v, fit_v)
            rmse = np.sqrt(mean_squared_error(true_v, fit_v))
            bias = np.mean(fit_v - true_v)
            
            hdi = az.hdi(rec_idata.posterior)[trace_name].values
            lower = hdi[:, 0]
            upper = hdi[:, 1]
            coverage = np.mean((true_v >= lower) & (true_v <= upper))
            
            metrics_list.append({'Parameter': p_name, 'Pearson_r': r_val, 'R2': r2_val, 'RMSE': rmse, 'Bias': bias, 'Coverage_95': coverage})
            
            sns.scatterplot(x=true_v, y=fit_v, ax=ax, s=60, color='indigo', alpha=0.7)
            min_v, max_v = min(true_v.min(), fit_v.min()), max(true_v.max(), fit_v.max())
            ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='y=x')
            ax.set_title(f'{p_name} (r: {r_val:.2f}, Cov: {coverage:.2f})')
            ax.set_xlabel('True values')
            ax.set_ylabel('Recovered values (Median)')

        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=300)
        plt.close()

        return df_results, pd.DataFrame(metrics_list)