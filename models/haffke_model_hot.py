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

# =============================================================================
# 1. ПОДГОТОВКА ДАННЫХ
# =============================================================================

def prepare_haffke_input(hot_df, total_cards=32):
    """
    Преобразует DataFrame CCT-hot и рассчитывает объективные (конъюнктивные) вероятности.
    """
    df = hot_df.copy()
    if 'trial_number' in df.columns:
        df = df.rename(columns={'trial_number': 'trial'})

    df['state_n'] = df.groupby(['user_id', 'trial'])['flip_number'].transform(lambda x: x)
    df['n_flipped'] = df.groupby(['user_id', 'trial']).cumcount() + 1
    df['cards_left'] = total_cards - df['n_flipped'] + 1
    df['total_gains_in_trial'] = total_cards - df['loss_cards']
    df['gains_left'] = df['total_gains_in_trial'] - (df['state_n'] - 1)
    df['is_first_choice'] = df['flip_number'] == 1
    df['total_cards'] = 32

    start_configs = df[df['is_first_choice']][['user_id', 'trial', 'state_n', 'gains_left', 'cards_left']]
    start_configs = start_configs.rename(columns={
        'state_n': 'start_state',
        'gains_left': 'start_gains',
        'cards_left': 'start_cards'
    })

    df = df.merge(start_configs, on=['user_id', 'trial'], how='left')
    df['seq_len'] = df['state_n'] - df['start_state'] + 1

    # Векторизованная или оптимизированная версия для расчета объективной вероятности Haffke
    def calc_model3_prob(row):
        sl = int(row['seq_len'])
        sg = int(row['start_gains'])
        sc = int(row['start_cards'])
        
        if sl <= 0 or sc <= 0 or sg < sl: 
            return 0.0
            
        p = 1.0
        for i in range(sl):
            p *= (sg - i) / (sc - i)
        return p

    df['p_obj'] = df.apply(calc_model3_prob, axis=1)
    df_model = df[['user_id', 'trial', 'state_n', 'gains_left', 'cards_left', 'p_obj', 'choice', 'is_first_choice', 'loss_amount', 'gain_amount', 'loss_cards', 'total_cards']].copy()
    
    if 'round_id' not in df_model.columns:
        df_model['round_id'] = df_model.groupby(['user_id', 'trial']).ngroup()
        
    return df_model


# =============================================================================
# 2. ИЕРАРХИЧЕСКАЯ БАЙЕСОВСКАЯ МОДЕЛЬ (HBA)
# =============================================================================

class HaffkeHBAModel_Hot:
    """
    Иерархическая реализация Haffke & Hübner (2020) Model 3 (CCT-hot).
    Оптимизирована для HPC и стандартов Q1/Q2.
    """
    def __init__(self, data_df, scale_factor=100.0):
        self.df = data_df.copy().reset_index(drop=True)
        self.scale_factor = scale_factor
        
        # Масштабирование наград и штрафов
        self.df['gain_scaled'] = self.df['gain_amount'] / self.scale_factor
        self.df['loss_scaled'] = self.df['loss_amount'].abs() / self.scale_factor
        
        self.df['subj_idx'], self.subj_labels = pd.factorize(self.df['user_id'])
        self.n_subj = len(self.subj_labels)

    def build_model(self):
        subj_idx = self.df['subj_idx'].values
        p_obj = self.df['p_obj'].values
        state_n = self.df['state_n'].values
        choices = self.df['choice'].values
        
        gain_amt = self.df['gain_scaled'].values
        loss_amt = self.df['loss_scaled'].values

        x_current = (state_n - 1.0) * gain_amt
        x_next = state_n * gain_amt
        magnitude_loss = x_current + loss_amt 

        with pm.Model() as model:
            # 1. Гиперпараметры (Групповой уровень)
            mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=1.5)
            mu_lam = pm.Normal('mu_lam', mu=0, sigma=1.5)
            mu_delta = pm.Normal('mu_delta', mu=0, sigma=1.5)
            mu_eta = pm.Normal('mu_eta', mu=0, sigma=1.5)
            mu_theta = pm.Normal('mu_theta', mu=0, sigma=1.5)

            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)
            sigma_lam = pm.HalfNormal('sigma_lam', sigma=1)
            sigma_delta = pm.HalfNormal('sigma_delta', sigma=1)
            sigma_eta = pm.HalfNormal('sigma_eta', sigma=1)
            sigma_theta = pm.HalfNormal('sigma_theta', sigma=1)

            # 2. Индивидуальный уровень (Non-centered parameterization)
            z_alpha = pm.Normal('z_alpha', mu=0, sigma=1, shape=self.n_subj)
            z_lam = pm.Normal('z_lam', mu=0, sigma=1, shape=self.n_subj)
            z_delta = pm.Normal('z_delta', mu=0, sigma=1, shape=self.n_subj)
            z_eta = pm.Normal('z_eta', mu=0, sigma=1, shape=self.n_subj)
            z_theta = pm.Normal('z_theta', mu=0, sigma=1, shape=self.n_subj)

            # 3. Трансформация в строгие и безопасные границы
            alpha = pm.Deterministic('alpha', 3.0 * pm.math.invlogit(mu_alpha + sigma_alpha * z_alpha))
            lam = pm.Deterministic('lam', 10.0 * pm.math.invlogit(mu_lam + sigma_lam * z_lam))
            delta = pm.Deterministic('delta', 3.0 * pm.math.invlogit(mu_delta + sigma_delta * z_delta))
            eta = pm.Deterministic('eta', 3.0 * pm.math.invlogit(mu_eta + sigma_eta * z_eta))
            theta = pm.Deterministic('theta', 5.0 * pm.math.invlogit(mu_theta + sigma_theta * z_theta))

            a_i = alpha[subj_idx]
            l_i = lam[subj_idx]
            d_i = delta[subj_idx]
            e_i = eta[subj_idx]
            t_i = theta[subj_idx]

            # 4. Вероятности Prelec
            p_safe = pt.clip(p_obj, 1e-6, 1.0 - 1e-6)
            p_loss_val = pt.clip(1.0 - p_obj, 1e-6, 1.0 - 1e-6)
            
            pi_p_safe = pt.exp(-d_i * (-pt.log(p_safe)) ** e_i)
            pi_p_loss = pt.exp(-d_i * (-pt.log(p_loss_val)) ** e_i)

            # 5. Полезность исходов
            V_stop = x_current ** a_i
            V_take = (pi_p_safe * (x_next ** a_i)) - (l_i * pi_p_loss * (magnitude_loss ** a_i))

            # 6. Softmax и Likelihood
            logits = t_i * (V_take - V_stop)
            p_turn = pm.math.invlogit(logits)
            
            pm.Bernoulli('choice_obs', p=p_turn, observed=choices)
            
        return model

    def fit(self, draws=1500, tune=1000, chains=4, target_accept=0.95):
        print(f"[*] HBA Fit Haffke Model 3: Subj={self.n_subj}, Trials={len(self.df)}")
        model = self.build_model()
        with model:
            idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept, 
                              progressbar=False, return_inferencedata=True)
            # ИСПРАВЛЕНО: Заменено pm.compute_rnall на корректную функцию лог-правдоподобия
            pm.compute_log_likelihood(idata)
            loo = az.loo(idata, pointwise=True)
            # ИСПРАВЛЕНО: Заменен вызов .loo на .elpd_loo во избежание AttributeError
            print(f"[FIT METRICS] Haffke Model LOOIC: {loo.elpd_loo:.2f}")
        return idata, model, loo

    def posterior_predictive_check(self, idata, model, save_plot_path="hot_haffke_ppc.png"):
        print("\n[*] Запуск True HBA PPC (Haffke Model)...")
        with model:
            ppc = pm.sample_posterior_predictive(idata, extend_inferencedata=False, progressbar=False)
            
        y_sim = ppc.posterior_predictive['choice_obs'].values
        y_sim_flat = y_sim.reshape(-1, y_sim.shape[-1])
        choice_real = self.df['choice'].values
        subj_idx = self.df['subj_idx'].values
        round_ids = self.df['round_id'].values

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
                ax.set_title(f'Haffke PPC Timecourse: User {uid_label}')
                ax.set_xlabel('Trial Sequence')
                ax.set_ylabel('Turn Rate')
                ax.legend()
                plot_idx += 1

        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=300)
        plt.close()

        return pd.DataFrame(metrics)

    # ИСПРАВЛЕНО: Сигнатура теперь не принимает лишний idata, так как генерация идет автономно
    def parameter_recovery(self, idata, n_subjects=50, trials_per_subj=48, save_plot_path="hot_haffke_recovery.png"):
        print(f"\n[*] Запуск Empirical HBA Parameter Recovery Haffke Model (N={n_subjects})...")
        
        # 1. Извлекаем популяционные гиперпараметры (групповой уровень) из реального idata
        post_real = idata.posterior
        
        # Получаем усредненные по цепям и шагам значения гиперпараметров
        mu_alpha_m = post_real['mu_alpha'].mean().item()
        sigma_alpha_m = post_real['sigma_alpha'].mean().item()
        
        mu_lam_m = post_real['mu_lam'].mean().item()
        sigma_lam_m = post_real['sigma_lam'].mean().item()
        
        mu_delta_m = post_real['mu_delta'].mean().item()
        sigma_delta_m = post_real['sigma_delta'].mean().item()
        
        mu_eta_m = post_real['mu_eta'].mean().item()
        sigma_eta_m = post_real['sigma_eta'].mean().item()
        
        mu_theta_m = post_real['mu_theta'].mean().item()
        sigma_theta_m = post_real['sigma_theta'].mean().item()
        
        rng = np.random.default_rng(42)
        
        # 2. ЭМПИРИЧЕСКАЯ ГЕНЕРАЦИЯ: сэмплируем true_raw из нормальных распределений выборки
        raw_alpha = rng.normal(mu_alpha_m, sigma_alpha_m, n_subjects)
        raw_lam = rng.normal(mu_lam_m, sigma_lam_m, n_subjects)
        raw_delta = rng.normal(mu_delta_m, sigma_delta_m, n_subjects)
        raw_eta = rng.normal(mu_eta_m, sigma_eta_m, n_subjects)
        raw_theta = rng.normal(mu_theta_m, sigma_theta_m, n_subjects)
        
        # Переводим в истинные ограниченные когнитивные параметры (аналогично блоку трансформации модели)
        true_alpha = 3.0 * expit(raw_alpha)
        true_lam = 10.0 * expit(raw_lam)
        true_delta = 3.0 * expit(raw_delta)
        true_eta = 3.0 * expit(raw_eta)
        true_theta = 5.0 * expit(raw_theta)
        
        loss_amounts = [250, 750]
        gain_amounts = [10, 30]

        sim_data = []
        subj_labels = [f"sim_{i}" for i in range(n_subjects)]

        # Симуляция структуры раундов на основе эмпирических параметров виртуальных агентов
        for i, uid in enumerate(subj_labels):
            a, l, d, e, t = true_alpha[i], true_lam[i], true_delta[i], true_eta[i], true_theta[i]
            
            for trial in range(1, trials_per_subj + 1):
                state_n = 1
                gains_left = 31
                cards_left = 32
                start_state = 1
                start_gains = 31
                start_cards = 32
                
                current_loss = rng.choice(loss_amounts)
                current_gain = rng.choice(gain_amounts)
                
                c_gain_sc = current_gain / self.scale_factor
                c_loss_sc = current_loss / self.scale_factor

                while True:
                    seq_len = state_n - start_state + 1
                    p_obj = 1.0
                    for step in range(seq_len):
                        p_obj *= max(0, start_gains - step) / max(1, start_cards - step)

                    p_safe = np.clip(p_obj, 1e-6, 1.0 - 1e-6)
                    pi_p_safe = np.exp(-d * (-np.log(p_safe)) ** e)
                    p_loss = np.clip(1.0 - p_obj, 1e-6, 1.0 - 1e-6)
                    pi_p_loss = np.exp(-d * (-np.log(p_loss)) ** e)

                    x_current = (state_n - 1) * c_gain_sc
                    x_next = state_n * c_gain_sc
                    magnitude_loss = x_current + c_loss_sc

                    V_stop = x_current ** a
                    V_take = (pi_p_safe * (x_next ** a)) - (l * pi_p_loss * (magnitude_loss ** a))

                    prob_take = expit(t * (V_take - V_stop))
                    choice = rng.binomial(1, prob_take)

                    sim_data.append({
                        'user_id': uid, 'trial_number': trial, 'flip_number': state_n,
                        'loss_cards': 1, 'total_cards': 32,
                        'loss_amount': current_loss, 'gain_amount': current_gain,
                        'choice': choice
                    })

                    if choice == 1:
                        win_prob = gains_left / cards_left
                        if rng.random() < win_prob:
                            state_n += 1
                            gains_left -= 1
                            cards_left -= 1
                            if gains_left == 0: break
                        else:
                            break 
                    else:
                        break 

        df_sim = pd.DataFrame(sim_data)
        df_model_sim = prepare_haffke_input(df_sim)

        # Переобучаем иерархическую (HBA) модель на сгенерированном датасете виртуальных агентов
        rec_model = HaffkeHBAModel_Hot(df_model_sim, scale_factor=self.scale_factor)
        rec_idata, _, _ = rec_model.fit(draws=800, tune=800, chains=4, target_accept=0.90)
        
        post = rec_idata.posterior
        fit_alpha = post['alpha'].median(dim=("chain", "draw")).values
        fit_lam = post['lam'].median(dim=("chain", "draw")).values
        fit_delta = post['delta'].median(dim=("chain", "draw")).values
        fit_eta = post['eta'].median(dim=("chain", "draw")).values
        fit_theta = post['theta'].median(dim=("chain", "draw")).values

        df_results = pd.DataFrame({
            "user_id": subj_labels,
            "true_alpha": true_alpha, "fit_alpha": fit_alpha,
            "true_lam": true_lam, "fit_lam": fit_lam,
            "true_delta": true_delta, "fit_delta": fit_delta,
            "true_eta": true_eta, "fit_eta": fit_eta,
            "true_theta": true_theta, "fit_theta": fit_theta
        })

        metrics_list = []
        params = [("alpha", true_alpha, fit_alpha, 'alpha'), ("lam", true_lam, fit_lam, 'lam'), 
                  ("delta", true_delta, fit_delta, 'delta'), ("eta", true_eta, fit_eta, 'eta'),
                  ("theta", true_theta, fit_theta, 'theta')]

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle('Empirical HBA Parameter Recovery Haffke Model (True vs Fit)', fontsize=16)

        for ax, (p_name, true_v, fit_v, trace_name) in zip(axes, params):
            r_val = np.corrcoef(true_v, fit_v)[0, 1]
            r2_val = r2_score(true_v, fit_v)
            rmse = np.sqrt(mean_squared_error(true_v, fit_v))
            bias = np.mean(fit_v - true_v)
            
            # Извлекаем границы 95% HDI для вычисления Coverage
            hdi_bounds = az.hdi(rec_idata.posterior)[trace_name].values
            lower = hdi_bounds[:, 0]
            upper = hdi_bounds[:, 1]
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