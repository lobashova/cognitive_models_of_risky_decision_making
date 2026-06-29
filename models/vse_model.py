import os
import matplotlib
matplotlib.use('Agg')  # Отключаем интерактивный бэкенд для суперкомпьютера

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan  # ИСПРАВЛЕНИЕ: правильный импорт scan
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from scipy.special import expit
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class VSEModelHBA:
    """
    Value plus Sequential Exploration (VSE) model (Ligneul, 2019)
    Реализация HBA с использованием PyMC 5+ и оптимизированного PyTensor графа.
    """
    def __init__(self, data_df):
        self.data = data_df.copy().sort_values(['user_id', 'trial_number']).reset_index(drop=True)
        
        self.data['gain_scaled'] = self.data['payout'] / 100.0
        self.data['loss_scaled'] = np.abs(self.data['penalty']) / 100.0
        
        self.users = self.data['user_id'].unique()
        self.n_users = len(self.users)
        self.max_trials = self.data.groupby('user_id').size().max()
        
        # Создаем матрицы (max_trials, n_users)
        self.gains_mat = np.zeros((self.max_trials, self.n_users))
        self.losses_mat = np.zeros((self.max_trials, self.n_users))
        self.choices_mat = np.zeros((self.max_trials, self.n_users), dtype=np.int32)
        self.mask = np.zeros((self.max_trials, self.n_users), dtype=bool) # Маска валидных попыток
        
        for i, user in enumerate(self.users):
            user_df = self.data[self.data['user_id'] == user]
            n_t = len(user_df)
            self.gains_mat[:n_t, i] = user_df['gain_scaled'].values
            self.losses_mat[:n_t, i] = user_df['loss_scaled'].values
            self.choices_mat[:n_t, i] = user_df['deck_num'].values
            self.mask[:n_t, i] = True

    def build_model(self):
        with pm.Model() as model:
            # --- ГРУППОВЫЕ ГИПЕРПАРАМЕТРЫ (Оставляем как у вас) ---
            mu_theta = pm.Normal('mu_theta', mu=0, sigma=1.5)
            mu_delta = pm.Normal('mu_delta', mu=0, sigma=1.5)
            mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=1.5)
            mu_phi   = pm.Normal('mu_phi',   mu=0, sigma=3.0)
            mu_c     = pm.Normal('mu_c',     mu=0, sigma=1.5)

            sigma_theta = pm.HalfNormal('sigma_theta', sigma=1.0)
            sigma_delta = pm.HalfNormal('sigma_delta', sigma=1.0)
            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1.0)
            sigma_phi   = pm.HalfNormal('sigma_phi',   sigma=3.0)
            sigma_c     = pm.HalfNormal('sigma_c',     sigma=1.0)

            z_theta = pm.Normal('z_theta', mu=0, sigma=1, shape=self.n_users)
            z_delta = pm.Normal('z_delta', mu=0, sigma=1, shape=self.n_users)
            z_alpha = pm.Normal('z_alpha', mu=0, sigma=1, shape=self.n_users)
            z_phi   = pm.Normal('z_phi',   mu=0, sigma=1, shape=self.n_users)
            z_c     = pm.Normal('z_c',     mu=0, sigma=1, shape=self.n_users)

            # --- ИНДИВИДУАЛЬНЫЕ ПАРАМЕТРЫ размерностью (n_users,) ---
            theta = pm.Deterministic('theta', pm.math.invlogit(mu_theta + z_theta * sigma_theta))
            delta = pm.Deterministic('delta', pm.math.invlogit(mu_delta + z_delta * sigma_delta))
            alpha = pm.Deterministic('alpha', pm.math.invlogit(mu_alpha + z_alpha * sigma_alpha))
            phi   = pm.Deterministic('phi',   mu_phi + z_phi * sigma_phi)
            c     = pm.Deterministic('c',     5.0 * pm.math.invlogit(mu_c + z_c * sigma_c))

            # Индексы строк для продвинутой индексации
            row_idx = pt.arange(self.n_users)

            # --- ВЕКТОРИЗОВАННЫЙ РЕКУРСИВНЫЙ ГРАФ ---
            def step(gain_t, loss_t, choice_t, mask_t, exploit_prev, explore_prev, th, dl, al, ph, c_val):
                # На входе: массивы (n_users,) и матрицы (n_users, 4)
                
                v_val = (gain_t ** th) - (loss_t ** th) # Форма: (n_users,)
                
                # Применяем забывание (decay)
                exploit_new = dl[:, None] * exploit_prev
                explore_new = explore_prev + al[:, None] * (ph[:, None] - explore_prev)
                
                # Обновляем ценности только выбранных колод
                exploit_new = pt.set_subtensor(
                    exploit_new[row_idx, choice_t], 
                    exploit_new[row_idx, choice_t] + v_val
                )
                explore_new = pt.set_subtensor(explore_new[row_idx, choice_t], 0.0)
                
                # Защита от паддинга: если это фиктивный триал для пользователя (короче остальных), 
                # мы "замораживаем" состояние (возвращаем prev)
                exploit_new = pt.switch(mask_t[:, None], exploit_new, exploit_prev)
                explore_new = pt.switch(mask_t[:, None], explore_new, explore_prev)
                
                # Softmax
                V = exploit_new + explore_new
                beta = (3.0 ** c_val) - 1.0
                v_scaled = V * beta[:, None]
                v_scaled = v_scaled - pt.max(v_scaled, axis=1, keepdims=True) # Стабилизация
                probs = pt.exp(v_scaled) / pt.sum(pt.exp(v_scaled), axis=1, keepdims=True)
                
                # Извлекаем вероятность только той колоды, которую выбрали (для Likelihood)
                prob_chosen = probs[row_idx, choice_t]
                
                return exploit_new, explore_new, prob_chosen

            # Начальные состояния
            exploit_0 = pt.zeros((self.n_users, 4), dtype='float64')
            explore_0 = pt.ones((self.n_users, 4), dtype='float64')

            [_, _, probs_chosen_seq], _ = scan(
                fn=step,
                sequences=[
                    pt.as_tensor_variable(self.gains_mat),
                    pt.as_tensor_variable(self.losses_mat),
                    pt.as_tensor_variable(self.choices_mat),
                    pt.as_tensor_variable(self.mask)
                ],
                non_sequences=[theta, delta, alpha, phi, c],
                outputs_info=[exploit_0, explore_0, None],
                strict=True
            )

            # --- CUSTOM LIKELIHOOD ---
            # Избегаем pm.Categorical, так как он будет считать likelihood и для паддинг-нулей
            # probs_chosen_seq имеет размерность (max_trials, n_users)
            
            # 1. Применяем логарифм (с защитой от 0)
            logp = pt.log(probs_chosen_seq + 1e-12)
            
            # 2. Оставляем только реальные триалы с помощью маски. 
            # Плоский массив идеально подходит для расчета LOOIC в ArviZ!
            logp_flat = logp[self.mask]
            
            # Добавляем для LOOIC
            pm.Deterministic('log_lik', logp_flat)
            
            # Передаем итоговую сумму в лог-вероятность модели
            pm.Potential('obs', pt.sum(logp_flat))

        return model

    def fit_hba(self, draws=1000, tune=1000, chains=4, cores=4):
        self.model = self.build_model()
        with self.model:
            print("[*] Запуск MCMC сэмплирования...")
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, return_inferencedata=True, progressbar=False)
            
            # --- ИСПРАВЛЕНИЕ 1: Копируем log_lik в нужную для ArviZ группу ---
            self.trace.add_groups(
                {"log_likelihood": {"log_lik": self.trace.posterior["log_lik"]}}
            )
            
            # --- ИСПРАВЛЕНИЕ 2: Обращаемся к правильному атрибуту (elpd_loo вместо loo) ---
            loo_obj = az.loo(self.trace, var_name="log_lik")
            self.looic = loo_obj.elpd_loo 
            print(f"[FIT METRICS] LOOIC: {self.looic:.2f}")
            
        return self.trace

    @staticmethod
    def calc_se_index(choices):
        se = np.zeros(len(choices))
        for i in range(len(choices) - 3):
            if len(set(choices[i:i+4])) == 4:
                se[i+3] = 1 
        return se

    def posterior_predictive_check(self, n_sims=150, block_size=30, save_path="igt_vse_ppc.png", seed=42):
        print(f"\n=== Запуск Posterior Predictive Check (n_sims={n_sims}) ===")
        if getattr(self, 'trace', None) is None:
            raise ValueError("Сначала запустите fit_hba()!")

        post = self.trace.posterior
        rng = np.random.default_rng(seed)
        
        n_chains = post.sizes['chain']
        n_draws = post.sizes['draw']
        total_samples = n_chains * n_draws
        
        # Индексы сэмплов для симуляции
        sample_indices = rng.choice(total_samples, size=n_sims, replace=False if total_samples >= n_sims else True)
        
        max_trials = self.max_trials
        
        # Матрицы для хранения результатов
        real_matrix = np.full((self.n_users, max_trials), np.nan)
        sim_matrix = np.full((n_sims, self.n_users, max_trials), np.nan)
        
        all_metrics = []
        
        # Извлекаем параметры (плоские массивы)
        th_flat = post['theta'].values.reshape(total_samples, self.n_users)
        dl_flat = post['delta'].values.reshape(total_samples, self.n_users)
        al_flat = post['alpha'].values.reshape(total_samples, self.n_users)
        ph_flat = post['phi'].values.reshape(total_samples, self.n_users)
        cv_flat = post['c'].values.reshape(total_samples, self.n_users)

        for i, user in enumerate(tqdm(self.users, desc="PPC Users")):
            user_df = self.data[self.data['user_id'] == user]
            n_trials = len(user_df)
            
            user_real = user_df['deck_num'].values
            gains = user_df['gain_scaled'].values
            losses = user_df['loss_scaled'].values
            
            n_blocks = n_trials // block_size
            
            real_adv = np.isin(user_real, [2, 3]).astype(int)
            real_matrix[i, :n_trials] = real_adv
            real_adv_rate = np.mean(real_adv)
            real_blocks = [np.mean(real_adv[b*block_size : (b+1)*block_size]) for b in range(n_blocks)]
            
            sim_adv_rates = []
            hit_rates = []
            sim_blocks_matrix = np.zeros((n_sims, n_blocks)) if n_blocks > 0 else np.zeros((n_sims, 1))
            
            # Симулируем агента для каждого извлеченного MCMC-сэмпла
            for s_idx, sample_idx in enumerate(sample_indices):
                th = th_flat[sample_idx, i]
                dl = dl_flat[sample_idx, i]
                al = al_flat[sample_idx, i]
                ph = ph_flat[sample_idx, i]
                cv = cv_flat[sample_idx, i]
                
                exploit = np.zeros(4)
                explore = np.ones(4)
                sim_ch = np.zeros(n_trials, dtype=int)
                
                for t in range(n_trials):
                    V = exploit + explore
                    beta = (3.0 ** cv) - 1.0
                    v_scaled = V * beta
                    v_scaled -= np.max(v_scaled)
                    probs = np.exp(v_scaled) / np.sum(np.exp(v_scaled))
                    
                    ch = rng.choice(4, p=probs)
                    sim_ch[t] = ch
                    
                    # Получаем награду из реального расписания
                    g = gains[t]
                    l = losses[t]
                    v_val = (g ** th) - (l ** th)
                    
                    # Обновляем ценности
                    exploit = dl * exploit
                    exploit[ch] += v_val
                    
                    explore = explore + al * (ph - explore)
                    explore[ch] = 0.0
                
                sim_adv = np.isin(sim_ch, [2, 3]).astype(int)
                sim_matrix[s_idx, i, :n_trials] = sim_adv
                sim_adv_rates.append(np.mean(sim_adv))
                hit_rates.append(np.mean(sim_ch == user_real))
                
                if n_blocks > 0:
                    for b in range(n_blocks):
                        sim_blocks_matrix[s_idx, b] = np.mean(sim_adv[b*block_size : (b+1)*block_size])
            
            sim_blocks_mean = np.mean(sim_blocks_matrix, axis=0)
            
            # Расчет метрик
            ppp = np.mean(np.array(sim_adv_rates) >= real_adv_rate)
            r2 = r2_score(real_blocks, sim_blocks_mean) if n_blocks > 1 else np.nan
            rmse = np.sqrt(mean_squared_error(real_blocks, sim_blocks_mean)) if n_blocks > 0 else np.nan
            mae = mean_absolute_error(real_blocks, sim_blocks_mean) if n_blocks > 0 else np.nan
            msd = np.mean((np.array(real_blocks) - sim_blocks_mean)**2) if n_blocks > 0 else np.nan
            
            all_metrics.append({
                'user_id': user,
                'ppp': ppp, 'R2': r2, 'RMSE': rmse,
                'MAE': mae, 'MSD': msd, 'Hit_Rate': np.mean(hit_rates)
            })

        metrics_df = pd.DataFrame(all_metrics)
        
        # Визуализация PPC Timecourse
        real_timecourse = np.nanmean(real_matrix, axis=0)
        sim_timecourses = np.nanmean(sim_matrix, axis=1)
        sim_mean_tc = np.nanmean(sim_timecourses, axis=0)
        
        valid_trials = ~np.isnan(sim_mean_tc)
        if np.any(valid_trials):
            # Вычисляем 95% HDI через перцентили (чтобы избежать проблем с размерностями arviz)
            hdi_lower = np.percentile(sim_timecourses[:, valid_trials], 2.5, axis=0)
            hdi_upper = np.percentile(sim_timecourses[:, valid_trials], 97.5, axis=0)
            
            plt.figure(figsize=(10, 5))
            plt.plot(np.where(valid_trials)[0], real_timecourse[valid_trials], color='black', linewidth=2, label='Observed')
            plt.plot(np.where(valid_trials)[0], sim_mean_tc[valid_trials], color='blue', linestyle='--', linewidth=2, label='VSE Model Mean')
            plt.fill_between(np.where(valid_trials)[0], hdi_lower, hdi_upper, color='blue', alpha=0.15, label='95% HDI')
            plt.xlabel('Trial')
            plt.ylabel('P(Advantageous Choices)')
            plt.title('PPC Timecourse (VSE Model)')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.2)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"[✓] PPC завершен. Ср. R2: {metrics_df['R2'].mean():.3f}, Hit Rate: {metrics_df['Hit_Rate'].mean():.3f}")
        return metrics_df

    def parameter_recovery(self, n_subjects=50):
        print(f"\n=== Запуск Parameter Recovery (N={n_subjects}) ===")
        post = self.trace.posterior
        
        rng = np.random.default_rng(42)
        
        # Генерируем истинные параметры из групповых распределений
        true_theta = expit(rng.normal(post['mu_theta'].mean().item(), post['sigma_theta'].mean().item(), n_subjects))
        true_delta = expit(rng.normal(post['mu_delta'].mean().item(), post['sigma_delta'].mean().item(), n_subjects))
        true_alpha = expit(rng.normal(post['mu_alpha'].mean().item(), post['sigma_alpha'].mean().item(), n_subjects))
        true_phi   = rng.normal(post['mu_phi'].mean().item(), post['sigma_phi'].mean().item(), n_subjects)
        true_c     = 5.0 * expit(rng.normal(post['mu_c'].mean().item(), post['sigma_c'].mean().item(), n_subjects))

        true_params = np.column_stack([true_theta, true_delta, true_alpha, true_phi, true_c])
        
        sim_data_list = []
        
        # ИСПРАВЛЕНИЕ: берем первый user_id из списка self.users вместо несуществующего user_idx
        first_user = self.users[0]
        template_df = self.data[self.data['user_id'] == first_user].copy().reset_index(drop=True)
        
        trials_per_subj = len(template_df)
        gains = template_df['gain_scaled'].values
        losses = template_df['loss_scaled'].values
        
        for i in tqdm(range(n_subjects), desc="Simulating subjects"):
            th, dl, al, ph, cv = true_theta[i], true_delta[i], true_alpha[i], true_phi[i], true_c[i]
            
            exploit = np.zeros(4)
            explore = np.ones(4)
            choices = []
            
            for t in range(trials_per_subj):
                V = exploit + explore
                beta = (3.0 ** cv) - 1.0
                v_scaled = V * beta
                v_scaled -= np.max(v_scaled)
                probs = np.exp(v_scaled) / np.sum(np.exp(v_scaled))
                
                ch = rng.choice(4, p=probs)
                choices.append(ch)
                
                g = gains[t]
                l = losses[t]
                v_val = (g ** th) - (l ** th)
                
                exploit = dl * exploit
                exploit[ch] += v_val
                
                explore = explore + al * (ph - explore)
                explore[ch] = 0.0
                
            df_subj = template_df.copy()
            df_subj['user_id'] = f"sim_{i}"
            df_subj['deck_num'] = choices
            sim_data_list.append(df_subj)
            
        sim_full_df = pd.concat(sim_data_list, ignore_index=True)
        
        print("Подгонка HBA на симулированных данных...")
        sim_model = VSEModelHBA(sim_full_df)
        sim_trace = sim_model.fit_hba(draws=1000, tune=1000, chains=4, cores=4)
        
        sim_post = sim_trace.posterior
        param_names = ['theta', 'delta', 'alpha', 'phi', 'c']
        
        fit_params = np.column_stack([
            sim_post[p].mean(dim=['chain', 'draw']).values for p in param_names
        ])

        hdi_data = az.hdi(sim_trace, hdi_prob=0.95)
        results = []
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        for i, p_name in enumerate(param_names):
            t_val = true_params[:, i]
            f_val = fit_params[:, i]
            
            hdi_bounds = hdi_data[p_name].values
            hdi_lower = hdi_bounds[:, 0]
            hdi_upper = hdi_bounds[:, 1]
            
            coverage = np.mean((t_val >= hdi_lower) & (t_val <= hdi_upper))
            r, _ = pearsonr(t_val, f_val)
            r2 = r2_score(t_val, f_val)
            rmse = np.sqrt(mean_squared_error(t_val, f_val))
            bias = np.mean(f_val - t_val)
            
            results.append({
                'Parameter': p_name, 'r': r, 'R2': r2, 
                'RMSE': rmse, 'Bias': bias, 'Coverage': coverage
            })
            
            axes[i].scatter(t_val, f_val, alpha=0.7, color='teal')
            min_v, max_v = min(t_val.min(), f_val.min()), max(t_val.max(), f_val.max())
            axes[i].plot([min_v, max_v], [min_v, max_v], 'k--')
            axes[i].set_title(f"{p_name}\nr={r:.2f}, R2={r2:.2f}\nRMSE={rmse:.2f}, Cov={coverage*100:.0f}%")
            axes[i].set_xlabel("True")
            axes[i].set_ylabel("Fitted")

        plt.tight_layout()
        plt.savefig("vse_parameter_recovery.png", dpi=300)
        plt.close()
        
        metrics_df = pd.DataFrame(results)
        
        recovery_details = pd.DataFrame(
            np.column_stack([true_params, fit_params]), 
            columns=[f"True_{p}" for p in param_names] + [f"Fit_{p}" for p in param_names]
        )
        recovery_details['user_id'] = [f"sim_{i}" for i in range(n_subjects)]
        
        print("[✓] Parameter Recovery завершен.")
        return metrics_df, recovery_details