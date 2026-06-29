import os
import matplotlib
# Серверный бэкенд для суперкомпьютера
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan
import arviz as az
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class VPPModelHBA:
    """
    Value-Plus-Perseveration (VPP) model с полным HBA на PyMC.
    Масштабирование наград: [-10, 10] (scale_factor = 100).
    Оптимизировано через pytensor.scan для работы с NUTS.
    """
    def __init__(self, data_df=None):
        self.data = None if data_df is None else data_df.copy().reset_index(drop=True)
        self.trace = None
        self.scale_factor = 100.0

    def _prepare_tensors(self, df):
        """Подготовка транспонированных тензоров [n_trials, n_users] для pytensor.scan"""
        users = df['user_id'].unique()
        n_subj = len(users)
        n_trials = df.groupby('user_id').size().max()
        
        choices = np.full((n_trials, n_subj), -1, dtype=np.int32)
        rewards = np.zeros((n_trials, n_subj), dtype=np.float64)
        mask = np.zeros((n_trials, n_subj), dtype=bool)
        
        for i, uid in enumerate(users):
            u_df = df[df['user_id'] == uid].sort_values('trial_number')
            n_t = len(u_df)
            choices[:n_t, i] = u_df['deck_num'].values
            rewards[:n_t, i] = u_df['points_earned'].values / self.scale_factor
            mask[:n_t, i] = True
            
        return choices, rewards, mask, users

    def fit(self, draws=1500, tune=1000, chains=4, cores=4):
        """Полная иерархическая подгонка на всех пользователях (NUTS)."""
        if self.data is None:
            raise ValueError("Данные не загружены.")

        choices, rewards, mask, self.users = self._prepare_tensors(self.data)
        n_subj = len(self.users)
        n_trials = choices.shape[0]

        print(f"[*] Инициализация HBA графа для {n_subj} субъектов (VPP Model)...")

        with pm.Model() as self.model:
            # ====================================================
            # Non-Centered Parameterization (NCP) для стабильности
            # ====================================================
            def ncp_01(name):
                mu = pm.Normal(f'mu_{name}', mu=0, sigma=1.5)
                sigma = pm.HalfNormal(f'sigma_{name}', sigma=1.0)
                raw = pm.Normal(f'{name}_raw', mu=0, sigma=1, shape=n_subj)
                return pm.Deterministic(name, pm.math.invlogit(mu + sigma * raw))

            phi = ncp_01('phi')
            alpha = ncp_01('alpha')
            w = ncp_01('w')
            K = ncp_01('K')

            def ncp_05(name):
                mu = pm.Normal(f'mu_{name}', mu=0, sigma=1.5)
                sigma = pm.HalfNormal(f'sigma_{name}', sigma=1.0)
                raw = pm.Normal(f'{name}_raw', mu=0, sigma=1, shape=n_subj)
                return pm.Deterministic(name, 5.0 * pm.math.invlogit(mu + sigma * raw))

            lam = ncp_05('lam')
            c = ncp_05('c')

            def ncp_m1_1(name):
                mu = pm.Normal(f'mu_{name}', mu=0, sigma=1.5)
                sigma = pm.HalfNormal(f'sigma_{name}', sigma=1.0)
                raw = pm.Normal(f'{name}_raw', mu=0, sigma=1, shape=n_subj)
                return pm.Deterministic(name, 2.0 * pm.math.invlogit(mu + sigma * raw) - 1.0)

            eps_pos = ncp_m1_1('eps_pos')
            eps_neg = ncp_m1_1('eps_neg')

            # ====================================================
            # PyTensor Scan Likelihood (Открывает NUTS сэмплер)
            # ====================================================
            choices_pt = pm.ConstantData('choices', choices)
            rewards_pt = pm.ConstantData('rewards', rewards)
            mask_pt = pm.ConstantData('mask', mask)
            
            # Защита от выхода индекса за пределы из-за паддинга -1
            choices_safe = pt.clip(choices_pt, 0, 3)

            def step_fn(choice_t, reward_t, mask_t, E_prev, P_prev, phi, alpha, lam, c, w, K, eps_pos, eps_neg):
                theta = (3.0 ** c) - 1.0
                
                # 1. Softmax Calculation
                V = w[:, None] * E_prev + (1.0 - w[:, None]) * P_prev
                v = theta[:, None] * V
                v = v - pt.max(v, axis=1, keepdims=True)
                ex = pt.exp(v)
                probs = ex / (pt.sum(ex, axis=1, keepdims=True) + 1e-16)

                # 2. Utility Calculation (Prospect Theory)
                u = pt.switch(reward_t >= 0, 
                              (pt.abs(reward_t) + 1e-12) ** alpha, 
                              -lam * ((pt.abs(reward_t) + 1e-12) ** alpha))

                idx = pt.arange(phi.shape[0])

                # 3. Update Expected Value (E)
                E_chosen = E_prev[idx, choice_t]
                E_new = pt.set_subtensor(E_prev[idx, choice_t], E_chosen + phi * (u - E_chosen))

                # 4. Update Perseveration (P)
                P_decay = K[:, None] * P_prev
                P_chosen = P_decay[idx, choice_t]
                eps_term = pt.switch(reward_t >= 0, eps_pos, eps_neg)
                P_new = pt.set_subtensor(P_decay[idx, choice_t], P_chosen + eps_term)

                # 5. Apply Mask (Ignore padding)
                E_next = pt.switch(mask_t[:, None], E_new, E_prev)
                P_next = pt.switch(mask_t[:, None], P_new, P_prev)

                return E_next, P_next, probs

            # Начальные состояния: E = 0, P = 0
            E_init = pt.zeros((n_subj, 4), dtype='float64')
            P_init = pt.zeros((n_subj, 4), dtype='float64')

            [_, _, probs_seq], _ = scan(
                fn=step_fn,
                sequences=[choices_safe, rewards_pt, mask_pt],
                outputs_info=[E_init, P_init, None],
                non_sequences=[phi, alpha, lam, c, w, K, eps_pos, eps_neg],
                strict=True
            )

            # Извлекаем вероятности фактически выбранных карт
            chosen_probs = probs_seq[pt.arange(n_trials)[:, None], pt.arange(n_subj)[None, :], choices_safe]
            
            # [LOOIC FIX]: Сохраняем log_likelihood по триалам
            logp_trial = pt.log(pt.clip(chosen_probs, 1e-12, 1.0)) * mask_pt
            pm.Deterministic('log_lik', logp_trial)
            
            pm.Potential('obs_logp', pt.sum(logp_trial))

            print("[*] Запуск MCMC сэмплирования (NUTS)...")
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, 
                                   target_accept=0.95, return_inferencedata=True, progressbar=False)
                                   
            self.trace.add_groups(
                {"log_likelihood": {"obs": self.trace.posterior["log_lik"]}}
            )

        return self.trace

    def _simulate_single(self, params, payoff_dict, n_trials=150, seed=None):
        """Симуляция одной сессии. Возвращает массив выбранных колод (0, 1, 2, 3)."""
        rng = np.random.RandomState(seed)
        phi, alpha, lam, c, w, K, eps_pos, eps_neg = params
        
        theta = np.power(3.0, c) - 1.0
        E = np.zeros(4, dtype=float)
        P = np.zeros(4, dtype=float)
        
        choices = np.zeros(n_trials, dtype=int)
        for t in range(n_trials):
            V = w * E + (1.0 - w) * P
            v = theta * V
            v = v - np.max(v)
            ex = np.exp(v)
            probs = ex / (np.sum(ex) + 1e-16)
            deck = int(rng.choice(4, p=probs))
            
            choices[t] = deck

            # Получаем награду и масштабируем ее
            raw_reward = float(rng.choice(payoff_dict[deck])) if payoff_dict else float(rng.normal(100 if deck in [0,1] else 50, 40))
            reward = raw_reward / self.scale_factor

            if reward >= 0:
                u = np.power(reward + 1e-12, alpha)
            else:
                u = -lam * np.power(np.abs(reward) + 1e-12, alpha)

            E[deck] = E[deck] + phi * (u - E[deck])
            P = K * P
            if reward >= 0: P[deck] += eps_pos
            else: P[deck] += eps_neg

        return choices

    def predictive_check(self, n_sims=100, block_size=30, save_path="igt_vpp_ppc.png", seed=42):
        print(f"[*] Запуск PPC (VPP Model, sims={n_sims})...")
        rng = np.random.default_rng(seed)
        df = self.data.copy()
        
        post = self.trace.posterior
        param_names = ['phi', 'alpha', 'lam', 'c', 'w', 'K', 'eps_pos', 'eps_neg']
        n_chains, n_draws = post.dims['chain'], post.dims['draw']
        total_samples = n_chains * n_draws
        
        sample_indices = rng.choice(total_samples, size=n_sims, replace=False if total_samples >= n_sims else True)
        
        max_trials = df.groupby('user_id').size().max()
        real_matrix = np.full((len(self.users), max_trials), np.nan)
        sim_matrix = np.full((n_sims, len(self.users), max_trials), np.nan)
        
        all_metrics = []

        for i, uid in enumerate(tqdm(self.users, desc="PPC Users")):
            udata = df[df['user_id'] == uid].sort_values('trial_number')
            real_choices = udata['deck_num'].values
            n_trials = len(real_choices)
            n_blocks = n_trials // block_size
            
            real_adv = np.isin(real_choices, [2, 3]).astype(int)
            real_matrix[i, :n_trials] = real_adv
            real_adv_rate = np.mean(real_adv)
            real_blocks = [np.mean(real_adv[b*block_size : (b+1)*block_size]) for b in range(n_blocks)]

            p_dict = {}
            for d in range(4):
                vals = udata.loc[udata['deck_num'] == d, 'points_earned'].values
                p_dict[d] = vals if len(vals) > 0 else [100.0 if d in [0,1] else 50.0]

            sim_adv_rates = []
            hit_rates = []
            sim_blocks_matrix = np.zeros((n_sims, n_blocks)) if n_blocks > 0 else np.zeros((n_sims, 1))

            for s in range(n_sims):
                idx_s = sample_indices[s]
                c_idx, d_idx = idx_s // n_draws, idx_s % n_draws
                
                p_vals = [post[p][c_idx, d_idx, i].values.item() for p in param_names]
                sim_choices = self._simulate_single(p_vals, p_dict, n_trials, seed=int(rng.integers(1e9)))
                sim_adv = np.isin(sim_choices, [2, 3]).astype(int)
                
                sim_matrix[s, i, :n_trials] = sim_adv
                sim_adv_rates.append(np.mean(sim_adv))
                hit_rates.append(np.mean(sim_choices == real_choices))
                
                if n_blocks > 0:
                    for b in range(n_blocks):
                        sim_blocks_matrix[s, b] = np.mean(sim_adv[b*block_size : (b+1)*block_size])

            sim_blocks_mean = np.mean(sim_blocks_matrix, axis=0)

            ppp = np.mean(np.array(sim_adv_rates) >= real_adv_rate)
            r2 = r2_score(real_blocks, sim_blocks_mean) if n_blocks > 1 else np.nan
            rmse = np.sqrt(mean_squared_error(real_blocks, sim_blocks_mean)) if n_blocks > 0 else np.nan
            mae = mean_absolute_error(real_blocks, sim_blocks_mean) if n_blocks > 0 else np.nan
            msd = np.mean((np.array(real_blocks) - sim_blocks_mean)**2) if n_blocks > 0 else np.nan

            all_metrics.append({
                'user_id': uid, 'ppp': ppp, 'R2': r2, 'RMSE': rmse,
                'MAE': mae, 'MSD': msd, 'Hit_Rate': np.mean(hit_rates)
            })

        metrics_df = pd.DataFrame(all_metrics)
        
        # Визуализация PPC Timecourse
        real_timecourse = np.nanmean(real_matrix, axis=0)
        sim_timecourses = np.nanmean(sim_matrix, axis=1)
        sim_mean_tc = np.nanmean(sim_timecourses, axis=0)
        
        valid_trials = ~np.isnan(sim_mean_tc)
        if np.any(valid_trials):
            hdi_tc = az.hdi(sim_timecourses[:, valid_trials], hdi_prob=0.95)
            
            plt.figure(figsize=(10, 5))
            plt.plot(np.where(valid_trials)[0], real_timecourse[valid_trials], color='black', linewidth=2, label='Observed')
            plt.plot(np.where(valid_trials)[0], sim_mean_tc[valid_trials], color='blue', linestyle='--', linewidth=2, label='VPP Model Mean')
            plt.fill_between(np.where(valid_trials)[0], hdi_tc[:, 0], hdi_tc[:, 1], color='blue', alpha=0.15, label='95% HDI')
            plt.xlabel('Trial')
            plt.ylabel('P(Advantageous Choices)')
            plt.title('PPC Timecourse (VPP Model)')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.2)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return metrics_df

    def parameter_recovery(self, n_subjects=40, n_trials=150, seed0=42):
        """Full HBA Parameter Recovery с расчетом Coverage и генерацией синтетической популяции."""
        rng = np.random.RandomState(seed0)
        param_names = ['phi', 'alpha', 'lam', 'c', 'w', 'K', 'eps_pos', 'eps_neg']
        
        true_params = []
        sim_data_rows = []
        
        print(f"[*] Генерация Recovery данных (N={n_subjects})...")
        for i in range(n_subjects):
            # Генерация истинных параметров из правдоподобных распределений
            p_true = [
                rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), rng.uniform(0.5, 4.5), 
                rng.uniform(0.5, 4.5), rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 
                rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)
            ]
            true_params.append(p_true)
            
            pseudo_payoffs = {
                0: [100, -150], 1: [100, -1150], 2: [50, -50], 3: [50, -200]
            }
            
            choices = self._simulate_single(p_true, pseudo_payoffs, n_trials, seed=seed0+i)
            
            for t, choice in enumerate(choices):
                # Назначаем реальную выплату на основе симулированного выбора
                reward = float(rng.choice(pseudo_payoffs[choice]))
                sim_data_rows.append({
                    'user_id': f'sim_{i}', 'trial_number': t+1, 
                    'deck_num': choice, 'points_earned': reward
                })
                
        sim_df = pd.DataFrame(sim_data_rows)
        
        print("[*] Подгонка HBA на восстановленных данных (Слепой тест)...")
        recovery_model = VPPModelHBA(sim_df)
        rec_trace = recovery_model.fit(draws=1000, tune=1000, chains=2, cores=2)
        
        rec_post = rec_trace.posterior
        true_arr = np.array(true_params)
        
        # Извлечение восстановленных (fit) средних из постериора
        fit_arr = np.column_stack([rec_post[p].mean(dim=["chain", "draw"]).values for p in param_names])

        # ==========================================
        # Вычисление Recovery метрик
        # ==========================================
        metrics_rows = []
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes = axes.flatten()

        hdi_data = az.hdi(rec_trace, hdi_prob=0.95)

        for i, c in enumerate(param_names):
            y_true = true_arr[:, i]
            y_fit = fit_arr[:, i]
            
            # [ARVIZ FIX]: Надежное извлечение HDI границ
            hdi_bounds = hdi_data[c].values
            hdi_lower = hdi_bounds[:, 0]
            hdi_upper = hdi_bounds[:, 1]
            
            coverage = np.mean((y_true >= hdi_lower) & (y_true <= hdi_upper))
            
            r_val = pearsonr(y_true, y_fit)[0]
            r2_val = r2_score(y_true, y_fit)
            bias = np.mean(y_fit - y_true)
            rmse = np.sqrt(mean_squared_error(y_true, y_fit))
            
            metrics_rows.append({
                'Parameter': c, 'Pearson_r': r_val, 'R2': r2_val, 
                'Bias': bias, 'RMSE': rmse, 'Coverage': coverage
            })

            ax = axes[i]
            sns.scatterplot(x=y_true, y=y_fit, ax=ax, alpha=0.7, edgecolor='k')
            ax.vlines(y_true, hdi_lower, hdi_upper, color='gray', alpha=0.3, zorder=0)
            
            ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
            ax.set_title(f"{c}\nr={r_val:.2f}, R²={r2_val:.2f}\nRMSE={rmse:.2f}, Cov={coverage*100:.0f}%")
            ax.set_xlabel("True Value")
            ax.set_ylabel("Recovered Posterior Mean")

        plt.tight_layout()
        plt.savefig("vpp_hba_recovery.png", dpi=300)
        plt.close(fig)

        df_metrics = pd.DataFrame(metrics_rows)
        
        # Формируем таблицу с индивидуальными параметрами
        df_params = pd.DataFrame(np.hstack((true_arr, fit_arr)), 
                                 columns=[f'true_{p}' for p in param_names] + [f'fit_{p}' for p in param_names])
        df_params['user_id'] = [f"sim_{i}" for i in range(n_subjects)]

        print("[✓] Parameter Recovery завершен.")
        return df_metrics, df_params