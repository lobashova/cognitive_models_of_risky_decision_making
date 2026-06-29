import os
import matplotlib
# [HPC FIX]: Отключаем интерактивный бэкенд ДО импорта pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan
import arviz as az
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.special import expit
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class EVModelHBA:
    """
    Hierarchical Bayesian Expectancy-Valence (EV) model for IGT.
    Адаптировано для PyMC v5 с использованием PyTensor Scan (NUTS sampler).
    """

    def __init__(self, data_df, scale_factor=100.0):
        self.scale_factor = scale_factor
        self.data = data_df.copy().reset_index(drop=True)
        self.user_ids = self.data['user_id'].unique()
        self.n_users = len(self.user_ids)
        self.user2idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.data['user_idx'] = self.data['user_id'].map(self.user2idx)
        
        # Подготовка тензоров для PyMC
        self.max_trials = self.data.groupby('user_id').size().max()
        
        self.choices_mat = np.zeros((self.max_trials, self.n_users), dtype=np.int32)
        self.rewards_mat = np.zeros((self.max_trials, self.n_users), dtype=np.float64)
        self.losses_mat = np.zeros((self.max_trials, self.n_users), dtype=np.float64)
        self.trials_mat = np.zeros((self.max_trials, self.n_users), dtype=np.float64)
        self.mask_mat = np.zeros((self.max_trials, self.n_users), dtype=bool)

        for u_idx, uid in enumerate(self.user_ids):
            u_data = self.data[self.data['user_id'] == uid].sort_values('trial_number')
            n_t = len(u_data)
            
            # Извлекаем данные и СРАЗУ МАСШТАБИРУЕМ
            pts = u_data['points_earned'].values.astype(float) / self.scale_factor
            rew = np.where(pts > 0, pts, 0.0)
            loss = np.where(pts < 0, pts, 0.0)
            
            self.choices_mat[:n_t, u_idx] = u_data['deck_num'].values.astype(int)
            self.rewards_mat[:n_t, u_idx] = rew
            self.losses_mat[:n_t, u_idx] = loss
            
            trials = u_data['trial_number'].values if 'trial_number' in u_data.columns else np.arange(1, n_t + 1)
            self.trials_mat[:n_t, u_idx] = trials
            self.mask_mat[:n_t, u_idx] = True

    @staticmethod
    def simulate_subject(w, a, c, n_trials=150, user_id="sim_0", scale_factor=100.0):
        """Симуляция с каноничным расписанием исходов IGT и масштабированием (scale_factor)."""
        rng = np.random.default_rng()
        Ev = np.zeros(4, dtype=float)
        rows = []

        for t in range(1, n_trials + 1):
            # Softmax
            theta = np.clip((t / 10.0) ** c, 1e-8, 10.0)
            v = Ev * theta
            v = v - np.max(v)
            p = np.exp(v) / np.sum(np.exp(v))
            
            deck = rng.choice(4, p=p)
            
            # Каноничное расписание IGT (масштабированное на scale_factor)
            if deck == 0:   # Deck A
                rew = 100.0 / scale_factor; loss = -250.0 / scale_factor if rng.random() < 0.5 else 0.0
            elif deck == 1: # Deck B
                rew = 100.0 / scale_factor; loss = -625.0 / scale_factor if rng.random() < 0.2 else 0.0
            elif deck == 2: # Deck C
                rew = 50.0 / scale_factor;  loss = -50.0 / scale_factor if rng.random() < 0.5 else 0.0
            else:           # Deck D
                rew = 50.0 / scale_factor;  loss = -125.0 / scale_factor if rng.random() < 0.2 else 0.0
                
            pts = rew + loss
            is_adv = 1 if deck in [2, 3] else 0
            
            rows.append({
                'user_id': user_id,
                'trial_number': t,
                'deck_num': deck,
                'points_earned': pts * scale_factor, # Оригинальные баллы для датасета
                'is_adv': is_adv
            })
            
            # Обновление ожиданий (масштабированные значения)
            V = (1.0 - w) * rew + w * loss
            Ev[deck] = (1.0 - a) * Ev[deck] + a * V

        return pd.DataFrame(rows)

    def build_model(self):
        """Создание иерархической модели в PyMC с Non-centered параметризацией."""
        coords = {"subject": self.user_ids, "trial": range(self.max_trials)}
        
        with pm.Model(coords=coords) as model:
            # 1. Гиперпараметры (групповой уровень)
            mu_w_raw = pm.Normal("mu_w_raw", mu=0, sigma=1.5)
            sigma_w_raw = pm.HalfNormal("sigma_w_raw", sigma=1.0)
            z_w = pm.Normal("z_w", mu=0, sigma=1, dims="subject")
            w = pm.Deterministic("w", pm.math.invlogit(mu_w_raw + sigma_w_raw * z_w), dims="subject") # [0, 1]

            mu_a_raw = pm.Normal("mu_a_raw", mu=0, sigma=1.5)
            sigma_a_raw = pm.HalfNormal("sigma_a_raw", sigma=1.0)
            z_a = pm.Normal("z_a", mu=0, sigma=1, dims="subject")
            a = pm.Deterministic("a", pm.math.invlogit(mu_a_raw + sigma_a_raw * z_a), dims="subject") # [0, 1]

            mu_c_raw = pm.Normal("mu_c_raw", mu=0, sigma=1.5)
            sigma_c_raw = pm.HalfNormal("sigma_c_raw", sigma=1.0)
            z_c = pm.Normal("z_c", mu=0, sigma=1, dims="subject")
            c = pm.Deterministic("c", 6.0 * pm.math.invlogit(mu_c_raw + sigma_c_raw * z_c) - 3.0, dims="subject") # [-3, 3]

            # 2. Данные
            choices = pm.ConstantData("choices", self.choices_mat)
            rewards = pm.ConstantData("rewards", self.rewards_mat)
            losses = pm.ConstantData("losses", self.losses_mat)
            trials = pm.ConstantData("trials", self.trials_mat)
            mask = pm.ConstantData("mask", self.mask_mat)

            # 3. Функция шага (PyTensor) для обновления Expectancy
            def step_fn(choice_t, rew_t, loss_t, t_val, Ev_prev, w, a, c):
                # [MCMC SAFE FIX]: Клиппинг theta защищает NUTS от Divergences при больших c и t
                theta = pt.clip((t_val / 10.0) ** c, 1e-8, 10.0)
                v = Ev_prev * theta[:, None] # pt.newaxis для надежности (эквивалент dimshuffle(0, 'x'))
                v = v - pt.max(v, axis=1, keepdims=True)
                ex = pt.exp(v)
                probs = ex / (pt.sum(ex, axis=1, keepdims=True) + 1e-12)
                
                # Обновление полезности и ожиданий
                V = (1.0 - w) * rew_t + w * loss_t
                idx = pt.arange(w.shape[0])
                
                Ev_chosen = Ev_prev[idx, choice_t]
                Ev_updated = (1.0 - a) * Ev_chosen + a * V
                Ev_new = pt.set_subtensor(Ev_prev[idx, choice_t], Ev_updated)
                
                return Ev_new, probs

            # 4. Scan цикл по времени
            Ev_init = pt.zeros((self.n_users, 4))
            
            [Ev_seq, probs_seq], _ = scan(
                fn=step_fn,
                sequences=[choices, rewards, losses, trials],
                outputs_info=[Ev_init, None],
                non_sequences=[w, a, c],
                strict=True
            )

            # Извлекаем вероятности фактически выбранных карт
            chosen_probs = probs_seq[
                pt.arange(self.max_trials)[:, None], 
                pt.arange(self.n_users)[None, :], 
                choices
            ]
            
            # [LOOIC FIX]: Сохраняем log_likelihood по триалам для az.loo()
            logp_trial = pt.log(pt.clip(chosen_probs, 1e-12, 1.0)) * mask
            pm.Deterministic('log_lik', logp_trial)
            
            # Целевая функция (Likelihood)
            pm.Potential("obs_logp", pt.sum(logp_trial))

        return model

    def fit(self, draws=1000, tune=1000, chains=4, cores=4):
        """Запуск MCMC."""
        model = self.build_model()
        with model:
            print(f"[*] Запуск HBA для {self.n_users} субъектов...")
            self.trace = pm.sample(
                draws=draws, tune=tune, chains=chains, cores=cores,
                target_accept=0.95, return_inferencedata=True, progressbar=False
            )
            # [LOOIC FIX]: Интегрируем log_lik в группу log_likelihood
            self.trace.add_groups(
                {"log_likelihood": {"obs": self.trace.posterior["log_lik"]}}
            )
        return self.trace

    def posterior_predictive_check(self, trace, block_size=30, n_sims=100, save_path="igt_ev_ppc.png", seed=42):
        """Стандартизированный индивидуальный и групповой Posterior Predictive Check."""
        print("[*] Запуск индивидуального PPC (EV Model)...")
        
        post = trace.posterior
        n_chains = post.dims['chain']
        n_draws = post.dims['draw']
        total_samples = n_chains * n_draws
        
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(total_samples, size=n_sims, replace=False if total_samples >= n_sims else True)
        
        max_trials = self.max_trials
        real_matrix = np.full((len(self.user_ids), max_trials), np.nan)
        sim_matrix = np.full((n_sims, len(self.user_ids), max_trials), np.nan)
        
        all_metrics = []

        for i, uid in enumerate(tqdm(self.user_ids, desc="PPC Users")):
            u_data = self.data[self.data['user_id'] == uid].sort_values('trial_number')
            real_choices = u_data['deck_num'].values
            n_trials = len(real_choices)
            n_blocks = n_trials // block_size
            
            real_adv = np.isin(real_choices, [2, 3]).astype(int)
            real_matrix[i, :n_trials] = real_adv
            real_blocks = [np.mean(real_adv[b*block_size : (b+1)*block_size]) for b in range(n_blocks)]
            real_adv_rate = np.mean(real_adv)
            
            sim_adv_rates = []
            hit_rates = []
            sim_block_matrix = np.zeros((n_sims, n_blocks))

            for s in range(n_sims):
                idx_s = sample_indices[s]
                c = idx_s // n_draws
                d = idx_s % n_draws
                
                w_sim = post['w'][c, d, i].values.item()
                a_sim = post['a'][c, d, i].values.item()
                c_sim = post['c'][c, d, i].values.item()
                
                sim_df = self.simulate_subject(w_sim, a_sim, c_sim, n_trials, uid, scale_factor=self.scale_factor)
                sim_choices = sim_df['deck_num'].values
                sim_adv = np.isin(sim_choices, [2, 3]).astype(int)
                
                sim_matrix[s, i, :n_trials] = sim_adv
                sim_adv_rates.append(np.mean(sim_adv))
                hit_rates.append(np.mean(sim_choices == real_choices))
                
                for b in range(n_blocks):
                    sim_block_matrix[s, b] = np.mean(sim_adv[b*block_size : (b+1)*block_size])

            sim_blocks_mean = np.mean(sim_block_matrix, axis=0)
            
            ppp = np.mean(np.array(sim_adv_rates) >= real_adv_rate)
            r2 = r2_score(real_blocks, sim_blocks_mean) if n_blocks > 1 else np.nan
            rmse = np.sqrt(mean_squared_error(real_blocks, sim_blocks_mean))
            mae = mean_absolute_error(real_blocks, sim_blocks_mean)
            msd = np.mean((np.array(real_blocks) - sim_blocks_mean)**2)
            
            all_metrics.append({
                'user_id': uid, 'ppp': ppp, 'R2': r2, 'RMSE': rmse, 
                'MAE': mae, 'MSD': msd, 'Hit_Rate': np.mean(hit_rates)
            })

        metrics_df = pd.DataFrame(all_metrics)
        
        # Визуализация PPC Timecourse
        real_timecourse = np.nanmean(real_matrix, axis=0)
        sim_timecourses = np.nanmean(sim_matrix, axis=1) # [n_sims, max_trials]
        sim_mean_tc = np.nanmean(sim_timecourses, axis=0)
        hdi_tc = az.hdi(sim_timecourses, hdi_prob=0.95)
        
        plt.figure(figsize=(10, 5))
        plt.plot(real_timecourse, color='black', linewidth=2, label='Observed')
        plt.plot(sim_mean_tc, color='blue', linestyle='--', linewidth=2, label='EV Model Mean')
        plt.fill_between(range(max_trials), hdi_tc[:, 0], hdi_tc[:, 1], color='blue', alpha=0.15, label='95% HDI')
        plt.xlabel('Trial')
        plt.ylabel('P(Advantageous Choices)')
        plt.title('PPC Timecourse (EV Model)')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.2)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[✓] PPC завершен. Mean R2={metrics_df['R2'].mean():.3f}, Mean HR={metrics_df['Hit_Rate'].mean():.3f}")
        return metrics_df

    # [METHOD FIX]: Замена на обычный метод (не @classmethod), чтобы использовать self.scale_factor
    def parameter_recovery(self, original_trace, n_subjects=30, n_trials=150, draws=1000, tune=1000):
        print(f"[*] Запуск Иерархического Parameter Recovery для EV (N={n_subjects})...")
        post = original_trace.posterior
        rng = np.random.default_rng(42)
        param_names = ['w', 'a', 'c']
        
        # Эмпирическая генерация
        true_w = expit(rng.normal(post['mu_w_raw'].mean().item(), post['sigma_w_raw'].mean().item(), n_subjects))
        true_a = expit(rng.normal(post['mu_a_raw'].mean().item(), post['sigma_a_raw'].mean().item(), n_subjects))
        
        # [BOUNDS FIX]: Границы для c в модели [-3.0, 3.0], формула должна быть `* 6.0 - 3.0`
        true_c = expit(rng.normal(post['mu_c_raw'].mean().item(), post['sigma_c_raw'].mean().item(), n_subjects)) * 6.0 - 3.0 

        synth_data = []
        for i in range(n_subjects):
            df = self.simulate_subject(true_w[i], true_a[i], true_c[i], n_trials, user_id=f"sim_{i}", scale_factor=self.scale_factor)
            synth_data.append(df)
            
        full_synth_df = pd.concat(synth_data, ignore_index=True)
        
        # Фитинг HBA
        recovery_model = EVModelHBA(full_synth_df, scale_factor=self.scale_factor)
        rec_trace = recovery_model.fit(draws=draws, tune=tune, chains=4, cores=4)
        
        # Метрики
        fit_w = rec_trace.posterior['w'].mean(dim=["chain", "draw"]).values
        fit_a = rec_trace.posterior['a'].mean(dim=["chain", "draw"]).values
        fit_c = rec_trace.posterior['c'].mean(dim=["chain", "draw"]).values
        
        hdi_data = az.hdi(rec_trace, hdi_prob=0.95)
        
        metrics_list = []
        true_vals = [true_w, true_a, true_c]
        fit_vals = [fit_w, fit_a, fit_c]
        
        for idx, p in enumerate(param_names):
            t_v = true_vals[idx]
            f_v = fit_vals[idx]
            
            # [ARVIZ FIX]: Индексация HDI без `.sel(hdi="lower")` для максимальной надежности
            hdi_bounds = hdi_data[p].values
            hdi_lower = hdi_bounds[:, 0]
            hdi_upper = hdi_bounds[:, 1]
            
            coverage = np.mean((t_v >= hdi_lower) & (t_v <= hdi_upper))
            r_val = pearsonr(t_v, f_v)[0]
            r2_val = r2_score(t_v, f_v)
            bias = np.mean(f_v - t_v)
            rmse = np.sqrt(mean_squared_error(t_v, f_v))
            
            metrics_list.append({'Parameter': p, 'r': r_val, 'R2': r2_val, 'Bias': bias, 'RMSE': rmse, 'Coverage': coverage})

        metrics_df = pd.DataFrame(metrics_list)
        params_df = pd.DataFrame({
            'user_id': [f"sim_{i}" for i in range(n_subjects)],
            'true_w': true_w, 'fit_w': fit_w,
            'true_a': true_a, 'fit_a': fit_a,
            'true_c': true_c, 'fit_c': fit_c
        })
        
        return metrics_df, params_df
