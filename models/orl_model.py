import os
import matplotlib
# [HPC FIX]: Серверный бэкенд для суперкомпьютера
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan
import arviz as az
from scipy.stats import pearsonr, norm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class ORLModel:
    """
    Outcome-Representation Learning (ORL) model (Haines et al., 2018).
    Векторизованная реализация с использованием pytensor.scan (NUTS-совместимая).
    """
    def __init__(self, data_by_subject, scale_factor=100.0):
        self.data_by_subject = data_by_subject
        self.trace = None
        self.scale_factor = scale_factor
        self.users = list(self.data_by_subject.keys())
        self.n_users = len(self.users)
        
    def _prepare_tensors(self):
        """Создает транспонированные матрицы [n_trials, n_users] с паддингом."""
        max_trials = max(len(df) for df in self.data_by_subject.values())
        
        # -1 заменяем на 0 для безопасности тензорных индексов, маска скроет эти шаги
        choices = np.zeros((max_trials, self.n_users), dtype=np.int32)
        outcomes = np.zeros((max_trials, self.n_users), dtype=np.float64)
        mask = np.zeros((max_trials, self.n_users), dtype=bool)
        
        for i, uid in enumerate(self.users):
            df = self.data_by_subject[uid]
            n_t = len(df)
            choices[:n_t, i] = df['deck_num'].values
            outcomes[:n_t, i] = df['points_earned'].values / self.scale_factor
            mask[:n_t, i] = True
            
        return choices, outcomes, mask

    def fit_hierarchical(self, draws=1000, tune=1000, chains=4, cores=4):
        if not self.data_by_subject:
            raise ValueError("Данные субъектов не переданы.")
            
        choices, outcomes, mask = self._prepare_tensors()
        
        print(f"[*] Запуск HBA ORL (NUTS) для {self.n_users} субъектов...")

        with pm.Model() as model:
            # 1. Гиперпараметры
            mu_Arew = pm.Normal('mu_Arew', 0, 1.5)
            sigma_Arew = pm.HalfNormal('sigma_Arew', 1.0)
            mu_Apun = pm.Normal('mu_Apun', 0, 1.5)
            sigma_Apun = pm.HalfNormal('sigma_Apun', 1.0)
            mu_K0 = pm.Normal('mu_K0', 0, 1.5)
            sigma_K0 = pm.HalfNormal('sigma_K0', 1.0)
            
            mu_bF = pm.Normal('mu_bF', 0, 1.5)
            sigma_bF = pm.HalfNormal('sigma_bF', 1.0)
            mu_bP = pm.Normal('mu_bP', 0, 1.5)
            sigma_bP = pm.HalfNormal('sigma_bP', 1.0)

            # 2. Индивидуальные параметры (NCP)
            Arew_pr = pm.Normal('Arew_pr', 0, 1, shape=self.n_users)
            Apun_pr = pm.Normal('Apun_pr', 0, 1, shape=self.n_users)
            K0_pr = pm.Normal('K0_pr', 0, 1, shape=self.n_users)
            bF_pr = pm.Normal('bF_pr', 0, 1, shape=self.n_users)
            bP_pr = pm.Normal('bP_pr', 0, 1, shape=self.n_users)

            Arew = pm.Deterministic('Arew', pm.math.invprobit(mu_Arew + sigma_Arew * Arew_pr))
            Apun = pm.Deterministic('Apun', pm.math.invprobit(mu_Apun + sigma_Apun * Apun_pr))
            K0 = pm.Deterministic('K0', 5.0 * pm.math.invprobit(mu_K0 + sigma_K0 * K0_pr))
            bF = pm.Deterministic('bF', mu_bF + sigma_bF * bF_pr)
            bP = pm.Deterministic('bP', mu_bP + sigma_bP * bP_pr)

            # 3. Данные
            choices_pt = pm.ConstantData('choices', choices)
            outcomes_pt = pm.ConstantData('outcomes', outcomes)
            mask_pt = pm.ConstantData('mask', mask)

            # 4. Функция шага (Scan)
            def step_fn(choice_t, x_t, mask_t, EV_prev, EF_prev, PS_prev, Arew, Apun, K0, bF, bP):
                K = (3.0 ** K0) - 1.0
                C = 3.0
                
                # Сигнал исхода
                s_t = pt.switch(x_t > 0, 1.0, pt.switch(x_t < 0, -1.0, 0.0))
                
                # Softmax V
                V = EV_prev + bF[:, None] * EF_prev + bP[:, None] * PS_prev
                V = V - pt.max(V, axis=1, keepdims=True)
                ex = pt.exp(V)
                probs = ex / (pt.sum(ex, axis=1, keepdims=True) + 1e-12)
                
                idx = pt.arange(self.n_users)
                
                # Разделение Learning Rates
                A_chosen = pt.switch(x_t >= 0, Arew, Apun)
                A_unchosen = pt.switch(x_t >= 0, Apun, Arew)
                
                # Обновление EF для невыбранных колод
                EF_new = EF_prev + A_unchosen[:, None] * ((-s_t[:, None] / C) - EF_prev)
                
                # Обновление EV и EF для выбранной колоды
                EV_chosen = EV_prev[idx, choice_t]
                EF_chosen = EF_prev[idx, choice_t]
                
                EV_new_c = EV_chosen + A_chosen * (x_t - EV_chosen)
                EF_new_c = EF_chosen + A_chosen * (s_t - EF_chosen)
                
                # Запись обновлений выбранной колоды в матрицу
                EV_new = pt.set_subtensor(EV_prev[idx, choice_t], EV_new_c)
                EF_new = pt.set_subtensor(EF_new[idx, choice_t], EF_new_c)
                
                # Обновление PS
                denom = 1.0 + K
                PS_new = PS_prev / denom[:, None]
                PS_new = pt.set_subtensor(PS_new[idx, choice_t], 1.0 / denom)
                
                # Применение маски паддинга
                EV_next = pt.switch(mask_t[:, None], EV_new, EV_prev)
                EF_next = pt.switch(mask_t[:, None], EF_new, EF_prev)
                PS_next = pt.switch(mask_t[:, None], PS_new, PS_prev)
                
                return EV_next, EF_next, PS_next, probs

            # Инициализация состояний
            EV_init = pt.zeros((self.n_users, 4), dtype='float64')
            EF_init = pt.zeros((self.n_users, 4), dtype='float64')
            PS_init = pt.zeros((self.n_users, 4), dtype='float64')

            [_, _, _, probs_seq], _ = scan(
                fn=step_fn,
                sequences=[choices_pt, outcomes_pt, mask_pt],
                outputs_info=[EV_init, EF_init, PS_init, None],
                non_sequences=[Arew, Apun, K0, bF, bP],
                strict=True
            )

            # 5. Функция правдоподобия (LOOIC Fix)
            chosen_probs = probs_seq[pt.arange(choices_pt.shape[0])[:, None], pt.arange(self.n_users)[None, :], choices_pt]
            logp_trial = pt.log(pt.clip(chosen_probs, 1e-12, 1.0)) * mask_pt
            
            pm.Deterministic('log_lik', logp_trial)
            pm.Potential('obs_logp', pt.sum(logp_trial))

            # Сэмплирование (Без progressbar для чистых логов на HPC)
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                                   target_accept=0.95, return_inferencedata=True, progressbar=False)
            
            self.trace.add_groups({"log_likelihood": {"obs": self.trace.posterior["log_lik"]}})
            
        return self.trace

    def get_parameters_df(self):
        if self.trace is None: return None
        post = self.trace.posterior
        
        results = []
        for i, uid in enumerate(self.users):
            results.append({
                'user_id': uid,
                'igt_orl_Arew': float(post['Arew'].mean(dim=['chain', 'draw'])[i]),
                'igt_orl_Apun': float(post['Apun'].mean(dim=['chain', 'draw'])[i]),
                'igt_orl_K0': float(post['K0'].mean(dim=['chain', 'draw'])[i]),
                'igt_orl_betaF': float(post['bF'].mean(dim=['chain', 'draw'])[i]),
                'igt_orl_betaP': float(post['bP'].mean(dim=['chain', 'draw'])[i])
            })
        return pd.DataFrame(results)

    def simulate(self, params, n_trials, seed=None):
        rng = np.random.default_rng(seed)
        Arew, Apun, K0, bF, bP = params
        K = (3.0 ** np.clip(K0, 0, 5)) - 1.0
        C = 3.0
        EV, EF, PS = np.zeros(4), np.zeros(4), np.zeros(4)
        rows = []
        
        for t in range(1, n_trials + 1):
            V = EV + bF * EF + bP * PS
            V_stable = V - np.max(V)
            probs = np.exp(V_stable) / (np.sum(np.exp(V_stable)) + 1e-12)
            deck = int(rng.choice(4, p=probs))
            
            if deck == 0:   x_scaled = 1.0 if rng.random() > 0.5 else -1.5
            elif deck == 1: x_scaled = 1.0 if rng.random() > 0.2 else -5.25 
            elif deck == 2: x_scaled = 0.5 if rng.random() > 0.5 else 0.0
            else:           x_scaled = 0.5 if rng.random() > 0.2 else -0.75
                
            rows.append({'trial_number': t, 'deck_num': deck, 'points_earned': x_scaled * self.scale_factor})
            
            s = 1.0 if x_scaled > 0 else (-1.0 if x_scaled < 0 else 0.0)
            
            if x_scaled >= 0:
                EV[deck] += Arew * (x_scaled - EV[deck])
                EF[deck] += Arew * (s - EF[deck])
                for j in range(4): 
                    if j != deck: EF[j] += Apun * ((-s / C) - EF[j])
            else:
                EV[deck] += Apun * (x_scaled - EV[deck])
                EF[deck] += Apun * (s - EF[deck])
                for j in range(4): 
                    if j != deck: EF[j] += Arew * ((-s / C) - EF[j])
                    
            PS /= (1.0 + K)
            PS[deck] = 1.0 / (1.0 + K)
            
        return pd.DataFrame(rows)

    def posterior_predictive_check(self, n_sims=100, block_size=30, save_path="igt_orl_ppc.png", seed=42):
        print(f"[*] Запуск PPC (ORL Model)...")
        post = self.trace.posterior
        n_chains, n_draws = post.sizes['chain'], post.sizes['draw']
        total_samples = n_chains * n_draws
        
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(total_samples, size=n_sims, replace=False if total_samples >= n_sims else True)

        max_trials = max(len(df) for df in self.data_by_subject.values())
        real_matrix = np.full((self.n_users, max_trials), np.nan)
        sim_matrix = np.full((n_sims, self.n_users, max_trials), np.nan)
        
        ppc_results = []
        for i, uid in enumerate(tqdm(self.users, desc="PPC Users")):
            real_df = self.data_by_subject[uid].reset_index(drop=True)
            real_choices = real_df['deck_num'].values
            n_trials = len(real_choices)
            n_blocks = int(np.ceil(n_trials / block_size))
            
            real_adv = np.isin(real_choices, [2, 3]).astype(int)
            real_matrix[i, :n_trials] = real_adv
            real_adv_rate = np.mean(real_adv)
            real_blocks = [np.mean(real_adv[b*block_size : min((b+1)*block_size, n_trials)]) for b in range(n_blocks)]

            sim_adv_rates, hit_rates = [], []
            sim_blocks_matrix = np.zeros((n_sims, n_blocks))

            for s in range(n_sims):
                idx_s = sample_indices[s]
                c_idx, d_idx = idx_s // n_draws, idx_s % n_draws
                
                p = [float(post[v][c_idx, d_idx, i]) for v in ['Arew', 'Apun', 'K0', 'bF', 'bP']]
                sim_df = self.simulate(p, n_trials, seed=int(rng.integers(1e9)))
                sim_choices = sim_df['deck_num'].values
                sim_adv = np.isin(sim_choices, [2, 3]).astype(int)
                
                sim_matrix[s, i, :n_trials] = sim_adv
                sim_adv_rates.append(np.mean(sim_adv))
                hit_rates.append(np.mean(sim_choices == real_choices))
                
                for b in range(n_blocks):
                    sim_blocks_matrix[s, b] = np.mean(sim_adv[b*block_size : min((b+1)*block_size, n_trials)])

            sim_blocks_mean = np.mean(sim_blocks_matrix, axis=0)

            ppp = np.mean(np.array(sim_adv_rates) >= real_adv_rate)
            r2 = r2_score(real_blocks, sim_blocks_mean) if n_blocks > 1 else np.nan
            rmse = np.sqrt(mean_squared_error(real_blocks, sim_blocks_mean))
            mae = mean_absolute_error(real_blocks, sim_blocks_mean)
            msd = np.mean((np.array(real_blocks) - sim_blocks_mean)**2)

            ppc_results.append({
                'user_id': uid, 'ppp': ppp, 'R2': r2, 'RMSE': rmse,
                'MAE': mae, 'MSD': msd, 'Hit_Rate': np.mean(hit_rates)
            })

        ppc_df = pd.DataFrame(ppc_results)
        
        # Визуализация PPC
        real_timecourse = np.nanmean(real_matrix, axis=0)
        sim_timecourses = np.nanmean(sim_matrix, axis=1)
        sim_mean_tc = np.nanmean(sim_timecourses, axis=0)
        
        valid_trials = ~np.isnan(sim_mean_tc)
        if np.any(valid_trials):
            hdi_tc = az.hdi(sim_timecourses[:, valid_trials], hdi_prob=0.95)
            
            plt.figure(figsize=(10, 5))
            plt.plot(np.where(valid_trials)[0], real_timecourse[valid_trials], color='black', linewidth=2, label='Observed')
            plt.plot(np.where(valid_trials)[0], sim_mean_tc[valid_trials], color='blue', linestyle='--', linewidth=2, label='ORL Model Mean')
            plt.fill_between(np.where(valid_trials)[0], hdi_tc[:, 0], hdi_tc[:, 1], color='blue', alpha=0.15, label='95% HDI')
            plt.xlabel('Trial')
            plt.ylabel('P(Advantageous Choices)')
            plt.title('PPC Timecourse (ORL Model)')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.2)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return ppc_df

    def parameter_recovery(self, n_subjects=30, n_trials=150, draws=500, tune=500):
        if self.trace is None: 
            raise ValueError("Сначала обучите модель на реальных данных для извлечения гиперпараметров!")
            
        print(f"[*] Запуск Иерархического Parameter Recovery для ORL (N={n_subjects})...")
        post = self.trace.posterior
        rng = np.random.default_rng(42)
        param_names = ['Arew', 'Apun', 'K0', 'bF', 'bP']
        
        true_params = {}
        for p in param_names:
            mu = post[f'mu_{p}'].mean().item()
            sigma = post[f'sigma_{p}'].mean().item()
            
            # [MATH FIX]: Используем norm.cdf вместо expit для соответствия invprobit
            if p in ['Arew', 'Apun']:
                true_params[p] = norm.cdf(rng.normal(mu, sigma, n_subjects))
            elif p in ['bF', 'bP']:
                true_params[p] = rng.normal(mu, sigma, n_subjects)
            else: # K0
                true_params[p] = 5.0 * norm.cdf(rng.normal(mu, sigma, n_subjects))

        sim_data_dict = {}
        for i in range(n_subjects):
            p_true = [true_params[p][i] for p in param_names]
            sim_df = self.simulate(p_true, n_trials, seed=int(rng.integers(1e9)))
            sim_df['user_id'] = f"sim_{i}"
            sim_data_dict[f"sim_{i}"] = sim_df
            
        print("[*] Подгонка HBA на синтетических данных...")
        recovery_model = ORLModel(sim_data_dict, scale_factor=self.scale_factor)
        rec_trace = recovery_model.fit_hierarchical(draws=draws, tune=tune, chains=2, cores=2)
        rec_post = rec_trace.posterior
        
        true_arr = np.column_stack([true_params[p] for p in param_names])
        fit_arr = np.column_stack([rec_post[p].mean(dim=["chain", "draw"]).values for p in param_names])
        hdi_data = az.hdi(rec_trace, hdi_prob=0.95)
        
        metrics_list = []
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        for idx, p in enumerate(param_names):
            t_v = true_arr[:, idx]
            f_v = fit_arr[:, idx]
            
            # [ARVIZ FIX]: Надежное извлечение индексов HDI
            hdi_bounds = hdi_data[p].values
            hdi_lower, hdi_upper = hdi_bounds[:, 0], hdi_bounds[:, 1]
            
            coverage = np.mean((t_v >= hdi_lower) & (t_v <= hdi_upper))
            r_val = pearsonr(t_v, f_v)[0]
            r2_val = r2_score(t_v, f_v)
            bias = np.mean(f_v - t_v)
            rmse = np.sqrt(mean_squared_error(t_v, f_v))
            
            metrics_list.append({'Parameter': p, 'r': r_val, 'R2': r2_val, 'Bias': bias, 'RMSE': rmse, 'Coverage': coverage})
            
            axes[idx].scatter(t_v, f_v, alpha=0.6, color='coral')
            axes[idx].vlines(t_v, hdi_lower, hdi_upper, color='gray', alpha=0.3)
            min_v, max_v = min(t_v.min(), f_v.min()), max(t_v.max(), f_v.max())
            axes[idx].plot([min_v, max_v], [min_v, max_v], 'k--')
            axes[idx].set_title(f"{p}\nr={r_val:.2f}, Cov={coverage*100:.0f}%")
            
        plt.tight_layout()
        plt.savefig('igt_orl_recovery.png', dpi=300)
        plt.close()
        
        metrics_df = pd.DataFrame(metrics_list)
        params_df = pd.DataFrame(np.hstack((true_arr, fit_arr)), 
                                 columns=[f'true_{p}' for p in param_names] + [f'fit_{p}' for p in param_names])
        params_df['user_id'] = [f"sim_{i}" for i in range(n_subjects)]
        
        return metrics_df, params_df