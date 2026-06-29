import matplotlib
matplotlib.use('Agg')  # Строго для суперкомпьютера

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
from scipy.stats import truncnorm
from sklearn.metrics import r2_score
from tqdm import tqdm
import warnings
from numba import njit
from scipy.stats import truncnorm, pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


warnings.filterwarnings("ignore")

def plot_ppc_timecourse(df, sim_matrix, model_name, window=5):
    """
    Строит график Group-Level PPC Timecourse.
    df: Оригинальный DataFrame с колонками 'trial_number', 'pumps', 'popped'.
    sim_matrix: Матрица симуляций формы (n_sims, общая_длина_df).
    model_name: Название модели для заголовка и сохранения файла.
    """
    work_df = df[['trial_number', 'pumps', 'popped']].copy()
    
    # Добавляем все симуляции в DataFrame
    for s in range(sim_matrix.shape[0]):
        work_df[f'sim_{s}'] = sim_matrix[s, :]
        
    # НАУЧНЫЙ СТАНДАРТ: Считаем Adjusted Pumps (только для нелопнувших шаров)
    # Заменяем значения на NaN там, где шар лопнул, чтобы исключить их из среднего
    work_df.loc[work_df['popped'] == True, 'pumps'] = np.nan
    for s in range(sim_matrix.shape[0]):
        work_df.loc[work_df['popped'] == True, f'sim_{s}'] = np.nan

    # Агрегируем среднее по каждому триалу (усредняем по всем пользователям)
    real_grouped = work_df.groupby('trial_number')['pumps'].mean()
    sim_grouped = work_df.drop(columns=['pumps', 'popped']).groupby('trial_number').mean()

    # Считаем среднее и 95% интервал наивысшей плотности (HDI) для симуляций
    sim_mean = sim_grouped.mean(axis=1)
    sim_hdi_low = np.percentile(sim_grouped, 2.5, axis=1)
    sim_hdi_high = np.percentile(sim_grouped, 97.5, axis=1)

    # Применяем скользящее среднее (Moving Average) для сглаживания кривой
    real_ma = real_grouped.rolling(window, min_periods=1).mean()
    sim_mean_ma = sim_mean.rolling(window, min_periods=1).mean()
    sim_low_ma = pd.Series(sim_hdi_low, index=sim_grouped.index).rolling(window, min_periods=1).mean()
    sim_high_ma = pd.Series(sim_hdi_high, index=sim_grouped.index).rolling(window, min_periods=1).mean()

    # Отрисовка
    plt.figure(figsize=(10, 6))
    plt.plot(real_ma.index, real_ma, color='black', linewidth=2, label='Real Adjusted Data')
    plt.plot(sim_mean_ma.index, sim_mean_ma, color='blue', linestyle='--', label='Simulated Mean')
    plt.fill_between(sim_mean_ma.index, sim_low_ma, sim_high_ma, color='blue', alpha=0.2, label='95% HDI')

    plt.title(f'PPC Timecourse: {model_name} (MA Window = {window})', fontsize=14)
    plt.xlabel('Trial Number', fontsize=12)
    plt.ylabel('Average Adjusted Pumps', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Сохраняем на диск (без plt.show(), чтобы скрипт не падал на суперкомпьютере)
    filename = f"{model_name.replace(' ', '_').lower()}_timecourse.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f" -> График Timecourse сохранен: {filename}")
    
# ==========================================
# 1. Оптимизированное ядро Numba
# ==========================================

@njit(fastmath=True)
def _compute_pburst_numba(psi, xi, cum_successes, cum_pumps, eps=1e-12):
    weight = np.exp(-xi * cum_pumps)
    if cum_pumps <= 0.0:
        P_emp = 0.0
    else:
        P_emp = (cum_pumps - cum_successes) / (cum_pumps + eps)
    
    val = weight * psi + (1.0 - weight) * P_emp
    if val < eps: val = eps
    if val > 1.0 - eps: val = 1.0 - eps
    return val

@njit(fastmath=True)
def loglik_user_numba(psi, xi, rho, tau, lam, pumps_arr, popped_arr, r_scaled):
    eps = 1e-12
    n_trials = len(pumps_arr)
    ll_arr = np.zeros(n_trials, dtype=np.float64) # <-- ТЕПЕРЬ ЭТО МАССИВ
    cum_successes = 0.0
    cum_pumps = 0.0

    for i in range(n_trials):
        ll_trial = 0.0
        pumps = pumps_arr[i]
        popped = popped_arr[i]
        p_burst_k = _compute_pburst_numba(psi, xi, cum_successes, cum_pumps, eps)

        for j in range(1, pumps + 1):
            gain_term = (1.0 - p_burst_k) * r_scaled
            loss_amt = (j - 1.0) * r_scaled
            loss_term = p_burst_k * lam * loss_amt
            variance_term = rho * p_burst_k * (1.0 - p_burst_k) * ((r_scaled + lam * loss_amt) ** 2)
            U_pump = gain_term - loss_term + variance_term
            
            if U_pump > 20.0: U_pump = 20.0
            if U_pump < -20.0: U_pump = -20.0
            p_pump = 1.0 / (1.0 + np.exp(-tau * U_pump))
            
            if j < pumps:
                ll_trial += np.log(p_pump + eps)
            else:
                if popped:
                    ll_trial += np.log(p_pump + eps)
                else:
                    gain_term_next = (1.0 - p_burst_k) * r_scaled
                    loss_amt_next = j * r_scaled 
                    loss_term_next = p_burst_k * lam * loss_amt_next
                    var_term_next = rho * p_burst_k * (1.0 - p_burst_k) * ((r_scaled + lam * loss_amt_next) ** 2)
                    U_next = gain_term_next - loss_term_next + var_term_next
                    
                    if U_next > 20.0: U_next = 20.0
                    if U_next < -20.0: U_next = -20.0
                    p_next = 1.0 / (1.0 + np.exp(-tau * U_next))
                    
                    ll_trial += np.log(1.0 - p_next + eps)
                break

        ll_arr[i] = ll_trial # <-- ЗАПИСЫВАЕМ ПОТРИАЛЬНО

        successes = (pumps - 1.0) if popped else float(pumps)
        cum_successes += successes
        cum_pumps += float(pumps)

    return ll_arr


# ==========================================
# 2. Основной класс модели HBA
# ==========================================

class EWMVModel_HBA:
    def __init__(self, r=1.0, max_pumps=64):
        self.r_raw = float(r)
        self.max_pumps = int(max_pumps)
        # 1. МАСШТАБИРОВАНИЕ НАГРАД (Сжатие в диапазон [-10, 10])
        # Если max_pumps=64 и r=1, max loss = 63. Делим на 10 -> max loss 6.3 (Идеально для Softmax)
        self.r_scaled = self.r_raw / 10.0 

    def fit_hba(self, all_data, draws=1000, tune=1000, chains=4, cores=4):
        user_ids = all_data['user_id'].unique()
        n_users = len(user_ids)
        r_val = self.r_scaled
        
        pumps_list = [all_data[all_data['user_id'] == uid].sort_values('trial_number')['pumps'].values.astype(np.int64) for uid in user_ids]
        popped_list = [all_data[all_data['user_id'] == uid].sort_values('trial_number')['popped'].values.astype(np.bool_) for uid in user_ids]

        @as_op(itypes=[pt.dvector, pt.dvector, pt.dvector, pt.dvector, pt.dvector], otypes=[pt.dvector])
        def loglik_all_users(psi_arr, xi_arr, rho_arr, tau_arr, lam_arr):
            all_lls = []
            for i in range(n_users):
                if not (0.0 < psi_arr[i] < 1.0 and xi_arr[i] >= 0 and tau_arr[i] >= 0 and lam_arr[i] >= 0):
                    # Штраф в виде массива той же длины, что и триалы пользователя
                    all_lls.append(np.full(len(pumps_list[i]), -1e10))
                else:
                    user_ll_arr = loglik_user_numba(
                        psi_arr[i], xi_arr[i], rho_arr[i], tau_arr[i], lam_arr[i],
                        pumps_list[i], popped_list[i], r_val
                    )
                    all_lls.append(user_ll_arr)
            # Склеиваем в один длинный вектор по всем триалам всех людей!
            return np.concatenate(all_lls)

        with pm.Model() as hba_model:
            # Априорные распределения (гиперпараметры)
            mu_psi = pm.Normal('mu_psi', mu=0.5, sigma=0.2)
            mu_xi = pm.HalfNormal('mu_xi', sigma=1.0)
            mu_rho = pm.Normal('mu_rho', mu=0.0, sigma=1.0)
            mu_tau = pm.HalfNormal('mu_tau', sigma=5.0)
            mu_lam = pm.HalfNormal('mu_lam', sigma=2.0)

            sigma_psi = pm.HalfNormal('sigma_psi', sigma=0.1)
            sigma_xi = pm.HalfNormal('sigma_xi', sigma=0.5)
            sigma_rho = pm.HalfNormal('sigma_rho', sigma=0.5)
            sigma_tau = pm.HalfNormal('sigma_tau', sigma=2.0)
            sigma_lam = pm.HalfNormal('sigma_lam', sigma=1.0)

            psi_raw = pm.Normal('psi_raw', mu=0, sigma=1, shape=n_users)
            xi_raw = pm.Normal('xi_raw', mu=0, sigma=1, shape=n_users)
            rho_raw = pm.Normal('rho_raw', mu=0, sigma=1, shape=n_users)
            tau_raw = pm.Normal('tau_raw', mu=0, sigma=1, shape=n_users)
            lam_raw = pm.Normal('lam_raw', mu=0, sigma=1, shape=n_users)

            psi = pm.Deterministic('psi', pm.math.invlogit(mu_psi + psi_raw * sigma_psi))
            xi = pm.Deterministic('xi', pm.math.exp(mu_xi + xi_raw * sigma_xi)) 
            rho = pm.Deterministic('rho', mu_rho + rho_raw * sigma_rho)
            tau = pm.Deterministic('tau', pm.math.exp(mu_tau + tau_raw * sigma_tau))
            lam = pm.Deterministic('lam', pm.math.exp(mu_lam + lam_raw * sigma_lam))

            # Вычисляем лог-правдоподобие
            ll_vector = loglik_all_users(psi, xi, rho, tau, lam)
            
            # Сохраняем потриальный лог-правдоподобие для LOOIC
            pm.Deterministic('log_lik', ll_vector)
            pm.Potential('likelihood', pt.sum(ll_vector))

            step = pm.DEMetropolisZ() 
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, step=step, return_inferencedata=True, progressbar=False)
            
            # Формируем группу для ArviZ
            idata.add_groups({"log_likelihood": {"log_lik": idata.posterior["log_lik"]}})

        self.idata = idata
        self.user_ids = user_ids
        
        # Печатаем LOOIC сразу, чтобы убедиться, что всё работает
        loo = az.loo(idata, var_name="log_lik")
        print(f"\n[FIT METRICS] LOOIC: {loo.elpd_loo:.2f}")
        
        return idata
    
    def simulate_user_trials_hba(self, param_samples, real_user_data, seed=None):
        """
        Принимает МАТРИЦУ параметров (n_sims, 5), где каждая строка - сэмпл из MCMC.
        Это обеспечивает НАСТОЯЩИЙ HBA PPC.
        """
        n_sims = param_samples.shape[0]
        rng = np.random.RandomState(seed)
        n_trials = len(real_user_data)
        
        explosion_points = []
        for _, row in real_user_data.iterrows():
            if row['popped']:
                explosion_points.append(int(row['pumps']))
            else:
                max_possible = max(int(row['pumps']) + 1, self.max_pumps) 
                explosion_points.append(rng.randint(int(row['pumps']) + 1, max_possible + 1))

        sim_pumps_matrix = np.zeros((n_sims, n_trials))

        for sim in range(n_sims):
            psi, xi, rho, tau, lam = param_samples[sim]
            cum_successes, cum_pumps = 0.0, 0.0
            
            for t in range(n_trials):
                p_burst_k = _compute_pburst_numba(psi, xi, cum_successes, cum_pumps)
                exp_pt = explosion_points[t]
                
                pumps, popped = 0, False
                while True:
                    pumps += 1
                    gain_term = (1.0 - p_burst_k) * self.r_scaled
                    loss_amt = (pumps - 1) * self.r_scaled
                    loss_term = p_burst_k * lam * loss_amt
                    var_term = rho * p_burst_k * (1.0 - p_burst_k) * ((self.r_scaled + lam * loss_amt) ** 2)
                    U_pump = gain_term - loss_term + var_term
                    
                    if rng.rand() > expit(tau * U_pump):
                        popped = False
                        pumps -= 1 
                        break
                    if pumps >= exp_pt:
                        popped = True
                        break
                
                sim_pumps_matrix[sim, t] = pumps
                successes = (pumps - 1) if popped else pumps
                cum_successes += successes
                cum_pumps += pumps

        return sim_pumps_matrix

    @classmethod
    def run_ppc(cls, model_hba, all_data, n_sims=100):
        print("=== Запуск TRUE HBA Posterior Predictive Check ===")
        if not hasattr(model_hba, 'idata'):
            raise ValueError("Сначала выполните подгонку (fit_hba).")

        from scipy.stats import mode
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        post = model_hba.idata.posterior
        user_ids = model_hba.user_ids
        
        chains, draws = post.dims['chain'], post.dims['draw']
        total_samples = chains * draws
        np.random.seed(42)
        sample_indices = np.random.choice(total_samples, size=n_sims, replace=False)
        
        psi_flat = post['psi'].values.reshape(total_samples, -1)[sample_indices, :]
        xi_flat = post['xi'].values.reshape(total_samples, -1)[sample_indices, :]
        rho_flat = post['rho'].values.reshape(total_samples, -1)[sample_indices, :]
        tau_flat = post['tau'].values.reshape(total_samples, -1)[sample_indices, :]
        lam_flat = post['lam'].values.reshape(total_samples, -1)[sample_indices, :]

        metrics_list = []
        global_sim_matrix = np.zeros((n_sims, len(all_data))) # <-- НОВОЕ
        start_idx = 0

        for i, uid in enumerate(tqdm(user_ids, desc="PPC users")):
            user_df = all_data[all_data['user_id'] == uid].sort_values('trial_number')
            n_trials = len(user_df)
            real_pumps = user_df['pumps'].values
            popped_mask = user_df['popped'].values.astype(bool)
            
            user_params_matrix = np.column_stack((
                psi_flat[:, i], xi_flat[:, i], rho_flat[:, i], 
                tau_flat[:, i], lam_flat[:, i]
            ))
            
            sim_matrix = model_hba.simulate_user_trials_hba(user_params_matrix, user_df, seed=42+i)
            
            # Аккумулируем данные для глобального графика
            global_sim_matrix[:, start_idx : start_idx + n_trials] = sim_matrix
            start_idx += n_trials
            
            unpopped_mask = ~popped_mask
            real_adj = real_pumps[unpopped_mask].mean() if unpopped_mask.any() else 0
            sim_adj = sim_matrix[:, unpopped_mask].mean(axis=1) if unpopped_mask.any() else np.zeros(n_sims)
            ppp = np.mean(sim_adj >= real_adj)
            
            sim_mean_pumps = sim_matrix.mean(axis=0)
            r2 = r2_score(real_pumps, sim_mean_pumps) if np.var(real_pumps) > 0 else np.nan
            rmse = np.sqrt(mean_squared_error(real_pumps, sim_mean_pumps))
            mae = mean_absolute_error(real_pumps, sim_mean_pumps)
            msd = np.mean((real_pumps - sim_mean_pumps)**2)
            
            mode_res = mode(sim_matrix, axis=0, keepdims=False)
            mode_sim = np.squeeze(mode_res.mode if hasattr(mode_res, 'mode') else mode_res[0])
            hit_rate = np.mean(mode_sim == real_pumps)
            
            metrics_list.append({
                'user_id': uid, 'ppp': ppp, 'R2': r2, 'RMSE': rmse, 
                'MAE': mae, 'Hit_Rate': hit_rate, 'MSD': msd
            })

        ppc_df = pd.DataFrame(metrics_list)
        print(f" -> EWMV PPC Metrics: Mean ppp={ppc_df['ppp'].mean():.3f}, Mean R2={ppc_df['R2'].mean():.3f}")
        
        # --- НОВОЕ: Отрисовка Timecourse ---
        plot_ppc_timecourse(all_data, global_sim_matrix, "EWMV_Model")

        return ppc_df

    @classmethod
    def run_parameter_recovery(cls, fitted_idata, n_subjects=40, n_trials=50, max_pumps=64):
        print("=== Запуск Parameter Recovery (HBA to HBA) ===")
        
        mu_psi = fitted_idata.posterior['mu_psi'].mean().item()
        sigma_psi = fitted_idata.posterior['sigma_psi'].mean().item()
        mu_xi = fitted_idata.posterior['mu_xi'].mean().item()
        sigma_xi = fitted_idata.posterior['sigma_xi'].mean().item()
        mu_rho = fitted_idata.posterior['mu_rho'].mean().item()
        sigma_rho = fitted_idata.posterior['sigma_rho'].mean().item()
        mu_tau = fitted_idata.posterior['mu_tau'].mean().item()
        sigma_tau = fitted_idata.posterior['sigma_tau'].mean().item()
        mu_lam = fitted_idata.posterior['mu_lam'].mean().item()
        sigma_lam = fitted_idata.posterior['sigma_lam'].mean().item()

        rng = np.random.RandomState(123)
        psi_true = expit(rng.normal(mu_psi, sigma_psi, n_subjects))
        xi_true = np.exp(rng.normal(mu_xi, sigma_xi, n_subjects))
        rho_true = rng.normal(mu_rho, sigma_rho, n_subjects)
        tau_true = np.exp(rng.normal(mu_tau, sigma_tau, n_subjects))
        lam_true = np.exp(rng.normal(mu_lam, sigma_lam, n_subjects))

        model_sim = cls(r=1.0, max_pumps=max_pumps)
        r_scaled = model_sim.r_scaled
        sim_data_list = []
        
        for i in tqdm(range(n_subjects), desc="Simulating Virtual Subjects"):
            params = (psi_true[i], xi_true[i], rho_true[i], tau_true[i], lam_true[i])
            cum_successes, cum_pumps = 0.0, 0.0
            
            for t in range(n_trials):
                p_burst_k = _compute_pburst_numba(params[0], params[1], cum_successes, cum_pumps)
                exp_pt = rng.randint(1, max_pumps + 1)
                pumps, popped = 0, False
                
                while True:
                    pumps += 1
                    gain_term = (1.0 - p_burst_k) * r_scaled
                    loss_amt = (pumps - 1) * r_scaled
                    loss_term = p_burst_k * params[4] * loss_amt
                    var_term = params[2] * p_burst_k * (1.0 - p_burst_k) * ((r_scaled + params[4] * loss_amt) ** 2)
                    U_pump = gain_term - loss_term + var_term
                    
                    if U_pump > 20.0: U_pump = 20.0
                    if U_pump < -20.0: U_pump = -20.0
                    
                    if rng.rand() > expit(params[3] * U_pump):
                        popped = False
                        pumps -= 1
                        break
                    if pumps >= exp_pt:
                        popped = True
                        break
                
                sim_data_list.append({'user_id': f"sim_{i}", 'trial_number': t, 'pumps': pumps, 'popped': popped})
                successes = (pumps - 1) if popped else pumps
                cum_successes += successes
                cum_pumps += pumps

        sim_df = pd.DataFrame(sim_data_list)

        print("Подгонка HBA на симулированных данных (Recovery)...")
        model_recovery = cls(r=1.0, max_pumps=max_pumps)
        recov_idata = model_recovery.fit_hba(sim_df, draws=1000, tune=1000, chains=4, cores=4)

        post = recov_idata.posterior
        hdi = az.hdi(recov_idata, hdi_prob=0.95)
        
        param_names = ['psi', 'xi', 'rho', 'tau', 'lam']
        true_params = [psi_true, xi_true, rho_true, tau_true, lam_true]
        
        recovery_dict = {'user_id': [f"sim_{i}" for i in range(n_subjects)]}
        metrics = []
        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        
        for idx, param in enumerate(param_names):
            true_vals = true_params[idx]
            fit_vals = post[param].mean(dim=['chain', 'draw']).values
            lower_hdi = hdi[param].sel(hdi='lower').values
            upper_hdi = hdi[param].sel(hdi='higher').values
            
            cov_arr = (true_vals >= lower_hdi) & (true_vals <= upper_hdi)
            coverage = np.mean(cov_arr)
            
            r, _ = pearsonr(true_vals, fit_vals)
            r2 = r2_score(true_vals, fit_vals) # Исправлено (вместо r**2)
            bias = np.mean(fit_vals - true_vals)
            rmse = np.sqrt(mean_squared_error(true_vals, fit_vals))
            
            recovery_dict[f'true_{param}'] = true_vals
            recovery_dict[f'fit_{param}'] = fit_vals
            recovery_dict[f'coverage_{param}'] = cov_arr.astype(int)
            
            metrics.append({'Parameter': param, 'r': r, 'R2': r2, 'Bias': bias, 'RMSE': rmse, 'Coverage': coverage})
            
            ax = axes[idx]
            ax.errorbar(true_vals, fit_vals, yerr=[fit_vals - lower_hdi, upper_hdi - fit_vals], fmt='o', alpha=0.6)
            min_v, max_v = min(true_vals.min(), fit_vals.min()), max(true_vals.max(), fit_vals.max())
            ax.plot([min_v, max_v], [min_v, max_v], 'r--')
            ax.set_title(f"{param}\nr={r:.2f}, R2={r2:.2f}, Cov={coverage:.2f}")
            ax.set_xlabel("True")
            ax.set_ylabel("Recovered (HBA Mean)")
            
        plt.tight_layout()
        plt.savefig("bart_ewmv_recovery_plot.png", dpi=300)
        plt.close()
        
        recovery_df = pd.DataFrame(recovery_dict)
        metrics_df = pd.DataFrame(metrics)
        return recovery_df, metrics_df