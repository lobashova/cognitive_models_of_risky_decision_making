import os
import matplotlib
matplotlib.use('Agg')  # [HPC FIX]: Отключаем интерактивный бэкенд ДО импорта pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy.stats import truncnorm, pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

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

class EWModel:
    """
    Hierarchical Bayesian Exponential-Weight (EW) model from Park et al. (2021).
    Реализация на PyMC (HBA) с векторизованным правдоподобием и масштабированием.
    """
    
    def __init__(self, r=5.0):
        # [МАСШТАБИРОВАНИЕ]: Делим на 100, чтобы награды были в диапазоне [-10, 10]
        # Для r=5.0 максимальная награда за 64 качка составит 320. После деления: 3.2.
        self.scaling_factor = 100.0
        self.r_scaled = float(r) / self.scaling_factor

    def _prepare_data(self, df):
        df = df.copy().sort_values(['user_id', 'trial_number'])
        unique_users = df['user_id'].unique()
        user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
        df['user_idx'] = df['user_id'].map(user_mapping)
        
        df['successes'] = np.where(df['popped'], df['pumps'] - 1, df['pumps'])
        df['cum_pumps'] = df.groupby('user_id')['pumps'].cumsum().shift(1).fillna(0)
        df['cum_successes'] = df.groupby('user_id')['successes'].cumsum().shift(1).fillna(0)
        
        return df, unique_users

    def fit(self, df, draws=1000, tune=1000, chains=4):
        df_prep, unique_users = self._prepare_data(df)
        N = len(unique_users)
        
        max_k = int(df_prep['pumps'].max() + 1)
        J_matrix = np.tile(np.arange(1, max_k + 1), (len(df_prep), 1))
        
        pumps_arr = df_prep['pumps'].values[:, None]
        popped_arr = df_prep['popped'].values[:, None]
        
        mask_pump = J_matrix <= pumps_arr
        mask_stop = (J_matrix == (pumps_arr + 1)) & (~popped_arr)
        
        with pm.Model() as ew_model:
            mu_pr = pm.Normal('mu_pr', mu=0, sigma=1, shape=5)
            sigma_pr = pm.HalfNormal('sigma_pr', sigma=0.5, shape=5)
            z_pr = pm.Normal('z_pr', mu=0, sigma=1, shape=(5, N))
            
            psi = pm.Deterministic('psi', pm.math.invlogit(mu_pr[0] + sigma_pr[0] * z_pr[0]))
            xi  = pm.Deterministic('xi',  5.0 * pm.math.invlogit(mu_pr[1] + sigma_pr[1] * z_pr[1]))
            rho = pm.Deterministic('rho', 3.0 * pm.math.invlogit(mu_pr[2] + sigma_pr[2] * z_pr[2]))
            tau = pm.Deterministic('tau', 50.0 * pm.math.invlogit(mu_pr[3] + sigma_pr[3] * z_pr[3]))
            lam = pm.Deterministic('lam', 20.0 * pm.math.invlogit(mu_pr[4] + sigma_pr[4] * z_pr[4]))
            
            subj_idx = df_prep['user_idx'].values
            psi_t, xi_t = psi[subj_idx], xi[subj_idx]
            rho_t, tau_t, lam_t = rho[subj_idx], tau[subj_idx], lam[subj_idx]
            
            weight = pm.math.exp(-xi_t * df_prep['cum_pumps'].values)
            p_emp = pt.switch(
                pt.gt(df_prep['cum_pumps'].values, 0),
                (df_prep['cum_pumps'].values - df_prep['cum_successes'].values) / df_prep['cum_pumps'].values,
                0.0
            )
            p_burst = pt.clip(weight * psi_t + (1.0 - weight) * p_emp, 1e-6, 1.0 - 1e-6)
            
            # Используем масштабированную награду
            loss_amount = (J_matrix - 1.0) * self.r_scaled
            gain_term = (1.0 - p_burst[:, None]) * (self.r_scaled ** rho_t[:, None])
            loss_term = p_burst[:, None] * lam_t[:, None] * pt.exp(rho_t[:, None] * pt.log(loss_amount + 1e-6))
            
            U = gain_term - loss_term
            p_pump = pt.clip(pm.math.sigmoid(tau_t[:, None] * U), 1e-6, 1.0 - 1e-6)
            
            ll_pump = pt.log(p_pump) * mask_pump
            ll_stop = pt.log(1.0 - p_pump) * mask_stop
            
            # [METRICS FIX]: Потриальный log_likelihood для расчета LOOIC
            ll_trial = pt.sum(ll_pump + ll_stop, axis=1)
            pm.Deterministic('log_lik', ll_trial)
            pm.Potential('likelihood', pt.sum(ll_trial))
            
            # [HPC FIX]: Отключение progressbar
            trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=chains, 
                              target_accept=0.9, return_inferencedata=True, progressbar=False)
            trace.add_groups(
                {"log_likelihood": {"log_lik": trace.posterior["log_lik"]}}
            )
        params_df = pd.DataFrame({
            'user_id': unique_users,
            'psi': trace.posterior['psi'].mean(dim=['chain', 'draw']).values,
            'xi': trace.posterior['xi'].mean(dim=['chain', 'draw']).values,
            'rho': trace.posterior['rho'].mean(dim=['chain', 'draw']).values,
            'tau': trace.posterior['tau'].mean(dim=['chain', 'draw']).values,
            'lam': trace.posterior['lam'].mean(dim=['chain', 'draw']).values
        })
        
        # Расчет LOOIC
        loo = az.loo(trace, var_name="log_lik")
        # Используем .elpd_loo для получения значения 
        print(f"\n[FIT METRICS] LOO (elpd): {loo.elpd_loo:.2f}")
        
        return trace, params_df

    def _simulate_agent_full(self, psi, xi, rho, tau, lam, user_df):
        """Симулирует агента триал-к-триалу с сохранением истинных точек взрыва"""
        cum_successes, cum_pumps = 0.0, 0.0
        sim_pumps = []
        
        # Если в данных нет explosion_point, генерируем случайно
        has_ep = 'explosion_point' in user_df.columns
        
        for _, row in user_df.iterrows():
            weight = np.exp(-xi * cum_pumps)
            p_emp = (cum_pumps - cum_successes) / cum_pumps if cum_pumps > 0 else 0.0
            p_burst = np.clip(weight * psi + (1.0 - weight) * p_emp, 1e-6, 1.0 - 1e-6)
            
            explosion_pt = int(row['explosion_point']) if has_ep else np.random.randint(1, 65)
            j, popped = 1, False
            
            while j <= 64:
                U = (1.0 - p_burst) * (self.r_scaled ** rho) - p_burst * lam * (((j - 1) * self.r_scaled) ** rho)
                p_pump = 1.0 / (1.0 + np.exp(-tau * U))
                
                if np.random.rand() < p_pump:
                    if j >= explosion_pt:
                        popped = True
                        break
                    j += 1
                else:
                    break
                    
            pumps_made = j if popped else j - 1
            sim_pumps.append(pumps_made)
                
            cum_successes += (pumps_made - 1) if popped else pumps_made
            cum_pumps += pumps_made
            
        return np.array(sim_pumps)

    def posterior_predictive_check(self, trace, df, n_draws=100):
        print(f"Запуск PPC ({n_draws} симуляций из постериора)...")
        from scipy.stats import mode
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        df = df.sort_values(['user_id', 'trial_number'])
        unique_users = df['user_id'].unique()
        
        post = trace.posterior
        n_chains, n_samples = post['psi'].shape[0], post['psi'].shape[1]
        
        all_real_pumps = df['pumps'].values
        all_popped_mask = df['popped'].values.astype(bool)
        sim_pumps_matrix = np.zeros((n_draws, len(df)))
        
        for i in tqdm(range(n_draws), desc="Сэмплирование PPC"):
            c, d = np.random.randint(0, n_chains), np.random.randint(0, n_samples)
            sim_draw_pumps = []
            
            for u_idx, uid in enumerate(unique_users):
                user_df = df[df['user_id'] == uid]
                pumps_arr = self._simulate_agent_full(
                    psi=float(post['psi'][c, d, u_idx]),
                    xi=float(post['xi'][c, d, u_idx]),
                    rho=float(post['rho'][c, d, u_idx]),
                    tau=float(post['tau'][c, d, u_idx]),
                    lam=float(post['lam'][c, d, u_idx]),
                    user_df=user_df
                )
                sim_draw_pumps.extend(pumps_arr)
            sim_pumps_matrix[i, :] = sim_draw_pumps
            
        metrics = []
        start_idx = 0
        for u_idx, uid in enumerate(unique_users):
            n_trials = len(df[df['user_id'] == uid])
            end_idx = start_idx + n_trials
            
            u_real = all_real_pumps[start_idx:end_idx]
            u_popped = all_popped_mask[start_idx:end_idx]
            u_sim = sim_pumps_matrix[:, start_idx:end_idx]
            
            unpopped_mask = ~u_popped
            real_adj = u_real[unpopped_mask].mean() if unpopped_mask.any() else 0
            sim_adj = u_sim[:, unpopped_mask].mean(axis=1) if unpopped_mask.any() else np.zeros(n_draws)
            ppp = np.mean(sim_adj >= real_adj)
            
            u_mean_sim = u_sim.mean(axis=0)
            r2 = r2_score(u_real, u_mean_sim) if np.var(u_real) > 0 else np.nan
            rmse = np.sqrt(mean_squared_error(u_real, u_mean_sim))
            mae = mean_absolute_error(u_real, u_mean_sim)
            msd = np.mean((u_real - u_mean_sim)**2)
            
            mode_res = mode(u_sim, axis=0, keepdims=False)
            mode_sim_pumps = np.squeeze(mode_res.mode if hasattr(mode_res, 'mode') else mode_res[0])
            hit_rate = np.mean(mode_sim_pumps == u_real)
            
            metrics.append({
                'user_id': uid, 'ppp': ppp, 'R2': r2, 'RMSE': rmse, 
                'MAE': mae, 'Hit_Rate': hit_rate, 'MSD': msd
            })
            start_idx = end_idx
            
        ppc_df = pd.DataFrame(metrics)
        print(f" -> Средний ppp: {ppc_df['ppp'].mean():.3f}, Средний R2: {ppc_df['R2'].mean():.3f}")
        plot_ppc_timecourse(df, sim_pumps_matrix, "EW_Model")
        return ppc_df

    def parameter_recovery(self, params_df, n_subjects=50, n_trials=30):
        print("Генерация данных для Parameter Recovery...")
        
        def gen_truncnorm(mean, std, lower, upper, size):
            a, b = (lower - mean) / std, (upper - mean) / std
            return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
            
        stats = {
            'psi': (params_df['psi'].mean(), params_df['psi'].std(), 0.001, 0.999),
            'xi':  (params_df['xi'].mean(), params_df['xi'].std(), 0.0001, 5.0),
            'rho': (params_df['rho'].mean(), params_df['rho'].std(), 0.0001, 3.0),
            'tau': (params_df['tau'].mean(), params_df['tau'].std(), 0.0001, 50.0),
            'lam': (params_df['lam'].mean(), params_df['lam'].std(), 0.0001, 20.0),
        }
        
        true_params = {param: gen_truncnorm(m, s, l, u, n_subjects) for param, (m, s, l, u) in stats.items()}
        
        sim_data = []
        for i in range(n_subjects):
            p_dict = {k: true_params[k][i] for k in stats.keys()}
            cum_successes, cum_pumps = 0.0, 0.0
            
            for t in range(1, n_trials + 1):
                weight = np.exp(-p_dict['xi'] * cum_pumps)
                p_emp = (cum_pumps - cum_successes) / cum_pumps if cum_pumps > 0 else 0.0
                p_burst = np.clip(weight * p_dict['psi'] + (1.0 - weight) * p_emp, 1e-6, 1.0 - 1e-6)
                
                explosion_pt = np.random.randint(1, 65)
                j, popped = 1, False
                
                while j <= 64:
                    U = (1.0 - p_burst)*(self.r_scaled**p_dict['rho']) - p_burst*p_dict['lam']*(((j-1)*self.r_scaled)**p_dict['rho'])
                    p_pump = 1.0 / (1.0 + np.exp(-p_dict['tau'] * U))
                    
                    if np.random.rand() < p_pump:
                        if j >= explosion_pt:
                            popped = True
                            break
                        j += 1
                    else:
                        break
                        
                pumps_made = j if popped else j - 1
                sim_data.append({
                    'user_id': f'sim_{i}', 'trial_number': t, 
                    'pumps': pumps_made, 'popped': popped, 'explosion_point': explosion_pt
                })
                cum_successes += (pumps_made - 1) if popped else pumps_made
                cum_pumps += pumps_made
                
        sim_df = pd.DataFrame(sim_data)
        
        print("Подгонка HBA на симулированных данных...")
        trace_recov, recov_params_df = self.fit(sim_df, draws=1000, tune=1000, chains=4)
        
        recovery_df = pd.DataFrame({'user_id': recov_params_df['user_id']})
        metrics = []
        
        fig, axes = plt.subplots(1, 5, figsize=(24, 5))
        param_keys = ['psi', 'xi', 'rho', 'tau', 'lam']
        titles = ['Prior belief (psi)', 'Updating (xi)', 'Risk (rho)', 'Consistency (tau)', 'Loss Aversion (lam)']
        
        print("\n[PARAMETER RECOVERY METRICS]")
        
        for idx, p_key in enumerate(param_keys):
            true_vals = true_params[p_key]
            recov_vals = recov_params_df[p_key].values
            
            # Использование az.hdi вместо ETI (percentile)
            hdi_bounds = az.hdi(trace_recov, var_names=[p_key], hdi_prob=0.95)[p_key].values
            hdi_lower, hdi_upper = hdi_bounds[:, 0], hdi_bounds[:, 1]
            cov_arr = (true_vals >= hdi_lower) & (true_vals <= hdi_upper)
            coverage = np.mean(cov_arr)
            
            r_val, _ = pearsonr(true_vals, recov_vals)
            r2_val = r2_score(true_vals, recov_vals) # sklearn r2_score
            rmse = np.sqrt(mean_squared_error(true_vals, recov_vals))
            bias = np.mean(recov_vals - true_vals)
            
            recovery_df[f'true_{p_key}'] = true_vals
            recovery_df[f'fit_{p_key}'] = recov_vals
            recovery_df[f'coverage_{p_key}'] = cov_arr.astype(int)
            
            metrics.append({'Parameter': p_key, 'r': r_val, 'R2': r2_val, 'RMSE': rmse, 'Bias': bias, 'Coverage': coverage})
            print(f"{p_key.upper()}: r={r_val:.3f}, R2={r2_val:.3f}, RMSE={rmse:.3f}, Bias={bias:.3f}, Coverage={coverage:.1%}")
            
            axes[idx].errorbar(true_vals, recov_vals, 
                               yerr=[recov_vals - hdi_lower, hdi_upper - recov_vals],
                               fmt='o', alpha=0.6, ecolor='lightgray', capsize=0, color='royalblue')
                               
            min_v, max_v = min(true_vals.min(), recov_vals.min()), max(true_vals.max(), recov_vals.max())
            axes[idx].plot([min_v, max_v], [min_v, max_v], 'k--', lw=1)
            axes[idx].set_title(f'{titles[idx]}\nr={r_val:.2f}, R2={r2_val:.2f}\nCov={coverage:.1%}, RMSE={rmse:.2f}')
            axes[idx].set_xlabel('True Parameter')
            axes[idx].set_ylabel('Recovered Parameter (95% HDI)')
            
        sns.despine()
        plt.tight_layout()
        plt.savefig("bart_ew_recovery_plot.png", dpi=300)
        plt.close()
            
        metrics_df = pd.DataFrame(metrics)
        return recovery_df, metrics_df