import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib
matplotlib.use('Agg')  # HPC Safe
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import truncnorm, mode, pearsonr
from scipy.special import expit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

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
    
class Par4Model:
    """
    Иерархическая Байесовская (HBA) реализация Par4 (Park et al., 2021).
    Адаптирована для HPC с масштабированием наград и полным спектром Q1-метрик.
    """

    def __init__(self, data_df, scale_factor=10.0):
        self.data = data_df.copy()
        self.scale_factor = float(scale_factor)
        self.data_dict = self._prepare_data(self.data)
        self.idata = None

    def _prepare_data(self, df):
        df = df.sort_values(by=['user_id', 'trial_number']).reset_index(drop=True)
        
        user_ids = df['user_id'].unique()
        user_mapping = {uid: i for i, uid in enumerate(user_ids)}
        df['sub_idx'] = df['user_id'].map(user_mapping)
        
        df['successes'] = np.where(df['popped'] == 1, df['pumps'] - 1, df['pumps'])
        df['cum_successes'] = df.groupby('user_id')['successes'].apply(lambda x: x.shift().cumsum().fillna(0)).values
        df['cum_pumps'] = df.groupby('user_id')['pumps'].apply(lambda x: x.shift().cumsum().fillna(0)).values
        
        n_obs = len(df)
        max_pumps_dataset = int(df['pumps'].max()) + 2
        
        # --- SCALING (приводим к безопасному диапазону) ---
        l_matrix = np.tile(np.arange(1, max_pumps_dataset + 1), (n_obs, 1)) / self.scale_factor
        pumps_arr = df['pumps'].values[:, None] / self.scale_factor
        popped_arr = df['popped'].values[:, None]
        
        pump_mask = (l_matrix <= pumps_arr).astype(np.float64)
        stop_mask = ((l_matrix == (pumps_arr + (1.0/self.scale_factor))) & (popped_arr == 0)).astype(np.float64)
        
        return {
            'sub_idx': df['sub_idx'].values,
            'cum_successes': df['cum_successes'].values / self.scale_factor,
            'cum_pumps': df['cum_pumps'].values / self.scale_factor,
            'pump_mask': pump_mask,
            'stop_mask': stop_mask,
            'n_subjects': len(user_ids),
            'user_ids': user_ids,
            'df': df
        }

    def fit(self, draws=2000, tune=1000, chains=4, cores=4):
        """HBA подгонка (с progressbar=False для суперкомпьютера)"""
        n_subj = self.data_dict['n_subjects']
        sub_idx = self.data_dict['sub_idx']
        
        with pm.Model() as model:
            # Гиперпараметры (Non-centered parameterization)
            mu_phi = pm.Normal('mu_phi', mu=0.0, sigma=1.0)
            sigma_phi = pm.HalfNormal('sigma_phi', sigma=1.0)
            phi_offset = pm.Normal('phi_offset', mu=0.0, sigma=1.0, shape=n_subj)
            phi = pm.Deterministic('phi', pm.math.invlogit(mu_phi + phi_offset * sigma_phi))
            
            mu_eta = pm.Normal('mu_eta', mu=-2.0, sigma=1.0)
            sigma_eta = pm.HalfNormal('sigma_eta', sigma=1.0)
            eta_offset = pm.Normal('eta_offset', mu=0.0, sigma=1.0, shape=n_subj)
            eta = pm.Deterministic('eta', pm.math.exp(mu_eta + eta_offset * sigma_eta))
            
            mu_gamma = pm.Normal('mu_gamma', mu=0.0, sigma=1.0)
            sigma_gamma = pm.HalfNormal('sigma_gamma', sigma=1.0)
            gamma_offset = pm.Normal('gamma_offset', mu=0.0, sigma=1.0, shape=n_subj)
            gamma = pm.Deterministic('gamma', pm.math.exp(mu_gamma + gamma_offset * sigma_gamma))
            
            mu_tau = pm.Normal('mu_tau', mu=0.0, sigma=1.0)
            sigma_tau = pm.HalfNormal('sigma_tau', sigma=1.0)
            tau_offset = pm.Normal('tau_offset', mu=0.0, sigma=1.0, shape=n_subj)
            tau = pm.Deterministic('tau', pm.math.exp(mu_tau + tau_offset * sigma_tau))
            
            phi_i = phi[sub_idx]
            eta_i = eta[sub_idx]
            gamma_i = gamma[sub_idx]
            tau_i = tau[sub_idx]
            
            # Векторизованная математика
            num = phi_i + eta_i * self.data_dict['cum_successes']
            den = 1.0 + eta_i * self.data_dict['cum_pumps']
            p_burst = pt.clip(1.0 - (num / den), 1e-6, 1.0 - 1e-6)
            
            nu = -gamma_i / pt.log(1.0 - p_burst)
            
            l_arr = np.arange(1, self.data_dict['pump_mask'].shape[1] + 1) / self.scale_factor
            
            exponent = tau_i[:, None] * (nu[:, None] - l_arr[None, :])
            p_pump = pt.clip(pm.math.invlogit(exponent), 1e-6, 1.0 - 1e-6)
            
            ll_obs = pt.sum(self.data_dict['pump_mask'] * pt.log(p_pump) + 
                            self.data_dict['stop_mask'] * pt.log(1.0 - p_pump), axis=1)
            
            pm.Deterministic('log_lik', ll_obs)
            pm.Potential('obs', ll_obs)
            
            print(f"Запуск MCMC (HBA) для {n_subj} пользователей...")
            # HPC SAFE
            self.idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, 
                                   target_accept=0.90, progressbar=False)
            self.idata.add_groups(
                {"log_likelihood": {"obs": self.idata.posterior["log_lik"]}}
            )
        return self.idata

    def calculate_looic(self):
        if self.idata is None: raise ValueError("Сначала запустите fit()")
        return az.loo(self.idata, var_name='log_lik')

    @staticmethod
    def _simulate_trial(phi, eta, gamma, tau, cum_successes, cum_pumps, scale_factor, max_capacity=64):
        """Симуляция с учетом масштабирования"""
        explosion_point = np.random.randint(1, max_capacity + 1)
        
        c_succ_s = cum_successes / scale_factor
        c_pump_s = cum_pumps / scale_factor
        
        num = phi + eta * c_succ_s
        den = 1.0 + eta * c_pump_s
        p_burst = np.clip(1.0 - (num / den), 1e-6, 1.0 - 1e-6)
        nu_s = -gamma / np.log(1.0 - p_burst)
        
        l = 1
        while True:
            l_s = l / scale_factor
            p_pump = expit(tau * (nu_s - l_s))
            if np.random.rand() > p_pump: return l - 1, 0  # Cash-out
            if l >= explosion_point: return l, 1           # Burst
            if l >= max_capacity: return l, 0
            l += 1

    def posterior_predictive_check(self, n_simulations=100, max_capacity=64):
        """PPC с индивидуальными (trial-by-trial) метриками"""
        from scipy.stats import mode
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        post = az.extract(self.idata)
        total_samples = post.sizes['sample']
        sample_idxs = np.random.choice(total_samples, size=n_simulations, replace=False)
        
        user_ids = self.data_dict['user_ids']
        df = self.data_dict['df']
        n_trials = df['trial_number'].nunique()
        
        real_pumps = np.zeros((len(user_ids), n_trials))
        real_popped = np.zeros((len(user_ids), n_trials), dtype=bool)
        
        for i, uid in enumerate(user_ids):
            user_data = df[df['user_id'] == uid].sort_values('trial_number')
            real_pumps[i, :] = user_data['pumps'].values
            real_popped[i, :] = user_data['popped'].values.astype(bool)
            
        sim_pumps_3d = np.zeros((n_simulations, len(user_ids), n_trials))
        
        print("\n--- Генерация PPC симуляций ---")
        for sim_idx, m_idx in enumerate(sample_idxs):
            phi_samp = post['phi'][:, m_idx].values
            eta_samp = post['eta'][:, m_idx].values
            gamma_samp = post['gamma'][:, m_idx].values
            tau_samp = post['tau'][:, m_idx].values
            
            for u_i, uid in enumerate(user_ids):
                c_succ, c_pump = 0.0, 0.0
                for t_i in range(n_trials):
                    pumps, popped = self._simulate_trial(
                        phi_samp[u_i], eta_samp[u_i], gamma_samp[u_i], tau_samp[u_i], 
                        c_succ, c_pump, self.scale_factor, max_capacity
                    )
                    sim_pumps_3d[sim_idx, u_i, t_i] = pumps
                    c_succ += pumps if popped == 0 else (pumps - 1)
                    c_pump += pumps

        metrics_list = []
        global_sim_matrix = np.zeros((n_simulations, len(df))) # <-- НОВОЕ
        start_idx = 0

        for i, uid in enumerate(user_ids):
            r_vec = real_pumps[i]
            popped_mask = real_popped[i]
            sim_matrix = sim_pumps_3d[:, i, :] 
            
            n_u_trials = len(df[df['user_id'] == uid])
            global_sim_matrix[:, start_idx : start_idx + n_u_trials] = sim_matrix[:, :n_u_trials]
            start_idx += n_u_trials
            
            unpopped_mask = ~popped_mask
            real_adj = r_vec[unpopped_mask].mean() if unpopped_mask.any() else 0
            sim_adj = sim_matrix[:, unpopped_mask].mean(axis=1) if unpopped_mask.any() else np.zeros(n_simulations)
            ppp = np.mean(sim_adj >= real_adj)
            
            s_vec = sim_matrix.mean(axis=0)
            r2 = r2_score(r_vec, s_vec) if np.var(r_vec) > 0 else np.nan
            rmse = np.sqrt(mean_squared_error(r_vec, s_vec))
            mae = mean_absolute_error(r_vec, s_vec)
            msd = np.mean((r_vec - s_vec)**2)
            
            mode_res = mode(sim_matrix, axis=0, keepdims=False)
            s_mode = np.squeeze(mode_res.mode if hasattr(mode_res, 'mode') else mode_res[0])
            hit_rate = np.mean(r_vec == s_mode)
            
            metrics_list.append({
                'user_id': uid, 'ppp': ppp, 'R2': r2, 'RMSE': rmse, 
                'MAE': mae, 'Hit_Rate': hit_rate, 'MSD': msd
            })
            
        metrics_df = pd.DataFrame(metrics_list)
        print(f" -> Средний ppp: {metrics_df['ppp'].mean():.3f}, Средний R2: {metrics_df['R2'].mean():.3f}")
        
        # --- НОВОЕ: Отрисовка Timecourse ---
        plot_ppc_timecourse(df, global_sim_matrix, "Par4_Model")

        return metrics_df

    def parameter_recovery(self, n_subjects=50, n_trials=50, max_capacity=64):
        """TRUE HBA Parameter Recovery. Метрики: r, R2, RMSE, Bias, Coverage"""
        post_summary = az.summary(self.idata, var_names=['phi', 'eta', 'gamma', 'tau'])
        
        def get_trunc(name, a, b):
            m = post_summary.loc[post_summary.index.str.startswith(name), 'mean'].mean()
            s = post_summary.loc[post_summary.index.str.startswith(name), 'sd'].mean()
            return truncnorm.rvs((a - m) / s, (b - m) / s, loc=m, scale=s, size=n_subjects)

        true_params = {
            'phi': get_trunc('phi', 0.001, 0.999),
            'eta': get_trunc('eta', 0.001, 5.0),
            'gamma': get_trunc('gamma', 0.001, 10.0),
            'tau': get_trunc('tau', 0.001, 5.0)
        }
        
        sim_data = []
        for i in range(n_subjects):
            c_succ, c_pump = 0, 0
            for t in range(n_trials):
                p, po = self._simulate_trial(true_params['phi'][i], true_params['eta'][i], 
                                             true_params['gamma'][i], true_params['tau'][i], 
                                             c_succ, c_pump, self.scale_factor, max_capacity)
                sim_data.append({'user_id': f'sim_{i}', 'trial_number': t+1, 'pumps': p, 'popped': po})
                c_succ += (p - 1) if po else p
                c_pump += p
                
        sim_df = pd.DataFrame(sim_data)
        
        print("\n--- Запуск HBA подгонки на искусственных данных (Parameter Recovery) ---")
        recovery_model = Par4Model(sim_df, scale_factor=self.scale_factor)
        # Уменьшенные draws для скорости Recovery
        rec_idata = recovery_model.fit(draws=500, tune=500, chains=4, cores=4)
        
        rec_summary = az.summary(rec_idata, var_names=['phi', 'eta', 'gamma', 'tau'])
        hdi_data = az.hdi(rec_idata, var_names=['phi', 'eta', 'gamma', 'tau'])
        
        metrics = []
        recovery_dict = {}
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for j, p_name in enumerate(['phi', 'eta', 'gamma', 'tau']):
            t_val = true_params[p_name]
            f_val = rec_summary.loc[rec_summary.index.str.startswith(f'{p_name}['), 'mean'].values
            
            # Coverage (95% HDI)
            hdi_lower = hdi_data[p_name].sel(hdi='lower').values
            hdi_upper = hdi_data[p_name].sel(hdi='higher').values
            coverage = np.mean((t_val >= hdi_lower) & (t_val <= hdi_upper))
            
            r_val = pearsonr(t_val, f_val)[0]
            r2_val = r2_score(t_val, f_val)
            bias = np.mean(f_val - t_val)
            rmse = np.sqrt(mean_squared_error(t_val, f_val))
            
            metrics.append({'param': p_name, 'r': r_val, 'R2': r2_val, 'bias': bias, 'rmse': rmse, 'coverage': coverage})
            
            recovery_dict[f'true_{p_name}'] = t_val
            recovery_dict[f'fit_{p_name}'] = f_val
            
            axes[j].scatter(t_val, f_val, alpha=0.7, edgecolors='k')
            min_v, max_v = min(t_val.min(), f_val.min()), max(t_val.max(), f_val.max())
            axes[j].plot([min_v, max_v], [min_v, max_v], 'r--')
            axes[j].set_title(f"{p_name}\nr={r_val:.2f}, Bias={bias:.2f}\nCov={coverage:.2f}")
            axes[j].set_xlabel("True")
            axes[j].set_ylabel("Recovered")
            
        plt.tight_layout()
        plt.savefig("bart_par4_recovery_scatter.png", dpi=300)
        plt.close()
        
        metrics_df = pd.DataFrame(metrics)
        recovery_df = pd.DataFrame(recovery_dict)
        
        return recovery_df, metrics_df