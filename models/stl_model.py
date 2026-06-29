import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
# Принудительно устанавливаем бэкенд Agg ДО импорта pyplot для суперкомпьютера
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.special import expit, logit
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
from scipy.stats import truncnorm, pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from numba import njit
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# ОПТИМИЗИРОВАННОЕ ЯДРО ДЛЯ LOOIC (ПОТРИАЛЬНО)
# ---------------------------------------------------------
@njit(fastmath=True)
def stl_logp_numba(w1_scaled, vwin, vloss, beta, pumps_arr, popped_arr, max_pumps, scale_factor):
    w = w1_scaled * max_pumps
    n_trials = len(pumps_arr)
    logp_arr = np.zeros(n_trials, dtype=np.float64) # Возвращаем МАССИВ
    eps = 1e-12
    
    for t in range(n_trials):
        p = int(pumps_arr[t])
        pop = bool(popped_arr[t])
        trial_logp = 0.0
        
        if p > 0:
            for k in range(1, p + 1):
                z = beta * ((k / scale_factor) - (w / scale_factor))
                # Защита от переполнения
                if z > 700.0: z = 700.0
                elif z < -700.0: z = -700.0
                p_pump = 1.0 / (1.0 + np.exp(z))
                trial_logp += np.log(p_pump + eps)
                
        if not pop and p < max_pumps:
            z_coll = beta * (((p + 1) / scale_factor) - (w / scale_factor))
            if z_coll > 700.0: z_coll = 700.0
            elif z_coll < -700.0: z_coll = -700.0
            p_pump_collect = 1.0 / (1.0 + np.exp(z_coll))
            trial_logp += np.log(1.0 - p_pump_collect + eps)
            
        # Сохраняем правдоподобие ИМЕННО ЭТОГО ТРИАЛА
        logp_arr[t] = trial_logp
            
        # Обновление стратегии w по правилам STL
        if pop:
            w = w * (1.0 - vloss * (1.0 - p / max_pumps))
        else:
            w = w * (1.0 + vwin * (p / max_pumps))
            
        if w < 1.0: w = 1.0
        if w > max_pumps: w = max_pumps
        
    return logp_arr

# ---------------------------------------------------------
# ФУНКЦИЯ ОТРИСОВКИ ТАЙМКОРСА
# ---------------------------------------------------------
def plot_ppc_timecourse(df, sim_matrix, model_name, window=5):
    work_df = df[['trial_number', 'pumps', 'popped']].copy()
    
    for s in range(sim_matrix.shape[0]):
        work_df[f'sim_{s}'] = sim_matrix[s, :]
        
    work_df.loc[work_df['popped'] == True, 'pumps'] = np.nan
    for s in range(sim_matrix.shape[0]):
        work_df.loc[work_df['popped'] == True, f'sim_{s}'] = np.nan

    real_grouped = work_df.groupby('trial_number')['pumps'].mean()
    sim_grouped = work_df.drop(columns=['pumps', 'popped']).groupby('trial_number').mean()

    sim_mean = sim_grouped.mean(axis=1)
    sim_hdi_low = np.percentile(sim_grouped, 2.5, axis=1)
    sim_hdi_high = np.percentile(sim_grouped, 97.5, axis=1)

    real_ma = real_grouped.rolling(window, min_periods=1).mean()
    sim_mean_ma = sim_mean.rolling(window, min_periods=1).mean()
    sim_low_ma = pd.Series(sim_hdi_low, index=sim_grouped.index).rolling(window, min_periods=1).mean()
    sim_high_ma = pd.Series(sim_hdi_high, index=sim_grouped.index).rolling(window, min_periods=1).mean()

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
    
    filename = f"{model_name.replace(' ', '_').lower()}_timecourse.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f" -> График Timecourse сохранен: {filename}")

# ---------------------------------------------------------
# ОСНОВНОЙ КЛАСС HBA МОДЕЛИ
# ---------------------------------------------------------
class STLModelHBA:
    def __init__(self, data_df, max_pumps=64):
        self.data = data_df.copy()
        self.max_pumps = float(max_pumps)
        self.scale_factor = 10.0
        self.user_ids = sorted(self.data['user_id'].unique())
        self.n_users = len(self.user_ids)
        self.trace = None
        
        print(f"Инициализация HBA STL: найдено {self.n_users} пользователей. Масштаб: 1/{self.scale_factor}")

    def fit(self, draws=1000, tune=1000, chains=4, cores=4):
        print("Подготовка данных и построение графа PyMC...")
        
        # Предварительно извлекаем данные (для numba)
        pumps_list = [self.data[self.data['user_id'] == uid].sort_values('trial_number')['pumps'].to_numpy(dtype=np.float64) for uid in self.user_ids]
        popped_list = [self.data[self.data['user_id'] == uid].sort_values('trial_number')['popped'].to_numpy(dtype=np.float64) for uid in self.user_ids]

        @as_op(itypes=[pt.dvector, pt.dvector, pt.dvector, pt.dvector], otypes=[pt.dvector])
        def hba_logp_op(w1_s, vwin, vloss, beta):
            all_lls = []
            for i in range(self.n_users):
                ll_arr = stl_logp_numba(
                    w1_s[i], vwin[i], vloss[i], beta[i], 
                    pumps_list[i], popped_list[i], 
                    self.max_pumps, self.scale_factor
                )
                all_lls.append(ll_arr)
            # Склеиваем массивы для потриального вывода
            return np.concatenate(all_lls)

        with pm.Model() as self.model:
            # 1. Групповые гиперпараметры
            mu_w1 = pm.Beta('mu_w1', alpha=2, beta=2)
            sigma_w1 = pm.HalfNormal('sigma_w1', sigma=1.0)
            mu_vwin = pm.Beta('mu_vwin', alpha=2, beta=2)
            sigma_vwin = pm.HalfNormal('sigma_vwin', sigma=1.0)
            mu_vloss = pm.Beta('mu_vloss', alpha=2, beta=2)
            sigma_vloss = pm.HalfNormal('sigma_vloss', sigma=1.0)
            mu_beta = pm.HalfNormal('mu_beta', sigma=2.0)
            sigma_beta = pm.HalfNormal('sigma_beta', sigma=1.0)

            # 2. Индивидуальные параметры (Non-centered)
            offset_w1 = pm.Normal('offset_w1', mu=0, sigma=1, shape=self.n_users)
            w1_s = pm.Deterministic('w1_s', pm.math.invlogit(pm.math.logit(mu_w1) + offset_w1 * sigma_w1))

            offset_vwin = pm.Normal('offset_vwin', mu=0, sigma=1, shape=self.n_users)
            vwin = pm.Deterministic('vwin', pm.math.invlogit(pm.math.logit(mu_vwin) + offset_vwin * sigma_vwin))

            offset_vloss = pm.Normal('offset_vloss', mu=0, sigma=1, shape=self.n_users)
            vloss = pm.Deterministic('vloss', pm.math.invlogit(pm.math.logit(mu_vloss) + offset_vloss * sigma_vloss))

            offset_beta = pm.Normal('offset_beta', mu=0, sigma=1, shape=self.n_users)
            beta = pm.Deterministic('beta', pm.math.exp(pm.math.log(mu_beta) + offset_beta * sigma_beta))

            # Точечный Log-likelihood для LOOIC
            logp_vec = pm.Deterministic('log_lik', hba_logp_op(w1_s, vwin, vloss, beta))
            pm.Potential('likelihood', pt.sum(logp_vec))

            print("Запуск MCMC сэмплирования (progressbar=False для HPC)...")
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, step=pm.Slice(), return_inferencedata=True, progressbar=False)
            
            # Сохраняем группу с правильным именем
            self.trace.add_groups(
                {"log_likelihood": {"log_lik": self.trace.posterior["log_lik"]}}
            )
            
        return self.trace

    def get_looic(self):
        if self.trace is None:
            return None
        return az.loo(self.trace, pointwise=True, var_name="log_lik")

    def simulate_user(self, w1_scaled, vwin, vloss, beta, real_user_df):
        w = w1_scaled * self.max_pumps
        sim_data = []
        rng = np.random.RandomState()
        
        for _, row in real_user_df.iterrows():
            bp = int(row['break_point'])
            trial = int(row['trial_number'])
            k = 0
            
            while True:
                k += 1
                z = beta * ((k / self.scale_factor) - (w / self.scale_factor))
                p_pump = 1.0 / (1.0 + np.exp(np.clip(z, -700, 700)))
                
                if rng.rand() > p_pump:
                    popped = False
                    k -= 1 # Последний качок не состоялся
                    break
                
                if k >= bp:
                    popped = True
                    break
                    
            pumps_made = k if popped else k
            sim_data.append({'trial_number': trial, 'pumps': pumps_made, 'popped': popped})
            
            if popped:
                w = w * (1.0 - vloss * (1.0 - pumps_made / self.max_pumps))
            else:
                w = w * (1.0 + vwin * (pumps_made / self.max_pumps))
            w = np.clip(w, 1.0, self.max_pumps)
            
        return pd.DataFrame(sim_data)

    def predictive_check(self, n_sim=100, window=5):
        if self.trace is None: raise ValueError("Запустите .fit()")
        print(f"Запуск PPC: {n_sim} симуляций из апостериора для каждого пользователя...")
        from scipy.stats import mode

        stacked = az.extract(self.trace)
        n_samples = stacked.sizes['sample']
        sample_indices = np.random.choice(n_samples, n_sim, replace=False)
        
        results = []
        global_sim_matrix = np.zeros((n_sim, len(self.data)))
        start_idx = 0
        
        for i, uid in enumerate(tqdm(self.user_ids, desc="PPC users")):
            real_data = self.data[self.data['user_id'] == uid].sort_values('trial_number')
            real_pumps = real_data['pumps'].values
            popped_mask = real_data['popped'].values.astype(bool)
            sim_pumps_matrix = np.zeros((n_sim, len(real_data)))
            
            for s_idx, idx in enumerate(sample_indices):
                w1_s = float(stacked['w1_s'][i, idx].values)
                vwin = float(stacked['vwin'][i, idx].values)
                vloss = float(stacked['vloss'][i, idx].values)
                beta = float(stacked['beta'][i, idx].values)
                
                sim_df = self.simulate_user(w1_s, vwin, vloss, beta, real_data)
                sim_pumps_matrix[s_idx, :] = sim_df['pumps'].values
                
            global_sim_matrix[:, start_idx : start_idx + len(real_data)] = sim_pumps_matrix
            start_idx += len(real_data)
                
            unpopped_mask = ~popped_mask
            real_adj = real_pumps[unpopped_mask].mean() if unpopped_mask.any() else 0
            sim_adj = sim_pumps_matrix[:, unpopped_mask].mean(axis=1) if unpopped_mask.any() else np.zeros(n_sim)
            ppp = np.mean(sim_adj >= real_adj)
            
            expected_pumps = np.mean(sim_pumps_matrix, axis=0) 
            r2 = r2_score(real_pumps, expected_pumps) if np.var(real_pumps) > 0 else np.nan
            rmse = np.sqrt(mean_squared_error(real_pumps, expected_pumps))
            mae = mean_absolute_error(real_pumps, expected_pumps)
            msd = np.mean((real_pumps - expected_pumps)**2)
            
            mode_res = mode(sim_pumps_matrix, axis=0, keepdims=False)
            mode_sim_pumps = np.squeeze(mode_res.mode if hasattr(mode_res, 'mode') else mode_res[0])
            hit_rate = np.mean(mode_sim_pumps == real_pumps)
            
            results.append({
                'user_id': uid, 'ppp': ppp, 'R2': r2, 'RMSE': rmse,
                'MAE': mae, 'Hit_Rate': hit_rate, 'MSD': msd
            })
            
        ppc_df = pd.DataFrame(results)
        print(f" -> Средний ppp: {ppc_df['ppp'].mean():.3f}, Средний R2: {ppc_df['R2'].mean():.3f}")
        
        plot_ppc_timecourse(self.data, global_sim_matrix, "STL_Model", window)
        return ppc_df

    def parameter_recovery(self, n_subjects=30, draws=1000, tune=1000, chains=4):
        if self.trace is None: raise ValueError("Сначала запустите fit()")
            
        print("Генерация истинных параметров из эмпирических гиперраспределений...")
        stacked = az.extract(self.trace)
        
        mu_w1_m, sig_w1_m = stacked['mu_w1'].mean().item(), stacked['sigma_w1'].mean().item()
        mu_vwin_m, sig_vwin_m = stacked['mu_vwin'].mean().item(), stacked['sigma_vwin'].mean().item()
        mu_vloss_m, sig_vloss_m = stacked['mu_vloss'].mean().item(), stacked['sigma_vloss'].mean().item()
        mu_beta_m, sig_beta_m = stacked['mu_beta'].mean().item(), stacked['sigma_beta'].mean().item()

        true_params = []
        simulated_data = []
        template_uid = self.user_ids[0]
        template_df = self.data[self.data['user_id'] == template_uid].sort_values('trial_number')
        
        for i in range(n_subjects):
            w1_t = expit(logit(mu_w1_m) + np.random.normal(0, 1) * sig_w1_m)
            vwin_t = expit(logit(mu_vwin_m) + np.random.normal(0, 1) * sig_vwin_m)
            vloss_t = expit(logit(mu_vloss_m) + np.random.normal(0, 1) * sig_vloss_m)
            beta_t = np.exp(np.log(mu_beta_m) + np.random.normal(0, 1) * sig_beta_m)
            
            true_params.append({'uid': f'sim_{i}', 'w1_s': w1_t, 'vwin': vwin_t, 'vloss': vloss_t, 'beta': beta_t})
            sim_df = self.simulate_user(w1_t, vwin_t, vloss_t, beta_t, template_df)
            sim_df['user_id'] = f'sim_{i}'
            simulated_data.append(sim_df)
            
        sim_full_df = pd.concat(simulated_data, ignore_index=True)
        true_df = pd.DataFrame(true_params)
        
        print(f"\nЗапуск HBA для восстановления (Fit on {n_subjects} Simulated Data)...")
        recov_model = STLModelHBA(sim_full_df, max_pumps=self.max_pumps)
        recov_trace = recov_model.fit(draws=draws, tune=tune, chains=chains)
        
        recov_summary = az.summary(recov_trace, var_names=['w1_s', 'vwin', 'vloss', 'beta'], hdi_prob=0.95)
        param_names = ['w1_s', 'vwin', 'vloss', 'beta']
        combined_data = []
        
        for i in range(n_subjects):
            row = {'user_id': f'sim_{i}'}
            for p in param_names:
                row[f'true_{p}'] = true_df.loc[i, p]
                idx_name = f"{p}[{i}]"
                
                if idx_name in recov_summary.index:
                    row[f'fit_{p}'] = recov_summary.loc[idx_name, 'mean']
                    lower = recov_summary.loc[idx_name, 'hdi_2.5%']
                    upper = recov_summary.loc[idx_name, 'hdi_97.5%']
                    row[f'coverage_{p}'] = 1 if lower <= row[f'true_{p}'] <= upper else 0
                else:
                    row[f'fit_{p}'] = np.nan
            combined_data.append(row)
            
        combined_df = pd.DataFrame(combined_data)
        metrics = []
        for p in param_names:
            t = combined_df[f'true_{p}'].values
            f = combined_df[f'fit_{p}'].values
            cov = combined_df[f'coverage_{p}'].values
            
            r, _ = pearsonr(t, f)
            r2 = r2_score(t, f)
            rmse = np.sqrt(mean_squared_error(t, f))
            bias = np.mean(f - t)
            coverage = np.mean(cov)
            
            metrics.append({'Parameter': p, 'r': r, 'R2': r2, 'RMSE': rmse, 'Bias': bias, 'Coverage': coverage})
            
        metrics_df = pd.DataFrame(metrics)
        print("\n=== Метрики Parameter Recovery ===")
        print(metrics_df.to_string(index=False))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for ax, p in zip(axes, param_names):
            t = combined_df[f'true_{p}'].values
            f = combined_df[f'fit_{p}'].values
            sns.scatterplot(x=t, y=f, ax=ax, color='teal', alpha=0.8)
            min_val, max_val = min(t.min(), f.min()), max(t.max(), f.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            metric_row = metrics_df[metrics_df['Parameter'] == p].iloc[0]
            ax.set_title(f"{p} (r={metric_row['r']:.3f}, R2={metric_row['R2']:.3f}, Cov={metric_row['Coverage']:.2f})")
            ax.set_xlabel('True Parameters')
            ax.set_ylabel('Recovered (Posterior Mean)')
            ax.grid(True, linestyle="--", alpha=0.5)
            
        plt.tight_layout()
        plt.savefig("bart_stl_hba_recovery.png", dpi=300)
        plt.close()
        
        return combined_df, metrics_df