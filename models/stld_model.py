import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import arviz as az
import matplotlib
matplotlib.use('Agg')  # КРИТИЧЕСКИ ВАЖНО ДЛЯ СУПЕРКОМПЬЮТЕРА (отключение X11)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from tqdm import tqdm
import warnings
from numba import njit

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
# 1. Оптимизированное ядро Log-Likelihood (Numba)
# ==========================================
@njit
def logp_numba_core(w1_norm, vwin, vloss, a, beta, pumps_arr, popped_arr, nmax, scale_factor):
    # Масштабируем параметры, чтобы логиты лежали в рамках [-10, 10]
    w = w1_norm * (nmax / scale_factor)
    nmax_scaled = nmax / scale_factor
    n_trials = len(pumps_arr)
    
    logp_arr = np.zeros(n_trials, dtype=np.float64)
    eps = 1e-10
    
    for t in range(n_trials):
        pumps = int(pumps_arr[t])
        popped = bool(popped_arr[t])
        pumps_scaled = pumps / scale_factor
        trial_logp = 0.0
        
        if popped:
            # Если шарик лопнул, участник качал до упора
            for k in range(1, pumps + 1):
                k_scaled = k / scale_factor
                z = beta * (k_scaled - w)
                if z > 700.0: z = 700.0
                elif z < -700.0: z = -700.0
                p_pump = 1.0 / (1.0 + np.exp(z))
                trial_logp += np.log(p_pump + eps)
        else:
            # Если шарик не лопнул, участник качал, а потом остановился
            for k in range(1, pumps + 2):
                k_scaled = k / scale_factor
                z = beta * (k_scaled - w)
                if z > 700.0: z = 700.0
                elif z < -700.0: z = -700.0
                p_pump = 1.0 / (1.0 + np.exp(z))
                
                if k <= pumps:
                    trial_logp += np.log(p_pump + eps) 
                else:
                    trial_logp += np.log(1.0 - p_pump + eps) 
        
        logp_arr[t] = trial_logp
        
        # Обновление w со спадом (decay)
        if popped:
            w *= 1.0 - (vloss * (1.0 - pumps_scaled / nmax_scaled)) / (1.0 + a * t)
        else:
            w *= 1.0 + (vwin * pumps_scaled / nmax_scaled) / (1.0 + a * t)
            
    return logp_arr

# ==========================================
# 2. Обертка для PyMC (Векторный выход для LOOIC)
# ==========================================
@as_op(itypes=[pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dvector, pt.dvector, pt.dscalar, pt.dscalar], 
       otypes=[pt.dvector])
def pytensor_stld_logp(w1_norm, vwin, vloss, a, beta, pumps_arr, popped_arr, nmax, scale_factor):
    val = logp_numba_core(float(w1_norm), float(vwin), float(vloss), float(a), float(beta), 
                          pumps_arr, popped_arr, float(nmax), float(scale_factor))
    return np.array(val, dtype=np.float64)

# ==========================================
# 3. Класс Модели и Проверок
# ==========================================
class STLDModelHBA:
    def __init__(self, data_df, nmax=64, scale_factor=10.0):
        self.data = data_df.copy()
        self.nmax = float(nmax)
        self.scale_factor = float(scale_factor)
        self.trace = None
        
        # Индексация пользователей
        self.users = sorted(self.data['user_id'].unique())
        self.n_users = len(self.users)
        self.user_map = {u: i for i, u in enumerate(self.users)}
        self.data['user_idx'] = self.data['user_id'].map(self.user_map)

    def fit_hba(self, draws=1000, tune=1000, chains=4, cores=4):
        print(f"🚀 Запуск HBA MCMC (Участников: {self.n_users}, Масштабирование: {self.scale_factor})...")
        
        pumps_list = [self.data[self.data['user_idx'] == i].sort_values('trial_number')['pumps'].to_numpy(dtype=np.float64) for i in range(self.n_users)]
        popped_list = [self.data[self.data['user_idx'] == i].sort_values('trial_number')['popped'].to_numpy(dtype=np.float64) for i in range(self.n_users)]

        with pm.Model() as model:
            # Гиперпараметры (групповой уровень)
            mu_w1 = pm.Normal('mu_w1', 0, 1)
            sigma_w1 = pm.Normal('sigma_w1', 0, 0.2)
            mu_vwin = pm.Normal('mu_vwin', 0, 1)
            sigma_vwin = pm.Normal('sigma_vwin', 0, 0.2)
            mu_vloss = pm.Normal('mu_vloss', 0, 1)
            sigma_vloss = pm.Normal('sigma_vloss', 0, 0.2)
            mu_a = pm.Normal('mu_a', 0, 1)
            sigma_a = pm.Normal('sigma_a', 0, 0.2)
            mu_beta = pm.Normal('mu_beta', 0, 1)
            sigma_beta = pm.HalfCauchy('sigma_beta', 1)

            # Индивидуальный уровень (Non-centered)
            w1_pr = pm.Normal('w1_pr', 0, 1, shape=self.n_users)
            vwin_pr = pm.Normal('vwin_pr', 0, 1, shape=self.n_users)
            vloss_pr = pm.Normal('vloss_pr', 0, 1, shape=self.n_users)
            a_pr = pm.Normal('a_pr', 0, 1, shape=self.n_users)
            beta_pr = pm.Normal('beta_pr', 0, 1, shape=self.n_users)

            # Трансформации в границы
            w1_norm = pm.Deterministic('w1_norm', pm.math.invprobit(mu_w1 + sigma_w1 * w1_pr))
            vwin = pm.Deterministic('vwin', pm.math.invprobit(mu_vwin + sigma_vwin * vwin_pr))
            vloss = pm.Deterministic('vloss', pm.math.invprobit(mu_vloss + sigma_vloss * vloss_pr))
            a = pm.Deterministic('a', 0.1 * pm.math.invprobit(mu_a + sigma_a * a_pr))
            beta = pm.Deterministic('beta', 10.0 * pm.math.invprobit(mu_beta + sigma_beta * beta_pr))

            nmax_pt = pt.constant(self.nmax, dtype='float64')
            scale_pt = pt.constant(self.scale_factor, dtype='float64')

            logp_list = []
            for i in range(self.n_users):
                pumps_pt = pt.as_tensor_variable(pumps_list[i], dtype='float64')
                popped_pt = pt.as_tensor_variable(popped_list[i], dtype='float64')
                
                logp_var = pytensor_stld_logp(w1_norm[i], vwin[i], vloss[i], a[i], beta[i], pumps_pt, popped_pt, nmax_pt, scale_pt)
                logp_list.append(logp_var)
            
            # Собираем все log_likelihood в один тензор для ArviZ LOOIC
            log_likelihood = pt.concatenate(logp_list)
            pm.Deterministic('log_likelihood', log_likelihood)
            pm.Potential('obs', pt.sum(log_likelihood))

            # Сэмплирование (без progressbar для HPC)
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, progressbar=False, return_inferencedata=True)
            self.trace.add_groups(
                {"log_likelihood": {"obs": self.trace.posterior["log_likelihood"]}}
            )
        return self.trace

    def get_individual_posteriors(self):
        if self.trace is None: return None
        post = self.trace.posterior
        results = []
        for i, uid in enumerate(self.users):
            results.append({
                'user_id': uid,
                'w1': float(post['w1_norm'].mean(dim=['chain', 'draw'])[i]) * self.nmax,
                'vwin': float(post['vwin'].mean(dim=['chain', 'draw'])[i]),
                'vloss': float(post['vloss'].mean(dim=['chain', 'draw'])[i]),
                'alpha': float(post['a'].mean(dim=['chain', 'draw'])[i]),
                'beta': float(post['beta'].mean(dim=['chain', 'draw'])[i])
            })
        return pd.DataFrame(results)

    def simulate(self, params, real_user_data, seed=None):
        w1_norm, vwin, vloss, a, beta = params
        w = w1_norm * (self.nmax / self.scale_factor)
        nmax_scaled = self.nmax / self.scale_factor
        data = []
        rng = np.random.RandomState(seed)

        real_user_data = real_user_data.sort_values('trial_number')
        
        for t_idx, row in enumerate(real_user_data.itertuples()):
            t = row.trial_number
            kappa = int(row.break_point) 
            k = 0
            popped = False

            while True:
                k += 1
                k_scaled = k / self.scale_factor
                z = np.clip(beta * (k_scaled - w), -700, 700)
                p_pump = 1.0 / (1.0 + np.exp(z))
                
                if rng.rand() > p_pump:
                    popped = False
                    k -= 1 
                    break
                
                if k >= kappa:
                    popped = True 
                    break

            data.append({'trial_number': t, 'pumps': k, 'popped': popped, 'break_point': kappa})

            k_scaled_final = k / self.scale_factor
            if popped:
                w *= 1.0 - (vloss * (1.0 - k_scaled_final / nmax_scaled)) / (1.0 + a * t_idx)
            else:
                w *= 1.0 + (vwin * k_scaled_final / nmax_scaled) / (1.0 + a * t_idx)

        return pd.DataFrame(data)

    def posterior_predictive_check(self, n_sims=100):
        if self.trace is None: raise ValueError("Сначала запустите fit_hba().")
        print("\n=== Проведение True HBA Posterior Predictive Check (PPC) ===")
        
        from scipy.stats import mode
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        post = self.trace.posterior
        n_chains = post.sizes['chain']
        n_draws = post.sizes['draw']
        
        ppc_results = []
        global_sim_matrix = np.zeros((n_sims, len(self.data))) # <-- НОВОЕ
        start_idx = 0

        for i, uid in enumerate(tqdm(self.users, desc="PPC (Trial-by-trial)")):
            user_df = self.data[self.data['user_id'] == uid].sort_values('trial_number')
            real_pumps = user_df['pumps'].values
            popped_mask = user_df['popped'].values.astype(bool)
            
            sim_pumps_matrix = np.zeros((n_sims, len(user_df)))
            
            for s_idx in range(n_sims):
                c = np.random.randint(0, n_chains)
                d = np.random.randint(0, n_draws)
                
                p_sim = [float(post[v][c, d, i]) for v in ['w1_norm', 'vwin', 'vloss', 'a', 'beta']]
                sim_df = self.simulate(p_sim, user_df, seed=np.random.randint(100000))
                sim_pumps_matrix[s_idx, :] = sim_df['pumps'].values
                
            global_sim_matrix[:, start_idx : start_idx + len(user_df)] = sim_pumps_matrix
            start_idx += len(user_df)
                
            unpopped_mask = ~popped_mask
            real_adj = real_pumps[unpopped_mask].mean() if unpopped_mask.any() else 0
            sim_adj = sim_pumps_matrix[:, unpopped_mask].mean(axis=1) if unpopped_mask.any() else np.zeros(n_sims)
            ppp = np.mean(sim_adj >= real_adj)

            expected_pumps = sim_pumps_matrix.mean(axis=0)
            r2 = r2_score(real_pumps, expected_pumps) if np.var(real_pumps) > 0 else np.nan
            rmse = np.sqrt(mean_squared_error(real_pumps, expected_pumps))
            mae = mean_absolute_error(real_pumps, expected_pumps)
            msd = np.mean((real_pumps - expected_pumps)**2)
            
            mode_res = mode(sim_pumps_matrix, axis=0, keepdims=False)
            mode_sim_pumps = np.squeeze(mode_res.mode if hasattr(mode_res, 'mode') else mode_res[0])
            hit_rate = np.mean(mode_sim_pumps == real_pumps)
            
            ppc_results.append({
                'user_id': uid, 'ppp': ppp, 'R2': r2, 'RMSE': rmse, 
                'MAE': mae, 'Hit_Rate': hit_rate, 'MSD': msd
            })

        ppc_df = pd.DataFrame(ppc_results)
        print(f" -> Средний ppp: {ppc_df['ppp'].mean():.3f}, Средний R2: {ppc_df['R2'].mean():.3f}")
        
        # --- НОВОЕ: Отрисовка Timecourse ---
        plot_ppc_timecourse(self.data, global_sim_matrix, "STLD_Model")

        return ppc_df

    def parameter_recovery(self, n_subjects=50):
        if self.trace is None: raise ValueError("Сначала запустите fit_hba().")
        print(f"\n=== Проведение True HBA Parameter Recovery (N={n_subjects}) ===")
        
        post = self.trace.posterior
        chains, draws = post.sizes['chain'], post.sizes['draw']
        total_samples = chains * draws
        
        sample_indices = np.random.choice(total_samples, size=n_subjects, replace=False)
        true_params_list = []
        sim_data_frames = []
        
        template_user_df = self.data[self.data['user_idx'] == 0].sort_values('trial_number').reset_index(drop=True)
        
        for s_idx, flat_idx in enumerate(sample_indices):
            c, d = flat_idx // draws, flat_idx % draws
            u = np.random.randint(0, self.n_users)
            
            p_true = {
                'w1_norm': float(post['w1_norm'][c, d, u]),
                'vwin': float(post['vwin'][c, d, u]),
                'vloss': float(post['vloss'][c, d, u]),
                'a': float(post['a'][c, d, u]),
                'beta': float(post['beta'][c, d, u])
            }
            true_params_list.append(p_true)
            
            p_vec = [p_true['w1_norm'], p_true['vwin'], p_true['vloss'], p_true['a'], p_true['beta']]
            sim_df = self.simulate(p_vec, template_user_df, seed=s_idx)
            sim_df['user_id'] = f"sim_{s_idx}"
            sim_data_frames.append(sim_df)

        synth_dataset = pd.concat(sim_data_frames, ignore_index=True)
        
        print("[*] Переобучение иерархической модели на синтетических данных (True PR)...")
        recovery_model = STLDModelHBA(synth_dataset, nmax=self.nmax, scale_factor=self.scale_factor)
        rec_trace = recovery_model.fit_hba(draws=500, tune=500, chains=2, cores=2) 
        
        rec_post = rec_trace.posterior
        param_names = ['w1_norm', 'vwin', 'vloss', 'a', 'beta']
        
        recovery_results = []
        for i in range(n_subjects):
            row = {'user_id': f"sim_{i}"}
            for p in param_names:
                true_val = true_params_list[i][p]
                fit_val = float(rec_post[p].mean(dim=['chain', 'draw'])[i])
                
                # Схлопываем сэмплы всех цепей в одномерный массив (chains * draws)
                flat_samples = rec_post[p].values[:, :, i].flatten()
                hdi = az.hdi(flat_samples, hdi_prob=0.95)
                
                # Теперь hdi — это гарантированно список/массив из 2-х скаляров: [нижняя_граница, верхняя_граница]
                coverage = 1 if (hdi[0] <= true_val <= hdi[1]) else 0
                
                row[f'true_{p}'] = true_val
                row[f'fit_{p}'] = fit_val
                row[f'coverage_{p}'] = coverage
            recovery_results.append(row)
            
        rec_df = pd.DataFrame(recovery_results)
        
        metrics = []
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for idx, p in enumerate(param_names):
            t_v = rec_df[f'true_{p}'].values
            f_v = rec_df[f'fit_{p}'].values
            
            r_val, _ = pearsonr(t_v, f_v)
            r2_val = r2_score(t_v, f_v) # ДОБАВЛЕНО
            rmse_val = np.sqrt(mean_squared_error(t_v, f_v))
            bias_val = np.mean(f_v - t_v)
            cov_val = rec_df[f'coverage_{p}'].mean()
            
            metrics.append({'Parameter': p, 'r': r_val, 'R2': r2_val, 'RMSE': rmse_val, 'Bias': bias_val, 'Coverage': cov_val})
            
            axes[idx].scatter(t_v, f_v, alpha=0.7, color='coral')
            axes[idx].plot([t_v.min(), t_v.max()], [t_v.min(), t_v.max()], 'k--', lw=2)
            axes[idx].set_title(f"{p}\nr={r_val:.2f}, R2={r2_val:.2f}\nBias={bias_val:.3f}, Cov={cov_val:.0%}", fontsize=11)
            axes[idx].set_xlabel("True Parameter")
            axes[idx].set_ylabel("Recovered (Fit)")
            axes[idx].grid(True, alpha=0.3)
            
        fig.tight_layout()
        fig.savefig('bart_stld_recovery_plot.png', dpi=300)
        plt.close(fig)

        metrics_df = pd.DataFrame(metrics)
        return rec_df, metrics_df