import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
from scipy.special import expit
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # Критически важно для суперкомпьютера
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from tqdm import tqdm
from numba import njit
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. СТАНДАРТИЗАЦИЯ И ВИЗУАЛИЗАЦИЯ
# =============================================================================
def standardize_rewards(df, cols=['gain_amount', 'loss_amount', 'payout', 'penalty']):
    df_scaled = df.copy()
    for col in cols:
        if col in df_scaled.columns:
            max_abs_val = df_scaled[col].abs().max()
            if max_abs_val > 0:
                df_scaled[col] = (df_scaled[col] / max_abs_val) * 10.0
    return df_scaled

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

# =============================================================================
# 2. МАТЕМАТИЧЕСКОЕ ЯДРО (NUMBA + PYTENSOR OP ДЛЯ ПОТРИАЛЬНОГО LOOIC)
# =============================================================================
@njit(fastmath=True)
def wallsten_loglike_numba(gamma_plus, beta, q1, m0, pumps, popped, user_idx, max_pumps):
    n_trials = len(pumps)
    ll_arr = np.zeros(n_trials, dtype=np.float64) # ВАЖНО: Возвращаем массив!
    eps = 1e-12
    n_users = len(gamma_plus)
    
    # Numba позволяет нам быстро пробежаться по каждому пользователю
    for i in range(n_users):
        q = q1[i]
        m = m0[i]
        g = gamma_plus[i]
        b = beta[i]
        
        for j in range(n_trials):
            if user_idx[j] == i:
                pump = pumps[j]
                is_pop = popped[j]
                trial_ll = 0.0
                
                # g_h (с округлением по статье)
                q_clip = q
                if q_clip < eps: q_clip = eps
                if q_clip > 1.0 - eps: q_clip = 1.0 - eps
                g_h = np.round(-g / np.log(q_clip))
                
                # Вероятности накачек
                if pump > 0:
                    for k in range(1, pump + 1):
                        r_val = 1.0 / (1.0 + np.exp(-b * (g_h - k))) # sigmoid(b * (g_h - k))
                        trial_ll += np.log(r_val + eps)
                        
                # Вероятность остановки (если шар не лопнул)
                if not is_pop and pump < max_pumps:
                    r_stop = 1.0 / (1.0 + np.exp(-b * (g_h - (pump + 1))))
                    trial_ll += np.log(1.0 - r_stop + eps)
                
                # Записываем точечный (trial-level) log-likelihood
                ll_arr[j] = trial_ll
                
                # Обновление априорных представлений (q и m)
                a_h = (pump - 1.0) if is_pop else float(pump)
                if a_h < 0.0: a_h = 0.0
                m_h = float(pump)
                
                a_new = q * m + a_h
                m_new = m + m_h
                q = a_new / m_new
                m = m_new
                
    return ll_arr

@as_op(itypes=[pt.dvector, pt.dvector, pt.dvector, pt.dvector, 
               pt.lvector, pt.bvector, pt.lvector, pt.dscalar], 
       otypes=[pt.dvector]) # <-- ИСПРАВЛЕНО НА pt.dvector
def wallsten_loglike_op(gamma_plus, beta, q1, m0, pumps, popped, user_idx, max_pumps):
    return wallsten_loglike_numba(gamma_plus, beta, q1, m0, pumps, popped, user_idx, float(max_pumps))

# =============================================================================
# 3. КЛАСС МОДЕЛИ HBA
# =============================================================================
class Model3_Wallsten_HBA:
    def __init__(self, max_pumps=64):
        self.max_pumps = int(max_pumps)
        self.idata = None
        self.users = []
        
    def fit_hba(self, user_data, draws=1000, tune=1000, chains=4, cores=4):
        print("\n[*] Подготовка данных и инициализация PyMC HBA (Wallsten)...")
        df = standardize_rewards(user_data)
        df = df.sort_values(['user_id', 'trial_number']).reset_index(drop=True)
        
        self.users = df['user_id'].astype('category')
        user_idx_val = self.users.cat.codes.values.astype(np.int64)
        n_users = len(self.users.cat.categories)
        
        pumps_val = df['pumps'].values.astype(np.int64)
        popped_val = df['popped'].values.astype(np.int8)
        
        with pm.Model() as self.model:
            # Гиперпараметры
            mu_g = pm.Normal('mu_g', mu=0, sigma=1)
            sig_g = pm.HalfNormal('sig_g', sigma=1)
            mu_b = pm.Normal('mu_b', mu=0, sigma=1)
            sig_b = pm.HalfNormal('sig_b', sigma=1)
            mu_q = pm.Normal('mu_q', mu=0, sigma=1)
            sig_q = pm.HalfNormal('sig_q', sigma=1)
            mu_m = pm.Normal('mu_m', mu=0, sigma=1)
            sig_m = pm.HalfNormal('sig_m', sigma=1)
            
            # Индивидуальные сырые параметры (Non-centered parameterization)
            g_raw = pm.Normal('g_raw', mu=0, sigma=1, shape=n_users)
            b_raw = pm.Normal('b_raw', mu=0, sigma=1, shape=n_users)
            q_raw = pm.Normal('q_raw', mu=0, sigma=1, shape=n_users)
            m_raw = pm.Normal('m_raw', mu=0, sigma=1, shape=n_users)
            
            # Преобразование в реальные границы
            gamma_plus = pm.Deterministic('gamma_plus', 5.0 * pm.math.invlogit(mu_g + sig_g * g_raw))  # [0, 5]
            beta = pm.Deterministic('beta', 0.2 + 9.8 * pm.math.invlogit(mu_b + sig_b * b_raw))        # [0.2, 10]
            q1 = pm.Deterministic('q1', 0.01 + 0.98 * pm.math.invlogit(mu_q + sig_q * q_raw))          # [0.01, 0.99]
            m0 = pm.Deterministic('m0', 2.0 + 49998.0 * pm.math.invlogit(mu_m + sig_m * m_raw))        # [2, 50000]
            
            # Тензоры данных
            pumps_pt = pt.as_tensor_variable(pumps_val)
            popped_pt = pt.as_tensor_variable(popped_val)
            user_idx_pt = pt.as_tensor_variable(user_idx_val)
            max_pumps_pt = pt.constant(self.max_pumps, dtype='float64')

            # ВАЖНО: Вычисляем векторный Log-likelihood для LOOIC
            ll_vec = wallsten_loglike_op(gamma_plus, beta, q1, m0, pumps_pt, popped_pt, user_idx_pt, max_pumps_pt)
            pm.Deterministic('log_lik', ll_vec)
            pm.Potential('likelihood', pt.sum(ll_vec))
            
            print(f"[*] Сэмплирование (chains={chains}, draws={draws})...")
            step = pm.DEMetropolisZ() # <--- Явный вызов безградиентного сэмплера для функции с @as_op
            self.idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, step=step, progressbar=False, return_inferencedata=True)

            # Интеграция с ArviZ
            self.idata.add_groups({"log_likelihood": {"log_lik": self.idata.posterior["log_lik"]}})
            
        # Сразу проверяем сходимость и LOOIC
        loo = az.loo(self.idata, var_name="log_lik")
        print(f"\n[FIT METRICS] LOO (elpd): {loo.elpd_loo:.2f}")
        return self.idata

    # =========================================================================
    # 4. СИМУЛЯЦИЯ И POSTERIOR PREDICTIVE CHECK (PPC)
    # =========================================================================
    def simulate_user(self, gamma, beta, q1, m0, real_user_df, seed=None):
        rng = np.random.default_rng(seed)
        q, m = float(q1), float(m0)
        sim_pumps = []
        
        for _, row in real_user_df.iterrows():
            explosion_point = int(row['explosion_point']) if 'explosion_point' in row else rng.integers(1, self.max_pumps + 1)
            g_h = np.round(-gamma / np.log(np.clip(q, 1e-12, 1-1e-12)))
            
            j = 0
            popped = False
            while True:
                j += 1
                r = expit(beta * (g_h - j))
                if rng.random() >= r:
                    popped = False
                    pumps = j - 1
                    break
                if j >= explosion_point:
                    popped = True
                    pumps = j
                    break
                if j >= self.max_pumps:
                    popped = False
                    pumps = j
                    break
            
            sim_pumps.append(pumps)
            
            # Обновление
            a_h = max(0, pumps - 1) if popped else pumps
            q = (q * m + a_h) / (m + pumps)
            m = m + pumps
            
        return np.array(sim_pumps)

    def predictive_check(self, df, n_sims=50, seed=42):
        if self.idata is None: raise ValueError("Сначала запустите fit_hba()!")
        print("\n=== Запуск Posterior Predictive Check (PPC) ===")
        
        from scipy.stats import mode
        rng = np.random.default_rng(seed)
        post = self.idata.posterior
        n_chains, n_draws = post['gamma_plus'].shape[:2]
        
        df = df.sort_values(['user_id', 'trial_number'])
        unique_users = df['user_id'].unique()
        metrics = []
        
        global_sim_matrix = np.zeros((n_sims, len(df)))
        start_idx = 0
        
        for i, uid in enumerate(tqdm(unique_users, desc="PPC Users")):
            user_df = df[df['user_id'] == uid]
            real_pumps = user_df['pumps'].values
            popped_mask = user_df['popped'].values.astype(bool)
            sim_matrix = np.zeros((n_sims, len(user_df)))
            
            for s in range(n_sims):
                c, d = rng.integers(0, n_chains), rng.integers(0, n_draws)
                g_samp = float(post['gamma_plus'][c, d, i])
                b_samp = float(post['beta'][c, d, i])
                q_samp = float(post['q1'][c, d, i])
                m_samp = float(post['m0'][c, d, i])
                
                sim_matrix[s, :] = self.simulate_user(g_samp, b_samp, q_samp, m_samp, user_df, seed=int(rng.integers(1e6)))
                
            global_sim_matrix[:, start_idx : start_idx + len(user_df)] = sim_matrix
            start_idx += len(user_df)
                
            unpopped_mask = ~popped_mask
            real_adj = real_pumps[unpopped_mask].mean() if unpopped_mask.any() else 0
            sim_adj = sim_matrix[:, unpopped_mask].mean(axis=1) if unpopped_mask.any() else np.zeros(n_sims)
            ppp = np.mean(sim_adj >= real_adj)
            
            mean_sim = sim_matrix.mean(axis=0)
            r2 = r2_score(real_pumps, mean_sim) if np.var(real_pumps) > 0 else np.nan
            rmse = np.sqrt(mean_squared_error(real_pumps, mean_sim))
            mae = mean_absolute_error(real_pumps, mean_sim)
            msd = np.mean((real_pumps - mean_sim)**2)
            
            mode_res = mode(sim_matrix, axis=0, keepdims=False)
            mode_sim = np.squeeze(mode_res.mode if hasattr(mode_res, 'mode') else mode_res[0])
            hit_rate = np.mean(mode_sim == real_pumps)
            
            metrics.append({
                'user_id': uid, 'ppp': ppp, 'R2': r2, 'RMSE': rmse, 
                'MAE': mae, 'Hit_Rate': hit_rate, 'MSD': msd
            })

        ppc_df = pd.DataFrame(metrics)
        print(f" -> Средний ppp: {ppc_df['ppp'].mean():.3f}, Средний R2: {ppc_df['R2'].mean():.3f}")
        plot_ppc_timecourse(df, global_sim_matrix, "Wallsten_Model")
        return ppc_df

    # =========================================================================
    # 5. HIERARCHICAL PARAMETER RECOVERY
    # =========================================================================
    def parameter_recovery(self, template_df, n_subjects=40, n_trials=60):
        if self.idata is None: raise ValueError("Сначала запустите fit_hba()!")
        print(f"\n=== Запуск Hierarchical Parameter Recovery (N={n_subjects}) ===")
        
        post = self.idata.posterior
        rng = np.random.default_rng(123)
        c, d = rng.integers(0, post['mu_g'].shape[0]), rng.integers(0, post['mu_g'].shape[1])
        
        m_g, s_g = post['mu_g'][c, d].values, post['sig_g'][c, d].values
        m_b, s_b = post['mu_b'][c, d].values, post['sig_b'][c, d].values
        m_q, s_q = post['mu_q'][c, d].values, post['sig_q'][c, d].values
        m_m, s_m = post['mu_m'][c, d].values, post['sig_m'][c, d].values
        
        true_params = []
        sim_data_list = []
        
        for i in range(n_subjects):
            g_true = 5.0 * expit(rng.normal(m_g, s_g))
            b_true = 0.2 + 9.8 * expit(rng.normal(m_b, s_b))
            q_true = 0.01 + 0.98 * expit(rng.normal(m_q, s_q))
            m_true = 2.0 + 49998.0 * expit(rng.normal(m_m, s_m))
            true_params.append([g_true, b_true, q_true, m_true])
            
            user_template = template_df[template_df['user_id'] == template_df['user_id'].unique()[0]].copy().head(n_trials)
            user_template['user_id'] = f"synth_{i}"
            
            sim_pumps = self.simulate_user(g_true, b_true, q_true, m_true, user_template, seed=1000+i)
            user_template['pumps'] = sim_pumps
            user_template['popped'] = user_template['pumps'] >= user_template.get('explosion_point', self.max_pumps)
            sim_data_list.append(user_template)
            
        sim_df = pd.concat(sim_data_list, ignore_index=True)
        true_arr = np.array(true_params)
        
        recovery_model = Model3_Wallsten_HBA(max_pumps=self.max_pumps)
        recov_idata = recovery_model.fit_hba(sim_df, draws=1000, tune=1000, chains=4, cores=4)
        
        recov_post = recov_idata.posterior
        param_names = ['gamma_plus', 'beta', 'q1', 'm0']
        recovery_dict = {'user_id': [f"synth_{i}" for i in range(n_subjects)]}
        metrics = []
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for idx, (p_name, ax) in enumerate(zip(param_names, axes.flatten())):
            true_vals = true_arr[:, idx]
            fit_means = recov_post[p_name].mean(dim=['chain', 'draw']).values
            
            hdi = az.hdi(recov_idata, var_names=[p_name], hdi_prob=0.95)[p_name].values
            cov_arr = (true_vals >= hdi[:, 0]) & (true_vals <= hdi[:, 1])
            coverage = np.mean(cov_arr)
            
            r, _ = pearsonr(true_vals, fit_means)
            r2 = r2_score(true_vals, fit_means)
            rmse = np.sqrt(mean_squared_error(true_vals, fit_means))
            bias = np.mean(fit_means - true_vals)
            
            recovery_dict[f'true_{p_name}'] = true_vals
            recovery_dict[f'fit_{p_name}'] = fit_means
            recovery_dict[f'coverage_{p_name}'] = cov_arr.astype(int)
            
            metrics.append({'Parameter': p_name, 'r': r, 'R2': r2, 'RMSE': rmse, 'Bias': bias, 'Coverage': coverage})
            
            ax.scatter(true_vals, fit_means, alpha=0.7, color='purple')
            min_v, max_v = min(true_vals.min(), fit_means.min()), max(true_vals.max(), fit_means.max())
            ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2)
            ax.set_title(f"{p_name}\nr={r:.2f}, R2={r2:.2f}\nCov={coverage:.2f}")
            ax.set_xlabel("True (Generative) Parameter")
            ax.set_ylabel("Fitted (Posterior Mean)")
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig('bart_wallsten_recovery_plot.png', dpi=300)
        plt.close()
        
        return pd.DataFrame(recovery_dict), pd.DataFrame(metrics)