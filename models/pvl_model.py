import os
import matplotlib
matplotlib.use('Agg') # Обязательно для суперкомпьютера

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.special import expit
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# =========================================================================
# ОСНОВНОЙ КЛАСС МОДЕЛИ HBA (С ИСПОЛЬЗОВАНИЕМ PYTENSOR SCAN)
# =========================================================================
class PVLModelHBA:
    def __init__(self, data_df=None, update_rule="delta"):
        self.data = None if data_df is None else data_df.copy().reset_index(drop=True)
        self.update_rule = update_rule.lower()
        assert self.update_rule in ["delta", "decay"], "update_rule must be 'delta' or 'decay'"
        
        self.trace = None
        self.scaling_factor = 100.0 # Стандартизация наград [-12.5, 1.0]

    def _prepare_tensors(self, df):
        """Подготовка транспонированных тензоров [n_trials, n_users] для pytensor.scan"""
        users = df['user_id'].unique()
        n_subj = len(users)
        n_trials = df.groupby('user_id').size().max()
        
        choices = np.full((n_trials, n_subj), -1, dtype=np.int32)
        wins = np.zeros((n_trials, n_subj), dtype=np.float64)
        losses = np.zeros((n_trials, n_subj), dtype=np.float64)
        mask = np.zeros((n_trials, n_subj), dtype=bool)
        
        # [ИСПРАВЛЕНИЕ]: Динамическое определение названий колонок
        # В реальных данных это 'payout' и 'penalty', в симуляциях 'win' и 'loss'
        win_col = 'win' if 'win' in df.columns else 'payout'
        loss_col = 'loss' if 'loss' in df.columns else 'penalty'
        
        for i, uid in enumerate(users):
            u_df = df[df['user_id'] == uid].sort_values('trial_number')
            n_t = len(u_df)
            
            choices[:n_t, i] = u_df['deck_num'].values
            
            # Масштабируем награды
            wins[:n_t, i] = u_df[win_col].values / self.scaling_factor
            
            # [ИСПРАВЛЕНИЕ]: Штраф обязательно должен быть отрицательным,
            # так как шаг делает `x = win_t + loss_t`. Защищаем через -np.abs()
            losses[:n_t, i] = -np.abs(u_df[loss_col].values) / self.scaling_factor
            
            mask[:n_t, i] = True
            
        return choices, wins, losses, mask, users

    def fit(self, draws=1000, tune=1000, chains=4, cores=4):
        choices, wins, losses, mask, self.users = self._prepare_tensors(self.data)
        n_subj = len(self.users)
        n_trials = choices.shape[0]
        
        print(f"[*] Старт HBA NUTS: PVL-{self.update_rule.capitalize()} (Субъектов: {n_subj}).")
        
        with pm.Model() as self.model:
            # 1. Гиперпараметры (Групповой уровень)
            mu_A = pm.Normal('mu_A', mu=0, sigma=1.5)
            sd_A = pm.HalfNormal('sd_A', sigma=1.0)
            mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=1.5)
            sd_alpha = pm.HalfNormal('sd_alpha', sigma=1.0)
            mu_lam = pm.Normal('mu_lam', mu=0, sigma=1.5)
            sd_lam = pm.HalfNormal('sd_lam', sigma=1.0)
            mu_c = pm.Normal('mu_c', mu=0, sigma=1.5)
            sd_c = pm.HalfNormal('sd_c', sigma=1.0)
            
            # 2. Индивидуальный уровень (Non-centered parameterization)
            A_raw = pm.Normal('A_raw', mu=0, sigma=1, shape=n_subj)
            alpha_raw = pm.Normal('alpha_raw', mu=0, sigma=1, shape=n_subj)
            lam_raw = pm.Normal('lam_raw', mu=0, sigma=1, shape=n_subj)
            c_raw = pm.Normal('c_raw', mu=0, sigma=1, shape=n_subj)
            
            # 3. Проекция параметров в нужные границы
            A = pm.Deterministic('A', pm.math.invlogit(mu_A + sd_A * A_raw))               # [0, 1]
            alpha = pm.Deterministic('alpha', pm.math.invlogit(mu_alpha + sd_alpha * alpha_raw)) # [0, 1]
            lam = pm.Deterministic('lam', pm.math.invlogit(mu_lam + sd_lam * lam_raw) * 10.0)    # [0, 10]
            c = pm.Deterministic('c', pm.math.invlogit(mu_c + sd_c * c_raw) * 5.0)               # [0, 5]
            
            # Константы данных
            choices_pt = pm.ConstantData('choices', choices)
            wins_pt = pm.ConstantData('wins', wins)
            losses_pt = pm.ConstantData('losses', losses)
            mask_pt = pm.ConstantData('mask', mask)
            
            # Безопасные индексы (заменяем -1 на 0 для работы тензоров, маска скроет эффект)
            choices_safe = pt.clip(choices_pt, 0, 3)
            
            # 4. Векторизованная логика шага для Scan
            # 4. Векторизованная логика шага для Scan
            def step_fn(choice_t, win_t, loss_t, mask_t, Q_prev, A, alpha, lam, c):
                theta = (3.0 ** c) - 1.0
                
                # Расчет Softmax
                v = Q_prev * theta[:, None]
                v = v - pt.max(v, axis=1, keepdims=True)
                ex = pt.exp(v)
                probs = ex / (pt.sum(ex, axis=1, keepdims=True) + 1e-12)
                
                # Полезность исхода
                x = win_t + loss_t
                u = pt.switch(x >= 0, (x + 1e-12) ** alpha, -lam * ((-x + 1e-12) ** alpha))
                
                idx = pt.arange(A.shape[0])
                
                # Обновление Q-values (Универсально: Delta или Decay)
                if self.update_rule == "delta":
                    Q_chosen = Q_prev[idx, choice_t]
                    Q_updated = Q_chosen + A * (u - Q_chosen)
                    Q_new = pt.set_subtensor(Q_prev[idx, choice_t], Q_updated)
                else: # decay
                    Q_decay = Q_prev * A[:, None]
                    Q_updated = Q_decay[idx, choice_t] + u
                    Q_new = pt.set_subtensor(Q_decay[idx, choice_t], Q_updated)
                
                # Применяем маску: если триала не было, оставляем старое Q
                Q_new = pt.switch(mask_t[:, None], Q_new, Q_prev)
                
                return Q_new, probs

            # 5. Запуск цикла Scan
            Q_init = pt.zeros((n_subj, 4), dtype='float64')
            
            [_, probs_seq], _ = scan(
                fn=step_fn,
                sequences=[choices_safe, wins_pt, losses_pt, mask_pt],
                outputs_info=[Q_init, None],
                non_sequences=[A, alpha, lam, c],  # Убрали is_delta отсюда
                strict=True
            )

            # 6. Извлекаем вероятности реально выбранных карт
            chosen_probs = probs_seq[pt.arange(n_trials)[:, None], pt.arange(n_subj)[None, :], choices_safe]
            
            # 7. Расчет и сохранение Log-Likelihood для LOOIC
            logp_trial = pt.log(pt.clip(chosen_probs, 1e-12, 1.0)) * mask_pt
            pm.Deterministic('log_lik', logp_trial)
            pm.Potential('obs_logp', pt.sum(logp_trial))
            
            # Запуск (Теперь работает NUTS, а не медленный Slice)
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                                   target_accept=0.95, return_inferencedata=True, progressbar=False)
            
            self.trace.add_groups(
                {"log_likelihood": {"obs": self.trace.posterior["log_lik"]}}
            )
            
        return self.trace

    # =========================================================================
    # СИМУЛЯЦИЯ И POSTERIOR PREDICTIVE CHECK (БЛОЧНЫЙ)
    # =========================================================================
    def simulate_subject(self, params, n_trials=150, seed=None):
        rng = np.random.RandomState(seed)
        A, alpha, lam, c = params
        theta = 3.0 ** c - 1.0
        Q = np.zeros(4, dtype=np.float64)
        
        sim_choices = []
        for t in range(n_trials):
            v = Q * theta
            v -= np.max(v)
            ex = np.exp(v)
            probs = ex / (np.sum(ex) + 1e-12)
            
            deck = rng.choice(4, p=probs)
            sim_choices.append(deck)
            
            if deck == 0:   win, loss = 1.0, -2.5 if rng.rand() < 0.5 else 0.0
            elif deck == 1: win, loss = 1.0, -6.25 if rng.rand() < 0.2 else 0.0
            elif deck == 2: win, loss = 0.5, -0.5 if rng.rand() < 0.5 else 0.0
            else:           win, loss = 0.5, -1.25 if rng.rand() < 0.2 else 0.0

            x = win + loss
            # Защита от переполнения (overflow) при alpha/loss aversion
            u = (x + 1e-12) ** alpha if x >= 0 else -lam * ((-x + 1e-12) ** alpha)
            u = np.clip(u, -20.0, 20.0) # ИСПРАВЛЕНИЕ: Стабилизация полезности исхода
            
            if self.update_rule == "delta":
                Q[deck] = Q[deck] + A * (u - Q[deck])
            else: # decay
                Q = Q * A
                Q[deck] += u
            
        return np.array(sim_choices)

    def posterior_predictive_check(self, n_sims=100, blocks=5, save_path="igt_pvl_ppc.png", seed=42):
        print(f"[*] Запуск PPC (PVL-{self.update_rule.capitalize()})...")
            
        post = self.trace.posterior
        n_trials = self.data.groupby('user_id').size().max()
        block_size = n_trials // blocks
        
        n_chains = post.dims['chain']
        n_draws = post.dims['draw']
        total_samples = n_chains * n_draws
        
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(total_samples, size=n_sims, replace=False if total_samples >= n_sims else True)
        
        real_matrix = np.full((len(self.users), n_trials), np.nan)
        sim_matrix = np.full((n_sims, len(self.users), n_trials), np.nan)
        
        all_metrics = []
        
        for i, uid in enumerate(tqdm(self.users, desc="PPC Users")):
            user_df = self.data[self.data['user_id'] == uid].sort_values('trial_number')
            real_choices = user_df['deck_num'].values
            u_trials = len(real_choices)
            
            real_adv = np.isin(real_choices, [2, 3]).astype(int)
            real_matrix[i, :u_trials] = real_adv
            real_adv_rate = np.mean(real_adv)
            real_blocks = [np.mean(real_adv[b*block_size : min((b+1)*block_size, u_trials)]) for b in range(blocks)]
            
            sim_block_matrix = np.zeros((n_sims, blocks))
            hit_rates = []
            sim_adv_rates = []
            
            for s in range(n_sims):
                idx_s = sample_indices[s]
                c_idx = idx_s // n_draws
                d_idx = idx_s % n_draws
                
                A_val = post['A'][c_idx, d_idx, i].values.item()
                alpha_val = post['alpha'][c_idx, d_idx, i].values.item()
                lam_val = post['lam'][c_idx, d_idx, i].values.item()
                c_val = post['c'][c_idx, d_idx, i].values.item()
                
                params = (A_val, alpha_val, lam_val, c_val)
                sim_c = self.simulate_subject(params, n_trials=u_trials, seed=seed+s)
                sim_adv = np.isin(sim_c, [2, 3]).astype(int)
                
                sim_matrix[s, i, :u_trials] = sim_adv
                sim_adv_rates.append(np.mean(sim_adv))
                hit_rates.append(np.mean(sim_c == real_choices))
                
                for b in range(blocks):
                    sim_block_matrix[s, b] = np.mean(sim_adv[b*block_size : min((b+1)*block_size, u_trials)])
            
            sim_blocks_mean = np.mean(sim_block_matrix, axis=0)
            # ====================================================
            # НАЧАЛО БЛОКА ОТЛАДКИ
            # ====================================================
            if np.isnan(real_blocks).any() or np.isnan(sim_blocks_mean).any():
                print(f"\n\n[!!!] НАЙДЕН NaN У ПОЛЬЗОВАТЕЛЯ: {uid}")
                print(f"Фактическое количество попыток: {u_trials}")
                print(f"Размер одного блока: {block_size}")
                print(f"Массив real_blocks: {real_blocks}")
                print(f"Массив sim_blocks_mean: {sim_blocks_mean}")
                print("====================================================\n")
            # ====================================================
            # КОНЕЦ БЛОКА ОТЛАДКИ
            # ====================================================
            ppp = np.mean(np.array(sim_adv_rates) >= real_adv_rate)
            r2 = r2_score(real_blocks, sim_blocks_mean) if blocks > 1 else np.nan
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
        sim_timecourses = np.nanmean(sim_matrix, axis=1)
        sim_mean_tc = np.nanmean(sim_timecourses, axis=0)
        hdi_tc = az.hdi(sim_timecourses, hdi_prob=0.95)
        
        plt.figure(figsize=(10, 5))
        plt.plot(real_timecourse, color='black', linewidth=2, label='Observed')
        plt.plot(sim_mean_tc, color='blue', linestyle='--', linewidth=2, label=f'PVL-{self.update_rule.capitalize()} Mean')
        plt.fill_between(range(n_trials), hdi_tc[:, 0], hdi_tc[:, 1], color='blue', alpha=0.15, label='95% HDI')
        plt.xlabel('Trial')
        plt.ylabel('P(Advantageous Choices)')
        plt.title(f'PPC Timecourse (PVL-{self.update_rule.capitalize()} Model)')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.2)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return metrics_df

    # =========================================================================
    # БАЙЕСОВСКИЙ PARAMETER RECOVERY
    # =========================================================================
    def parameter_recovery(self, n_subjects=30, n_trials=150, draws=500, tune=500):
        print(f"[*] Иерархическое Parameter Recovery (PVL-{self.update_rule.capitalize()})")
        if self.trace is None:
            raise ValueError("Сначала обучите модель на реальных данных для получения априорных гиперпараметров!")
            
        post = self.trace.posterior
        rng = np.random.RandomState(123)
        from scipy.special import expit
        
        # 1. Генерируем истинные параметры
        true_params = {
            'A': expit(rng.normal(np.median(post['mu_A']), np.median(post['sd_A']), n_subjects)),
            'alpha': expit(rng.normal(np.median(post['mu_alpha']), np.median(post['sd_alpha']), n_subjects)),
            'lam': expit(rng.normal(np.median(post['mu_lam']), np.median(post['sd_lam']), n_subjects)) * 10.0,
            'c': expit(rng.normal(np.median(post['mu_c']), np.median(post['sd_c']), n_subjects)) * 5.0
        }
        
        param_names = ['A', 'alpha', 'lam', 'c']
        true_arr = np.column_stack([true_params[p] for p in param_names])
        
        # 2. Симуляция (Генерация Dataframe в формате сырых выплат (неделенных))
        sim_data = []
        for i in range(n_subjects):
            p = (true_params['A'][i], true_params['alpha'][i], true_params['lam'][i], true_params['c'][i])
            sim_choices = self.simulate_subject(p, n_trials=n_trials, seed=100+i)
            
            for t, ch in enumerate(sim_choices):
                win = 100.0 if ch in [0, 1] else 50.0
                if ch == 0: loss = -250.0 if rng.rand() < 0.5 else 0.0
                elif ch == 1: loss = -625.0 if rng.rand() < 0.2 else 0.0 
                elif ch == 2: loss = -50.0 if rng.rand() < 0.5 else 0.0
                else: loss = -125.0 if rng.rand() < 0.2 else 0.0 
                sim_data.append({'user_id': f'sim_{i}', 'trial_number': t+1, 'deck_num': ch, 'win': win, 'loss': loss})
                
        sim_df = pd.DataFrame(sim_data)
        
        # 3. Фитинг HBA
        recovery_model = PVLModelHBA(sim_df, update_rule=self.update_rule)
        rec_trace = recovery_model.fit(draws=draws, tune=tune, chains=2, cores=2)
        rec_post = rec_trace.posterior
        
        # Извлекаем средние апостериорные
        fit_arr = np.column_stack([rec_post[p].mean(dim=["chain", "draw"]).values for p in param_names])
        
        # 4. Расчет метрик
        hdi_data = az.hdi(rec_trace, hdi_prob=0.95)
        metrics_list = []
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        for idx, p in enumerate(param_names):
            t_v = true_arr[:, idx]
            f_v = fit_arr[:, idx]
            
            # [ARVIZ FIX]: Надежное извлечение границ
            hdi_bounds = hdi_data[p].values
            hdi_lower = hdi_bounds[:, 0]
            hdi_upper = hdi_bounds[:, 1]
            
            coverage = np.mean((t_v >= hdi_lower) & (t_v <= hdi_upper))
            r_val = pearsonr(t_v, f_v)[0]
            r2_val = r2_score(t_v, f_v)
            bias = np.mean(f_v - t_v)
            rmse = np.sqrt(mean_squared_error(t_v, f_v))
            
            metrics_list.append({'Parameter': p, 'r': r_val, 'R2': r2_val, 'Bias': bias, 'RMSE': rmse, 'Coverage': coverage})
            
            ax = axes[idx]
            ax.errorbar(t_v, f_v, yerr=[f_v - hdi_lower, hdi_upper - f_v], fmt='o', alpha=0.6)
            vmin, vmax = min(t_v), max(t_v)
            ax.plot([vmin, vmax], [vmin, vmax], 'k--')
            ax.set_title(f"{p}\nr={r_val:.2f}, Cov={coverage*100:.0f}%")
            
        plt.tight_layout()
        plt.savefig(f"pvl_{self.update_rule}_hba_recovery.png", dpi=300)
        plt.close()
        
        metrics_df = pd.DataFrame(metrics_list)
        params_df = pd.DataFrame(np.hstack((true_arr, fit_arr)), 
                                 columns=[f'true_{p}' for p in param_names] + [f'fit_{p}' for p in param_names])
        params_df['user_id'] = [f"sim_{i}" for i in range(n_subjects)]
        
        return metrics_df, params_df