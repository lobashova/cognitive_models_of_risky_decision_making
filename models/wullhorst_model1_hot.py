import pymc as pm
import pytensor.tensor as pt
import arviz as az
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
from scipy.stats import mode
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class HBA_HOTModel:
    """
    Иерархическая Байесовская Модель для CCT-hot (Wüllhorst et al., 2024 - Model 1).
    Полностью оптимизировано под суперкомпьютер (HPC) и стандарты Q1/Q2.
    """
    def __init__(self, data_df, scale_factor=100.0):
        self.data = data_df.copy()
        
        # Масштабирование наград и штрафов для предотвращения Softmax Overflow
        self.scale_factor = scale_factor
        self.data['gain_scaled'] = self.data['gain_amount'] / self.scale_factor
        self.data['loss_scaled'] = self.data['loss_amount'].abs() / self.scale_factor
        
        # Индексация пользователей
        self.data['subj_idx'], self.subj_labels = pd.factorize(self.data['user_id'])
        self.n_subj = len(self.subj_labels)
        
        if 'round_id' not in self.data.columns:
            self.data['round_id'] = self.data.groupby(['user_id', 'trial_number']).ngroup()

    def build_model(self):
        subj_idx = self.data['subj_idx'].values
        gain_amt = self.data['gain_scaled'].values
        loss_amt = self.data['loss_scaled'].values
        loss_cards = self.data['loss_cards'].values
        flip_no = self.data['flip_number'].values
        choice = self.data['choice'].values

        denom = 32.0 - (flip_no - 1.0)
        p_loss_val = loss_cards / denom

        with pm.Model() as model:
            # 1. Гиперпараметры
            mu_rho = pm.Normal('mu_rho', mu=0, sigma=1.5)
            sigma_rho = pm.HalfNormal('sigma_rho', sigma=1)
            
            mu_lam = pm.Normal('mu_lam', mu=0, sigma=1.5)
            sigma_lam = pm.HalfNormal('sigma_lam', sigma=1)
            
            mu_beta = pm.Normal('mu_beta', mu=0, sigma=1.5)
            sigma_beta = pm.HalfNormal('sigma_beta', sigma=1)

            # 2. Индивидуальный уровень
            rho_raw = pm.Normal('rho_raw', mu=0, sigma=1, shape=self.n_subj)
            lam_raw = pm.Normal('lam_raw', mu=0, sigma=1, shape=self.n_subj)
            beta_raw = pm.Normal('beta_raw', mu=0, sigma=1, shape=self.n_subj)

            # 3. Трансформация в границы
            rho = pm.Deterministic('rho', 3.0 * pm.math.invlogit(mu_rho + sigma_rho * rho_raw))
            lam = pm.Deterministic('lam', 10.0 * pm.math.invlogit(mu_lam + sigma_lam * lam_raw))
            beta = pm.Deterministic('beta', 1.0 * pm.math.invlogit(mu_beta + sigma_beta * beta_raw))

            # 4. Полезность
            rho_t = rho[subj_idx]
            lam_t = lam[subj_idx]
            beta_t = beta[subj_idx]

            u_gain = gain_amt ** rho_t
            u_loss = -lam_t * (loss_amt ** rho_t) 
            
            EU = (1 - p_loss_val) * u_gain + p_loss_val * u_loss
            p_turn = pm.math.invlogit(EU / beta_t)

            # 5. Правдоподобие (Оставляем как есть, для HBA это идеально)
            pm.Bernoulli('choice_obs', p=p_turn, observed=choice)
            
        return model

    def fit(self, draws=1500, tune=1000, chains=4, target_accept=0.95):
        print(f"[*] HBA Fit: Subj={self.n_subj}, Trials={len(self.data)}")
        model = self.build_model()
        
        with model:
            idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept, 
                              progressbar=False, return_inferencedata=True)
            pm.compute_log_likelihood(idata)
            loo = az.loo(idata)
            
        return idata, loo

    # =========================================================
    # ИСПРАВЛЕНО: Строгая генеративная симуляция
    # =========================================================
    def _simulate_sequential_rounds(self, template_df, rho, lam, beta):
        """
        Истинная генеративная симуляция CCT Hot.
        Агент принимает решения шаг за шагом. Раунд завершается при выборе STOP или вытягивании LOSS.
        """
        sim_rows = []
        
        # Проходим по каждому уникальному раунду из шаблона
        for trial_num, round_df in template_df.groupby('trial_number'):
            first_row = round_df.iloc[0]
            gain_amt = first_row['gain_amount']
            loss_amt = first_row['loss_amount']
            loss_cards = first_row['loss_cards']
            uid = first_row['user_id']
            
            g_s = gain_amt / self.scale_factor
            l_s = abs(loss_amt) / self.scale_factor
            
            for flip in range(1, 33):  # CCT usually has 32 cards
                cards_left = 32.0 - (flip - 1.0)
                p_loss = loss_cards / cards_left
                
                u_gain = g_s ** rho
                u_loss = -lam * (l_s ** rho)
                EU = (1.0 - p_loss) * u_gain + p_loss * u_loss
                
                z = np.clip(EU / beta, -700, 700)
                p_turn = 1.0 / (1.0 + np.exp(-z))
                
                # Агент бросает монетку
                choice = 1 if np.random.rand() < p_turn else 0
                
                sim_rows.append({
                    'user_id': uid,
                    'trial_number': trial_num,
                    'flip_number': flip,
                    'gain_amount': gain_amt,
                    'loss_amount': loss_amt,
                    'loss_cards': loss_cards,
                    'choice': choice
                })
                
                if choice == 0:
                    break  # Агент решил остановиться -> раунд окончен
                else:
                    # Агент перевернул карту. Проверяем, не взорвался ли он
                    if np.random.rand() < p_loss:
                        break  # Вытянул Loss карту -> раунд окончен
                        
        return pd.DataFrame(sim_rows)

    def posterior_predictive_check(self, idata, n_sims=50, save_plot_path="hot_wullhorst1_ppc.png"):
        print("\n[*] Запуск True Generative Posterior Predictive Check (PPC)...")
        post = idata.posterior
        chains, draws = post.sizes['chain'], post.sizes['draw']
        
        # Эмпирическое агрегирование: количество перевернутых карт за раунд
        real_cards_df = self.data[self.data['choice'] == 1].groupby(['user_id', 'trial_number']).size().reset_index(name='cards')
        
        metrics = []
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        plot_idx = 0
        
        for uid_code, uid_label in enumerate(tqdm(self.subj_labels, desc="PPC Users")):
            # Шаблон раундов для пользователя (берем по одной записи на раунд)
            user_template = self.data[self.data['user_id'] == uid_label].drop_duplicates(subset=['trial_number'])
            trials_seq = user_template['trial_number'].values
            
            # Реальное количество вытянутых карт
            user_real = real_cards_df[real_cards_df['user_id'] == uid_label]
            y_real_cards = np.array([user_real[user_real['trial_number'] == t]['cards'].sum() for t in trials_seq])
            
            sim_matrix = np.zeros((n_sims, len(trials_seq)))
            
            for s_idx in range(n_sims):
                c = np.random.randint(0, chains)
                d = np.random.randint(0, draws)
                
                rho = float(post['rho'][c, d, uid_code])
                lam = float(post['lam'][c, d, uid_code])
                beta = float(post['beta'][c, d, uid_code])
                
                # Симулируем раунды ПРАВИЛЬНО
                sim_df = self._simulate_sequential_rounds(user_template, rho, lam, beta)
                sim_cards = sim_df[sim_df['choice'] == 1].groupby('trial_number').size()
                
                for t_idx, t_num in enumerate(trials_seq):
                    sim_matrix[s_idx, t_idx] = sim_cards.get(t_num, 0)
                    
            # ================= ДОБАВЛЕНО =================
            # Расчет метрик на основе СРЕДНЕГО КОЛИЧЕСТВА КАРТ ЗА РАУНД
            sim_mean_cards = sim_matrix.mean(axis=0)
            r2 = r2_score(y_real_cards, sim_mean_cards) if np.var(y_real_cards) > 0 else np.nan
            rmse = np.sqrt(mean_squared_error(y_real_cards, sim_mean_cards))
            mae = mean_absolute_error(y_real_cards, sim_mean_cards)
            msd = np.mean((y_real_cards - sim_mean_cards)**2)
            
            # Hit Rate (с использованием моды по симуляциям)
            mode_res = mode(sim_matrix, axis=0, keepdims=False)
            mode_sim = np.squeeze(mode_res.mode if hasattr(mode_res, 'mode') else mode_res[0])
            hit_rate = np.mean(mode_sim == y_real_cards)
            
            # Bayesian p-value
            real_avg_global = y_real_cards.mean()
            sim_avg_global = sim_matrix.mean(axis=1)
            ppp = np.mean(sim_avg_global >= real_avg_global)
            ppp_two_sided = 2 * min(ppp, 1 - ppp)
            
            metrics.append({
                'user_id': uid_label, 'R2': r2, 'RMSE': rmse, 
                'MAE': mae, 'MSD': msd, 'Hit_Rate': hit_rate,  # <-- ДОБАВЛЕНЫ МЕТРИКИ
                'ppp': ppp_two_sided, 'real_cards_avg': real_avg_global, 
                'sim_cards_avg': np.mean(sim_avg_global)
            })
            # =============================================
            
            # Отрисовка Timecourse
            if plot_idx < 4:
                window = 5
                real_rolling = pd.Series(y_real_cards).rolling(window, min_periods=1).mean()
                
                sim_rollings = np.array([pd.Series(row).rolling(window, min_periods=1).mean() for row in sim_matrix])
                hdi_lower = np.percentile(sim_rollings, 2.5, axis=0)
                hdi_upper = np.percentile(sim_rollings, 97.5, axis=0)
                sim_mean_rolling = sim_rollings.mean(axis=0)

                ax = axes[plot_idx]
                ax.plot(real_rolling.values, label='Real (MA)', color='crimson', lw=2)
                ax.plot(sim_mean_rolling, label='Simulated Mean', color='teal', lw=2)
                ax.fill_between(range(len(real_rolling)), hdi_lower, hdi_upper, color='teal', alpha=0.3, label='95% HDI')
                ax.set_title(f'PPC Timecourse: User {uid_label}')
                ax.set_xlabel('Round Sequence')
                ax.set_ylabel('Cards Turned (Moving Avg)')
                ax.legend()
                plot_idx += 1

        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=300)
        plt.close()

        return pd.DataFrame(metrics)

    def parameter_recovery(self, idata, n_subjects=50, save_plot_path="hot_wullhorst1_recovery.png"):
        print(f"\n[*] True HBA Parameter Recovery (N={n_subjects})...")
        
        post = idata.posterior
        uids = self.data['user_id'].unique()[:n_subjects]
        template = pd.concat([self.data[self.data['user_id'] == uid] for uid in uids]).copy()
        template['subj_idx'], subj_labels = pd.factorize(template['user_id'])
        
        chains, draws = post.sizes['chain'], post.sizes['draw']
        
        true_rho, true_lam, true_beta = [], [], []
        simulated_dfs = []
        
        for uid_code, uid_label in enumerate(tqdm(subj_labels, desc="Simulating Virtual Subjects")):
            c, d = np.random.randint(0, chains), np.random.randint(0, draws)
            
            rho_t = float(post['rho'][c, d, uid_code])
            lam_t = float(post['lam'][c, d, uid_code])
            beta_t = float(post['beta'][c, d, uid_code])
            
            true_rho.append(rho_t)
            true_lam.append(lam_t)
            true_beta.append(beta_t)
            
            user_template = template[template['user_id'] == uid_label].drop_duplicates(subset=['trial_number'])
            
            # Генерация нового валидного датасета
            sim_df = self._simulate_sequential_rounds(user_template, rho_t, lam_t, beta_t)
            simulated_dfs.append(sim_df)
            
        recovery_data = pd.concat(simulated_dfs)
        
        print("\n[*] Подгонка модели на симулированных данных...")
        rec_model = HBA_HOTModel(recovery_data, scale_factor=self.scale_factor)
        rec_idata, _ = rec_model.fit(draws=800, tune=800, chains=4) # Меньше draws для HPC экономии
        
        rec_post = rec_idata.posterior
        fit_rho = rec_post['rho'].median(dim=("chain", "draw")).values
        fit_lam = rec_post['lam'].median(dim=("chain", "draw")).values
        fit_beta = rec_post['beta'].median(dim=("chain", "draw")).values
        
        true_rho, true_lam, true_beta = np.array(true_rho), np.array(true_lam), np.array(true_beta)
        
        # --- Графики и метрики ---
        metrics_list = []
        params = [("rho", true_rho, fit_rho, 'rho'), 
                  ("lambda", true_lam, fit_lam, 'lam'), 
                  ("beta", true_beta, fit_beta, 'beta')]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('HBA Parameter Recovery (True vs Fit)', fontsize=16)

        for ax, (p_name, true_v, fit_v, trace_name) in zip(axes, params):
            r_val = np.corrcoef(true_v, fit_v)[0, 1]
            r2_val = r2_score(true_v, fit_v)
            rmse = np.sqrt(mean_squared_error(true_v, fit_v))
            bias = np.mean(fit_v - true_v)
            
            hdi = az.hdi(rec_idata)[trace_name]
            lower = hdi.sel(hdi='lower').values
            upper = hdi.sel(hdi='higher').values
            coverage = np.mean((true_v >= lower) & (true_v <= upper))
            
            metrics_list.append({
                'Parameter': p_name, 'Pearson_r': r_val, 'R2': r2_val, 
                'RMSE': rmse, 'Bias': bias, 'Coverage_95': coverage
            })
            
            sns.scatterplot(x=true_v, y=fit_v, ax=ax, s=60, color='indigo', alpha=0.7)
            min_v = min(true_v.min(), fit_v.min())
            max_v = max(true_v.max(), fit_v.max())
            ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='y=x')
            ax.set_title(f'{p_name} (r: {r_val:.2f}, Cov: {coverage:.2f})')
            ax.set_xlabel('True values')
            ax.set_ylabel('Recovered values (Median)')
            ax.legend()

        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=300)
        plt.close()

        df_metrics = pd.DataFrame(metrics_list)
        return recovery_data, df_metrics