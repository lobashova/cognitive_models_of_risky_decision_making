# model_igt.py

import os
import matplotlib
# Отключаем интерактивный бэкенд для работы на сервере ДО импорта pyplot
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from joblib import Parallel, delayed
import arviz as az

# Импорты ваших модулей (убедитесь, что папка models лежит рядом со скриптом)
from models.vse_model import VSEModelHBA
from models.ev_model import EVModelHBA
from models.pvl_model import PVLModelHBA
from models.vpp_model import VPPModelHBA
from models.orl_model import ORLModel

# --- 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def compute_ic(nll, n_params, n_obs):
    aic = 2 * n_params + 2 * nll
    bic = n_params * np.log(n_obs) + 2 * nll
    return aic, bic

# --- 2. ГЛАВНЫЙ ИСПОЛНЯЕМЫЙ БЛОК ---

if __name__ == '__main__':
    print("=== ЗАПУСК ПАЙПЛАЙНА IGT МОДЕЛЕЙ ===")
    
    # Загрузите ваш реальный датасет
    igt_df = pd.read_csv('data/igt_data.csv')
    valid_users = [uid for uid, df in igt_df.groupby('user_id') if len(df) == 150]
    clean_igt_df = igt_df[igt_df['user_id'].isin(valid_users)].reset_index(drop=True)
    # =========================================================
    # МОДЕЛЬ 1: VSE
    # =========================================================
    print("\n--- Модель 1: VSE ---")
    vse = VSEModelHBA(igt_df)
    
    print("1. Запуск HBA Fitting...")
    # trace_vse = vse.fit_hba(draws=1500, tune=1000, chains=4, cores=4)
    
    # # [SAVE TRACE]: Сохраняем полный апостериорный след модели VSE
    # print("[*] Сохранение trace для VSE...")
    # az.to_netcdf(trace_vse, 'igt_vse_trace.nc')
    
    # # Сохранение апостериорных параметров
    # summary = az.summary(trace_vse, var_names=['theta', 'delta', 'alpha', 'phi', 'c'])
    # summary.to_csv('igt_vse_hba_params.csv')
    
    # print("2. Запуск PPC...")
    # vse_ppc_df = vse.posterior_predictive_check(n_sims=100)
    # vse_ppc_df.to_csv('igt_vse_ppc_metrics.csv', index=False)
    
    # vse.trace = az.from_netcdf("igt_vse_trace.nc")

    # print("3. Запуск Parameter Recovery...")
    # rec_metrics_df, rec_params_df = vse.parameter_recovery(n_subjects=50) 
    # rec_metrics_df.to_csv('igt_vse_recovery_metrics.csv', index=False)
    # rec_params_df.to_csv('igt_vse_recovery_params.csv', index=False)

    # # =========================================================
    # # МОДЕЛЬ 2: EV
    # # =========================================================
    # print("\n--- Модель 2: EV ---")
    # model_ev = EVModelHBA(igt_df)
    # trace_ev = model_ev.fit(draws=1000, tune=1000, chains=4, cores=4)
    
    # # [SAVE TRACE]: Сохраняем полный апостериорный след модели EV
    # print("[*] Сохранение trace для EV...")
    # az.to_netcdf(trace_ev, 'igt_ev_trace.nc')
    
    # # Сохранение параметров
    # summary = az.summary(trace_ev, var_names=['w', 'a', 'c'])
    # summary.to_csv('igt_ev_hba_params.csv')
    
    # # Расчет и сохранение LOOIC
    # loo = az.loo(trace_ev, var_name="obs")
    # with open("igt_ev_looic.txt", "w") as f:
    #     f.write(str(loo))

    # # 2. PPC
    # print("Запуск PPC...")
    # ev_ppc_df = model_ev.posterior_predictive_check(trace_ev, block_size=30, n_sims=100)
    # ev_ppc_df.to_csv('igt_ev_ppc_metrics.csv', index=False)
    
    # # 3. Parameter Recovery
    # print("Запуск Parameter Recovery...")
    # rec_metrics_df, rec_params_df = model_ev.parameter_recovery(trace_ev, n_subjects=15, n_trials=150)
    # rec_metrics_df.to_csv('igt_ev_recovery_metrics.csv', index=False)
    # rec_params_df.to_csv('igt_ev_recovery_params.csv', index=False)

    # =========================================================
    # МОДЕЛЬ 3 & 4: PVL-Decay & PVL-Delta
    # =========================================================
    # rules = ["delta", "decay"]

    # for rule in rules:
    #     print(f"\n=========================================================")
    #     print(f" ЗАПУСК МОДЕЛИ PVL-{rule.upper()}")
    #     print(f"=========================================================")
        
    #     hba_model_pvl = PVLModelHBA(clean_igt_df, update_rule=rule)

    #     trace_pvl = hba_model_pvl.fit(draws=1000, tune=1000, chains=4, cores=4)

    #     # [SAVE TRACE]: Сохраняем полный апостериорный след для PVL-Delta и PVL-Decay соответственно
    #     print(f"[*] Сохранение trace для PVL-{rule.upper()}...")
    #     az.to_netcdf(trace_pvl, f'igt_pvl_{rule}_trace.nc')

    #     # 2. Сохранение параметров
    #     summary_df = az.summary(trace_pvl, var_names=['A', 'alpha', 'lam', 'c'])
    #     summary_df.to_csv(f'igt_pvl_{rule}_params.csv')
            
    #     # Расчет и сохранение LOOIC
    #     loo = az.loo(trace_pvl, var_name="obs")
    #     with open(f"igt_pvl_{rule}_looic.txt", "w") as f:
    #         f.write(str(loo))

    #     # 3. PPC
    #     print(f"Запуск PPC (PVL-{rule.upper()})...")
    #     ppc_metrics_df = hba_model_pvl.posterior_predictive_check(n_sims=100, blocks=5)
    #     ppc_metrics_df.to_csv(f'igt_pvl_{rule}_ppc_metrics.csv', index=False)

    #     # 4. Parameter Recovery
    #     print(f"Запуск Parameter Recovery (PVL-{rule.upper()})...")
    #     rec_metrics_df, rec_params_df = hba_model_pvl.parameter_recovery(n_subjects=30, n_trials=150)
    #     rec_metrics_df.to_csv(f'igt_pvl_{rule}_recovery_metrics.csv', index=False)
    #     rec_params_df.to_csv(f'igt_pvl_{rule}_recovery_params.csv', index=False)

    # # =========================================================
    # # МОДЕЛЬ 5: VPP
    # # =========================================================
    # print("\n--- Модель 5: VPP ---")
    # model_vpp = VPPModelHBA(clean_igt_df)
    # trace_vpp = model_vpp.fit(draws=1500, tune=1000, chains=4, cores=4)

    # # [SAVE TRACE]: Сохраняем полный след MCMC для модели VPP
    # print("[*] Сохранение trace для VPP...")
    # az.to_netcdf(trace_vpp, 'igt_vpp_trace.nc')

    # # Сохранение апостериорных параметров
    # param_names = ['phi', 'alpha', 'lam', 'c', 'w', 'K', 'eps_pos', 'eps_neg']
    # summary = az.summary(trace_vpp, var_names=param_names)
    # summary.to_csv('igt_vpp_hba_params.csv')
    
    # # Расчет и сохранение LOOIC
    # loo = az.loo(trace_vpp, var_name="obs")
    # with open("igt_vpp_looic.txt", "w") as f:
    #     f.write(str(loo))

    # # 2. Posterior Predictive Check
    # print("Запуск PPC...")
    # vpp_ppc_metrics = model_vpp.predictive_check(n_sims=100, seed=42)
    # vpp_ppc_metrics.to_csv('igt_vpp_ppc_metrics.csv', index=False)

    # # 3. Parameter Recovery
    # print("Запуск Parameter Recovery...")
    # vpp_rec_metrics, vpp_rec_params = model_vpp.parameter_recovery(n_subjects=40, n_trials=150)
    # vpp_rec_metrics.to_csv('igt_vpp_recovery_metrics.csv', index=False)
    # vpp_rec_params.to_csv('igt_vpp_recovery_params.csv', index=False)

    # =========================================================
    # МОДЕЛЬ 6: ORL
    # =========================================================
    print("\n--- Модель 6: ORL ---")
    data_by_subject = {uid: df.sort_values('trial_number') for uid, df in clean_igt_df.groupby('user_id') if len(df) >= 20}
    orl_model = ORLModel(data_by_subject, scale_factor=100.0) 
    
    print("Фиттинг HBA...")
    # trace_orl = orl_model.fit_hierarchical(draws=1000, tune=1000, chains=4, cores=4)
    
    # # [SAVE TRACE]: Сохраняем полный след MCMC для модели ORL
    # print("[*] Сохранение trace для ORL...")
    # az.to_netcdf(trace_orl, 'igt_orl_trace.nc')
    
    # orl_model.get_parameters_df().to_csv('igt_orl_params.csv', index=False)
    trace_orl = az.from_netcdf("igt_orl_trace.nc")
    orl_model.trace = trace_orl
    # Расчет и сохранение LOOIC
    loo = az.loo(trace_orl, var_name="obs")
    with open("igt_orl_looic.txt", "w") as f:
        f.write(str(loo))

    # 2. PPC
    print("Posterior Predictive Check (блоками по 30)...")
    orl_ppc_df = orl_model.posterior_predictive_check(n_sims=100, block_size=30)
    orl_ppc_df.to_csv('igt_orl_ppc.csv', index=False)

    # 3. Parameter Recovery
    print("Parameter Recovery...")
    metrics_pr_df, params_pr_df = orl_model.parameter_recovery(n_subjects=30, n_trials=150)
    metrics_pr_df.to_csv('igt_orl_pr_metrics.csv', index=False)
    params_pr_df.to_csv('igt_orl_pr_params.csv', index=False)
    
    print("\n[УСПЕХ] Все этапы завершены. Данные стандартизированы, графики, цепи MCMC и метрики сохранены на диск.")