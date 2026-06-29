import os
import pandas as pd
import numpy as np
import arviz as az
import matplotlib

# Отключаем интерактивный бэкенд для суперкомпьютера
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

try:
    from models.stld_model import STLDModelHBA
    from models.par4_model import Par4Model
    from models.ew_model import EWModel
    from models.ewmv_model import EWMVModel_HBA
    from models.wallsten_model import Model3_Wallsten_HBA
    from models.stl_model import STLModelHBA  
except ImportError as e:
    print(f"[!] Предупреждение при импорте модулей моделей: {e}")

def main():
    print("=" * 70)
    print("=== ЗАПУСК КОМПЛЕКСНОГО ПАКЕТНОГО ВЫЧИСЛЕНИЯ BART МОДЕЛЕЙ (HBA) ===")
    print("=" * 70)
    
    data_path = 'data/bart_data.csv'
    bart_df = pd.read_csv(data_path)
    print(f"[✓] Датасет загружен. Строк: {len(bart_df)}, Пользователей: {bart_df['user_id'].nunique()}")

    # -----------------------------------------------------------------
    # 1. STL-D Model
    # -----------------------------------------------------------------
    # print("\n" + "#" * 50)
    # print("=== ЭТАП 1: HBA STL-D ===")
    # model_stld = STLDModelHBA(bart_df, nmax=64, scale_factor=10.0)
    # trace_stld = model_stld.fit_hba(tune=1000, draws=1000)
    # az.to_netcdf(trace_stld, "bart_stld_trace.nc")
    # model_stld.get_individual_posteriors().to_csv('bart_stld_hba_params.csv', index=False)
    # with open("bart_stld_looic.txt", "w") as f: f.write(str(az.loo(trace_stld, var_name="obs")))

    # # Индивидуальный PPC
    # ppc_df_stld = model_stld.posterior_predictive_check(n_sims=100)
    # ppc_df_stld.to_csv("bart_stld_ppc_metrics.csv", index=False)

    # # ИСПРАВЛЕНО: Распаковка кортежа Parameter Recovery
    # rec_df_stld, metrics_df_stld = model_stld.parameter_recovery(n_subjects=50)
    # rec_df_stld.to_csv("bart_stld_recovery_data.csv", index=False)
    # metrics_df_stld.to_csv("bart_stld_recovery_metrics.csv", index=False)

    # # -----------------------------------------------------------------
    # # 2. Par 4 Model
    # # -----------------------------------------------------------------
    # print("\n" + "#" * 50)
    # print("=== ЭТАП 2: HBA Par4 ===")
    # par4_model = Par4Model(bart_df, scale_factor=10.0)
    # idata_par4 = par4_model.fit(draws=1000, tune=1000, chains=4, cores=4)
    # az.to_netcdf(idata_par4, "bart_par4_trace.nc")
    # az.summary(idata_par4, var_names=['phi', 'eta', 'gamma', 'tau']).to_csv("bart_par4_hba_params.csv")
    
    # with open("bart_par4_looic.txt", "w") as f: 
    #     f.write(str(az.loo(idata_par4, var_name="obs")))

    # # Индивидуальный PPC
    # ppc_df_par4 = par4_model.posterior_predictive_check(n_simulations=100, max_capacity=64)
    # ppc_df_par4.to_csv("bart_par4_ppc_metrics.csv", index=False)
    # print(f" -> [✓] Индивидуальные метрики сохранены. Mean ppp: {ppc_df_par4['ppp'].mean():.3f}")

    # # БЫЛО ПРАВИЛЬНО: Оставляем как есть, слегка причесав названия
    # recovery_df_par4, rec_metrics_summary = par4_model.parameter_recovery(n_subjects=50, n_trials=50, max_capacity=64)
    # recovery_df_par4.to_csv("bart_par4_recovery_data.csv", index=False)
    # rec_metrics_summary.to_csv("bart_par4_recovery_metrics.csv", index=False)

    # -----------------------------------------------------------------
    # 3. EW Model
    # -----------------------------------------------------------------
    # print("\n" + "#" * 50)
    # print("=== ЭТАП 3: HBA EW ===")
    # model_ew = EWModel(r=5.0)
    # # trace_ew, params_df_ew = model_ew.fit(bart_df, draws=1000, tune=1000, chains=4)
    # # az.to_netcdf(trace_ew, "bart_ew_trace.nc")
    # # params_df_ew.to_csv('bart_ew_hba_params.csv', index=False)
    # # with open("bart_ew_looic.txt", "w") as f: 
    # #     # Обязательно "log_lik", так как именно так он назван в ew_model.py
    # #     f.write(str(az.loo(trace_ew, var_name="log_lik")))

    # print("[*] Загрузка сохраненного трейса и параметров...")
    # trace_ew = az.from_netcdf("bart_ew_trace.nc")
    # params_df_ew = pd.read_csv('bart_ew_hba_params.csv')

    # # Индивидуальный PPC
    # ppc_df_ew = model_ew.posterior_predictive_check(trace_ew, bart_df, n_draws=100)
    # ppc_df_ew.to_csv('bart_ew_ppc_metrics.csv', index=False)

    # # ИСПРАВЛЕНО: Распаковка кортежа Parameter Recovery
    # res_ew = model_ew.parameter_recovery(params_df_ew, n_subjects=50, n_trials=30)
    # if res_ew is not None:
    #     rec_df_ew, metrics_df_ew = res_ew
    #     rec_df_ew.to_csv('bart_ew_recovery_data.csv', index=False)
    #     metrics_df_ew.to_csv('bart_ew_recovery_metrics.csv', index=False)

    # -----------------------------------------------------------------
    # 4. EWMV Model
    # -----------------------------------------------------------------
    # print("\n" + "#" * 50)
    # print("=== ЭТАП 4: HBA EWMV ===")
    # model_ewmv = EWMVModel_HBA(r=1.0, max_pumps=64)
    # idata_ewmv = model_ewmv.fit_hba(bart_df, draws=1000, tune=1000, chains=4, cores=4)
    # az.to_netcdf(idata_ewmv, "bart_ewmv_trace.nc")
    # with open("bart_ewmv_looic.txt", "w") as f: 
    #     f.write(str(az.loo(idata_ewmv, var_name="log_lik")))

    # # Выгрузка параметров
    # user_ids = model_ewmv.user_ids
    # pd.DataFrame([{
    #     'user_id': uid,
    #     'psi': idata_ewmv.posterior['psi'][:, :, i].mean().item(),
    #     'xi': idata_ewmv.posterior['xi'][:, :, i].mean().item(),
    #     'rho': idata_ewmv.posterior['rho'][:, :, i].mean().item(),
    #     'tau': idata_ewmv.posterior['tau'][:, :, i].mean().item(),
    #     'lam': idata_ewmv.posterior['lam'][:, :, i].mean().item(),
    # } for i, uid in enumerate(user_ids)]).to_csv('bart_ewmv_hba_params.csv', index=False)

    # # Индивидуальный PPC
    # ppc_df_ewmv = EWMVModel_HBA.run_ppc(model_ewmv, bart_df, n_sims=100)
    # ppc_df_ewmv.to_csv('bart_ewmv_hba_ppc.csv', index=False)
    # idata_ewmv = az.from_netcdf("bart_ewmv_trace.nc")
    # # ИСПРАВЛЕНО: Распаковка кортежа Parameter Recovery
    # rec_df_ewmv, metrics_df_ewmv = EWMVModel_HBA.run_parameter_recovery(idata_ewmv, n_subjects=40, n_trials=50, max_pumps=64)
    # rec_df_ewmv.to_csv('bart_ewmv_recovery_data.csv', index=False)
    # metrics_df_ewmv.to_csv('bart_ewmv_recovery_metrics.csv', index=False)

    # -----------------------------------------------------------------
    # 5. Wallsten Model 3
    # -----------------------------------------------------------------
    # print("\n" + "#" * 50)
    # print("=== ЭТАП 5: HBA Wallsten ===")
    # model_wallsten = Model3_Wallsten_HBA(max_pumps=64)
    # idata_w = model_wallsten.fit_hba(bart_df, draws=1000, tune=1000, chains=4, cores=4)
    # az.to_netcdf(idata_w, "bart_wallsten_trace.nc")
    # with open("bart_wallsten_looic.txt", "w") as f: 
    #     f.write(str(az.loo(idata_w, var_name="log_lik")))

    # post_w = idata_w.posterior
    # q_m, m_m = post_w['q1'].mean(dim=['chain','draw']).values, post_w['m0'].mean(dim=['chain','draw']).values
    # pd.DataFrame({
    #     'user_id': model_wallsten.users.cat.categories,
    #     'bart_wallsten_gamma_plus': post_w['gamma_plus'].mean(dim=['chain','draw']).values,
    #     'bart_wallsten_beta': post_w['beta'].mean(dim=['chain','draw']).values,
    #     'bart_wallsten_q1': q_m,
    #     'bart_wallsten_a0': q_m * m_m,
    #     'bart_wallsten_m0': m_m
    # }).to_csv('bart_wallsten_hba_params.csv', index=False)

    # # Индивидуальный PPC
    # ppc_df_wallsten = model_wallsten.predictive_check(bart_df, n_sims=50, seed=42)
    # ppc_df_wallsten.to_csv('bart_wallsten_ppc_metrics.csv', index=False)

    # # ИСПРАВЛЕНО: Распаковка кортежа Parameter Recovery
    # rec_df_w, metrics_df_w = model_wallsten.parameter_recovery(template_df=bart_df, n_subjects=40, n_trials=60)
    # rec_df_w.to_csv('bart_wallsten_recovery_data.csv', index=False)
    # metrics_df_w.to_csv('bart_wallsten_recovery_metrics.csv', index=False)


    # -----------------------------------------------------------------
    # 6. STL Model
    # -----------------------------------------------------------------
    print("\n" + "#" * 50)
    print("=== ЭТАП 6: HBA STL ===")
    model_stl_hba = STLModelHBA(bart_df, max_pumps=64)
    # trace_stl = model_stl_hba.fit(draws=1000, tune=1000, chains=2)
    # az.to_netcdf(trace_stl, "bart_stl_trace.nc")
    # with open("bart_stl_looic.txt", "w") as f: 
    #     f.write(str(az.loo(trace_stl, var_name="log_lik")))

    trace_stl = az.from_netcdf("bart_stl_trace.nc")
    model_stl_hba.trace = trace_stl
    # az.summary(trace_stl, var_names=['w1_s', 'vwin', 'vloss', 'beta']).to_csv('bart_stl_hba_params.csv')

    # # Индивидуальный PPC
    # ppc_df_stl = model_stl_hba.predictive_check(n_sim=100)
    # ppc_df_stl.to_csv("bart_stl_ppc_hba.csv", index=False)

    # БЫЛО ПРАВИЛЬНО: Оставляем, слегка причесав названия сохранения
    res_stl = model_stl_hba.parameter_recovery(n_subjects=50, draws=1000, tune=1000)
    if res_stl is not None:
        res_stl[0].to_csv("bart_stl_recovery_data.csv", index=False)
        res_stl[1].to_csv("bart_stl_recovery_metrics.csv", index=False)

    print("\n" + "=" * 70)
    print("=== РАСЧЕТЫ ДЛЯ ВСЕХ МОДЕЛЕЙ BART ЗАВЕРШЕНЫ УСПЕШНО ===")
    print("=" * 70)

if __name__ == '__main__':
    main()