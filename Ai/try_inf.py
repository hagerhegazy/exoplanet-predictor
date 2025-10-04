import pandas as pd
import joblib
import numpy as np

model = joblib.load('exoplanet_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
feature_columns = joblib.load('feature_columns.pkl')
feature_encoders = joblib.load('feature_encoders.pkl')
imputer_numeric = joblib.load('imputer_numeric.pkl')
imputer_cat = joblib.load('imputer_cat.pkl')
numeric_cols = joblib.load('numeric_columns.pkl')
cat_cols = joblib.load('cat_columns.pkl')

sample_data = {
    'koi_vet_stat': ['done'],
    'koi_vet_date': ['2015-06-15'],
    'koi_pdisposition': ['CANDIDATE'],
    'koi_fpflag_nt': [0],
    'koi_fpflag_ss': [0],
    'koi_fpflag_co': [0],
    'koi_fpflag_ec': [0],
    'koi_disp_prov': ['Kepler'],
    'koi_comment': ['none'],
    'koi_period': [10.5],
    'koi_period_err1': [0.001],
    'koi_period_err2': [-0.001],
    'koi_time0bk': [135.5],
    'koi_time0bk_err1': [0.01],
    'koi_time0bk_err2': [-0.01],
    'koi_time0': [135.5],
    'koi_time0_err1': [0.01],
    'koi_time0_err2': [-0.01],
    'koi_eccen': [0.0],
    'koi_impact': [0.5],
    'koi_impact_err1': [0.1],
    'koi_impact_err2': [-0.1],
    'koi_duration': [3.2],
    'koi_duration_err1': [0.1],
    'koi_duration_err2': [-0.1],
    'koi_depth': [500],
    'koi_depth_err1': [50],
    'koi_depth_err2': [-50],
    'koi_ror': [0.05],
    'koi_ror_err1': [0.005],
    'koi_ror_err2': [-0.005],
    'koi_srho': [1.5],
    'koi_srho_err1': [0.1],
    'koi_srho_err2': [-0.1],
    'koi_fittype': ['LS'],
    'koi_prad': [2.5],
    'koi_prad_err1': [0.3],
    'koi_prad_err2': [-0.3],
    'koi_sma': [0.1],
    'koi_incl': [89.5],
    'koi_teq': [300],
    'koi_insol': [1.2],
    'koi_insol_err1': [0.1],
    'koi_insol_err2': [-0.1],
    'koi_dor': [15.0],
    'koi_dor_err1': [1.0],
    'koi_dor_err2': [-1.0],
    'koi_limbdark_mod': ['nonlinear'],
    'koi_ldm_coeff4': [0.1],
    'koi_ldm_coeff3': [0.2],
    'koi_ldm_coeff2': [0.3],
    'koi_ldm_coeff1': [0.4],
    'koi_parm_prov': ['Kepler'],
    'koi_max_sngle_ev': [100],
    'koi_max_mult_ev': [200],
    'koi_model_snr': [25],
    'koi_count': [1],
    'koi_num_transits': [15],
    'koi_tce_plnt_num': [1],
    'koi_tce_delivname': ['q1_q17_dr25_tce'],
    'koi_quarters': [17],  # Should be numeric (integer)
    'koi_bin_oedp_sig': [5.0],
    'koi_trans_mod': ['mandel-agol'],
    'koi_datalink_dvr': ['link1'],
    'koi_datalink_dvs': ['link2'],
    'koi_steff': [5500],
    'koi_steff_err1': [100],
    'koi_steff_err2': [-100],
    'koi_slogg': [4.5],
    'koi_slogg_err1': [0.1],
    'koi_slogg_err2': [-0.1],
    'koi_smet': [0.0],
    'koi_smet_err1': [0.1],
    'koi_smet_err2': [-0.1],
    'koi_srad': [1.0],
    'koi_srad_err1': [0.05],
    'koi_srad_err2': [-0.05],
    'koi_smass': [1.0],
    'koi_smass_err1': [0.05],
    'koi_smass_err2': [-0.05],
    'koi_sparprov': ['Kepler'],
    'ra': [290.5],
    'dec': [45.2],
    'koi_kepmag': [14.5],
    'koi_gmag': [15.0],
    'koi_rmag': [14.2],
    'koi_imag': [13.8],
    'koi_zmag': [13.5],
    'koi_jmag': [13.0],
    'koi_hmag': [12.8],
    'koi_kmag': [12.5],
    'koi_fwm_stat_sig': [10.0],
    'koi_fwm_sra': [0.1],
    'koi_fwm_sra_err': [0.01],
    'koi_fwm_sdec': [0.1],
    'koi_fwm_sdec_err': [0.01],
    'koi_fwm_srao': [0.05],
    'koi_fwm_srao_err': [0.01],
    'koi_fwm_sdeco': [0.05],
    'koi_fwm_sdeco_err': [0.01],
    'koi_fwm_prao': [0.02],
    'koi_fwm_prao_err': [0.01],
    'koi_fwm_pdeco': [0.02],
    'koi_fwm_pdeco_err': [0.01],
    'koi_dicco_mra': [0.1],
    'koi_dicco_mra_err': [0.01],
    'koi_dicco_mdec': [0.1],
    'koi_dicco_mdec_err': [0.01],
    'koi_dicco_msky': [0.15],
    'koi_dicco_msky_err': [0.01],
    'koi_dikco_mra': [0.1],
    'koi_dikco_mra_err': [0.01],
    'koi_dikco_mdec': [0.1],
    'koi_dikco_mdec_err': [0.01],
    'koi_dikco_msky': [0.15],
    'koi_dikco_msky_err': [0.01]}

df_new = pd.DataFrame(sample_data)
# print(df_new.shape)

numeric_cols_exist = [c for c in numeric_cols if c in df_new.columns]
if numeric_cols_exist:
    df_new[numeric_cols_exist] = imputer_numeric.transform(df_new[numeric_cols_exist])

cat_cols_exist = [c for c in cat_cols if c in df_new.columns]
if cat_cols_exist:
    df_new[cat_cols_exist] = imputer_cat.transform(df_new[cat_cols_exist])

#Encode categorical features
for col, encoder in feature_encoders.items():
    if col in df_new.columns:
        df_new[col] = df_new[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
        df_new[col] = encoder.transform(df_new[col])
df_new = df_new[feature_columns]

prediction = model.predict(df_new)
proba = model.predict_proba(df_new)
print(f"Prediction: {label_encoder.inverse_transform(prediction)[0]}")
print(f"\nProbabilities:")
for cls, prob in zip(label_encoder.classes_, proba[0]):
    print(f"  {cls}: {prob:.4f}")