from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd
import joblib
import uvicorn

app = FastAPI(title="Kepler KOI Prediction API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoders
model = joblib.load('AI/exoplanet_model.pkl')
label_encoder = joblib.load('AI/label_encoder.pkl')
feature_columns = joblib.load('AI/feature_columns.pkl')
feature_encoders = joblib.load('AI/feature_encoders.pkl')
imputer_numeric = joblib.load('AI/imputer_numeric.pkl')
imputer_cat = joblib.load('AI/imputer_cat.pkl')
numeric_cols = joblib.load('AI/numeric_columns.pkl')
cat_cols = joblib.load('AI/cat_columns.pkl')

# Pydantic model for request validation
class KOIData(BaseModel):
    koi_vet_stat: Optional[str] = None
    koi_vet_date: Optional[str] = None
    koi_pdisposition: Optional[str] = None
    koi_fpflag_nt: Optional[int] = None
    koi_fpflag_ss: Optional[int] = None
    koi_fpflag_co: Optional[int] = None
    koi_fpflag_ec: Optional[int] = None
    koi_disp_prov: Optional[str] = None
    koi_comment: Optional[str] = None
    koi_period: Optional[float] = None
    koi_period_err1: Optional[float] = None
    koi_period_err2: Optional[float] = None
    koi_time0bk: Optional[float] = None
    koi_time0bk_err1: Optional[float] = None
    koi_time0bk_err2: Optional[float] = None
    koi_time0: Optional[float] = None
    koi_time0_err1: Optional[float] = None
    koi_time0_err2: Optional[float] = None
    koi_eccen: Optional[float] = None
    koi_impact: Optional[float] = None
    koi_impact_err1: Optional[float] = None
    koi_impact_err2: Optional[float] = None
    koi_duration: Optional[float] = None
    koi_duration_err1: Optional[float] = None
    koi_duration_err2: Optional[float] = None
    koi_depth: Optional[float] = None
    koi_depth_err1: Optional[float] = None
    koi_depth_err2: Optional[float] = None
    koi_ror: Optional[float] = None
    koi_ror_err1: Optional[float] = None
    koi_ror_err2: Optional[float] = None
    koi_srho: Optional[float] = None
    koi_srho_err1: Optional[float] = None
    koi_srho_err2: Optional[float] = None
    koi_fittype: Optional[str] = None
    koi_prad: Optional[float] = None
    koi_prad_err1: Optional[float] = None
    koi_prad_err2: Optional[float] = None
    koi_sma: Optional[float] = None
    koi_incl: Optional[float] = None
    koi_teq: Optional[float] = None
    koi_insol: Optional[float] = None
    koi_insol_err1: Optional[float] = None
    koi_insol_err2: Optional[float] = None
    koi_dor: Optional[float] = None
    koi_dor_err1: Optional[float] = None
    koi_dor_err2: Optional[float] = None
    koi_limbdark_mod: Optional[str] = None
    koi_ldm_coeff4: Optional[float] = None
    koi_ldm_coeff3: Optional[float] = None
    koi_ldm_coeff2: Optional[float] = None
    koi_ldm_coeff1: Optional[float] = None
    koi_parm_prov: Optional[str] = None
    koi_max_sngle_ev: Optional[float] = None
    koi_max_mult_ev: Optional[float] = None
    koi_model_snr: Optional[float] = None
    koi_count: Optional[int] = None
    koi_num_transits: Optional[int] = None
    koi_tce_plnt_num: Optional[int] = None
    koi_tce_delivname: Optional[str] = None
    koi_quarters: Optional[str] = None
    koi_bin_oedp_sig: Optional[float] = None
    koi_trans_mod: Optional[str] = None
    koi_datalink_dvr: Optional[str] = None
    koi_datalink_dvs: Optional[str] = None
    koi_steff: Optional[float] = None
    koi_steff_err1: Optional[float] = None
    koi_steff_err2: Optional[float] = None
    koi_slogg: Optional[float] = None
    koi_slogg_err1: Optional[float] = None
    koi_slogg_err2: Optional[float] = None
    koi_smet: Optional[float] = None
    koi_smet_err1: Optional[float] = None
    koi_smet_err2: Optional[float] = None
    koi_srad: Optional[float] = None
    koi_srad_err1: Optional[float] = None
    koi_srad_err2: Optional[float] = None
    koi_smass: Optional[float] = None
    koi_smass_err1: Optional[float] = None
    koi_smass_err2: Optional[float] = None
    koi_sparprov: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    koi_kepmag: Optional[float] = None
    koi_gmag: Optional[float] = None
    koi_rmag: Optional[float] = None
    koi_imag: Optional[float] = None
    koi_zmag: Optional[float] = None
    koi_jmag: Optional[float] = None
    koi_hmag: Optional[float] = None
    koi_kmag: Optional[float] = None
    koi_fwm_stat_sig: Optional[float] = None
    koi_fwm_sra: Optional[float] = None
    koi_fwm_sra_err: Optional[float] = None
    koi_fwm_sdec: Optional[float] = None
    koi_fwm_sdec_err: Optional[float] = None
    koi_fwm_srao: Optional[float] = None
    koi_fwm_srao_err: Optional[float] = None
    koi_fwm_sdeco: Optional[float] = None
    koi_fwm_sdeco_err: Optional[float] = None
    koi_fwm_prao: Optional[float] = None
    koi_fwm_prao_err: Optional[float] = None
    koi_fwm_pdeco: Optional[float] = None
    koi_fwm_pdeco_err: Optional[float] = None
    koi_dicco_mra: Optional[float] = None
    koi_dicco_mra_err: Optional[float] = None
    koi_dicco_mdec: Optional[float] = None
    koi_dicco_mdec_err: Optional[float] = None
    koi_dicco_msky: Optional[float] = None
    koi_dicco_msky_err: Optional[float] = None
    koi_dikco_mra: Optional[float] = None
    koi_dikco_mra_err: Optional[float] = None
    koi_dikco_mdec: Optional[float] = None
    koi_dikco_mdec_err: Optional[float] = None
    koi_dikco_msky: Optional[float] = None
    koi_dikco_msky_err: Optional[float] = None

    class Config:
        extra = "allow"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(data: KOIData):
    try:
        # Convert to dictionary and then DataFrame
        data_dict = data.dict()
        df_new = pd.DataFrame([data_dict])
        
        # Impute numeric columns
        numeric_cols_exist = [c for c in numeric_cols if c in df_new.columns]
        if numeric_cols_exist:
            df_new[numeric_cols_exist] = imputer_numeric.transform(df_new[numeric_cols_exist])
        
        # Impute categorical columns
        cat_cols_exist = [c for c in cat_cols if c in df_new.columns]
        if cat_cols_exist:
            df_new[cat_cols_exist] = imputer_cat.transform(df_new[cat_cols_exist])
        
        # Encode categorical features
        for col, encoder in feature_encoders.items():
            if col in df_new.columns:
                df_new[col] = df_new[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
                df_new[col] = encoder.transform(df_new[col])
        
        # Ensure correct column order
        df_new = df_new[feature_columns]
        
        # Make prediction
        prediction = model.predict(df_new)
        proba = model.predict_proba(df_new)
        
        # Prepare response
        result = {
            'prediction': label_encoder.inverse_transform(prediction)[0],
            'probabilities': {
                cls: float(prob) for cls, prob in zip(label_encoder.classes_, proba[0])
            }
        }
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)