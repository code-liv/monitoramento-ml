from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
from functools import lru_cache
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score

router = APIRouter()

# ====== CAMINHOS FIXOS (SEUS) ======
ROOT = Path(r"C:\Users\liliz\Downloads\Desafio Neurotech\monitoramento-ml")
MONITORING = ROOT / "monitoring"
MODEL_PATH = MONITORING / "model.pkl"
# ===================================

class RecordsPayload(BaseModel):
    records: List[Dict[str, Any]]

def to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df = df.replace({None: np.nan})
    if "REF_DATE" in df.columns:
        df["REF_DATE"] = pd.to_datetime(df["REF_DATE"], errors="coerce")
    return df

@lru_cache()
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def score_with_model(df: pd.DataFrame) -> np.ndarray:
    model = load_model()
    X = df.drop(columns=[c for c in ["TARGET", "REF_DATE"] if c in df.columns], errors="ignore")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        preds = model.predict(X)
        probs = np.asarray(preds, dtype=float)
    return probs

@router.post("/performance")
def performance(payload: RecordsPayload):
    """
    Recebe: { "records": [ { ... }, { ... } ] }
    Retorna:
      - volumetria por mês (REF_DATE)
      - AUC-ROC usando TARGET
    """
    df = to_dataframe(payload.records)

    # (a) volumetria
    if "REF_DATE" not in df.columns:
        raise HTTPException(status_code=400, detail="Coluna REF_DATE não encontrada nos registros.")
    vol = (
        df.assign(month=df["REF_DATE"].dt.to_period("M").astype(str))
          .groupby("month").size().to_dict()
    )

    # (b) AUC-ROC
    if "TARGET" not in df.columns:
        raise HTTPException(status_code=400, detail="Coluna TARGET não encontrada nos registros.")
    y_true = df["TARGET"].astype(int)
    try:
        y_score = score_with_model(df)
        auc = float(roc_auc_score(y_true, y_score))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao calcular AUC: {e}")

    return {"volumetry_by_month": vol, "roc_auc": round(auc, 6), "n_records": int(len(df))}
