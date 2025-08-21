# monitoring/app/api/endpoints/aderencia.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp
from pandas.api.types import is_numeric_dtype

router = APIRouter(tags=["adherence"])
IMPL_VERSION = "adherence-v9"  # marcador de versão

# ---------------------------------
# Caminhos (layout do seu projeto)
# __file__ = ...\monitoramento-ml\monitoring\app\api\endpoints\aderencia.py
# parents[4] = ...\monitoramento-ml
# ---------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parents[4]
MONITORING_DIR = PROJECT_ROOT / "monitoring"
MODEL_PATH     = MONITORING_DIR / "model.pkl"
REF_DATASET    = PROJECT_ROOT / "challenge-data-scientist" / "datasets" / "credit_01" / "test.gz"


class AdherencePayload(BaseModel):
    dataset_path: str  # caminho completo para train.gz ou oot.gz


def read_any_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Caminho não existe: {path}")
    try:
        return pd.read_csv(path, compression="gzip")
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Não consegui ler o arquivo '{path}'. Erro: {e}")


@lru_cache()
def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=400, detail=f"Modelo não encontrado: {MODEL_PATH}")
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao carregar modelo: {e}")


def drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
    # remove colunas que não devem ir pro modelo
    for c in ("TARGET", "REF_DATE", "ref_date", "target"):
        if c in df.columns:
            df = df.drop(columns=[c])
    return df


def get_expected_cols(model, ref_df: pd.DataFrame) -> list:
    """
    Conjunto de colunas de entrada:
      - se o modelo declara feature_names_in_, usamos essa ordem
      - caso contrário, usamos as colunas da referência (sem TARGET/REF_DATE)
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(drop_non_features(ref_df).columns)


def split_num_cat(expected_cols: list, ref_df: pd.DataFrame):
    """Separa colunas esperadas em numéricas vs categóricas com base na referência."""
    ref_df = drop_non_features(ref_df)
    num_cols = [c for c in expected_cols if c in ref_df.columns and is_numeric_dtype(ref_df[c])]
    cat_cols = [c for c in expected_cols if c not in num_cols]
    return num_cols, cat_cols


def build_aligned_df(df: pd.DataFrame, expected_cols: list, ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Monta DataFrame com TODAS as expected_cols (na ordem):
      - numéricas: to_numeric(coerce) + mediana da referência + 0.0 no que sobrar
      - categóricas: manter dtype 'object'; preencher ausências e categorias
                     NÃO vistas na referência com a MODA da referência
    """
    df = drop_non_features(df).copy()
    ref_df = drop_non_features(ref_df).copy()

    num_cols, cat_cols = split_num_cat(expected_cols, ref_df)

    out = pd.DataFrame(index=df.index)

    # ---- CATEGÓRICAS: usar SOMENTE categorias conhecidas na referência
    for c in cat_cols:
        # categorias válidas e moda da referência
        if c in ref_df.columns:
            ref_col = ref_df[c].astype("object")
            valid = set(ref_col.dropna().astype(str).unique().tolist())
            # moda (se não houver, tenta pegar um valor qualquer da referência)
            moda_series = ref_col.mode(dropna=True)
            if not moda_series.empty:
                moda = str(moda_series.iloc[0])
            else:
                vals = list(valid)
                moda = str(vals[0]) if vals else ""  # fallback extremo
        else:
            valid = set()
            moda = ""  # sem referência para essa coluna

        # coluna da base de entrada
        if c in df.columns:
            s = df[c].astype("object")
        else:
            s = pd.Series(np.nan, index=df.index, dtype="object")

        # preenche ausentes com moda
        s = s.where(~s.isna(), other=moda)

        # troca categorias desconhecidas pela moda (somente se tivermos conjunto válido)
        if valid:
            s_str = s.astype(str)
            mask_bad = ~s_str.isin(valid)
            if mask_bad.any():
                s = s_str.where(~mask_bad, other=moda)
            else:
                s = s_str  # já está tudo válido
        else:
            # se não temos conjunto válido da referência, mantemos como string
            s = s.astype(str)

        out[c] = s

    # ---- NUMÉRICAS: coerção e imputação via mediana da referência
    med = {}
    for c in num_cols:
        if c in ref_df.columns:
            med[c] = pd.to_numeric(ref_df[c], errors="coerce").median()
        else:
            med[c] = 0.0

    for c in num_cols:
        if c in df.columns:
            s_num = pd.to_numeric(df[c], errors="coerce")
        else:
            s_num = pd.Series(np.nan, index=df.index, dtype="float64")
        s_num = s_num.fillna(med[c]).fillna(0.0).astype(np.float32)
        out[c] = s_num

    # Ordena na ordem esperada
    out = out[expected_cols]

    return out


def score_df(df: pd.DataFrame, model) -> np.ndarray:
    # 1) probabilidade
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(df)[:, 1]
        except Exception as e:
            last_err = e
    else:
        last_err = None

    # 2) decision_function (se existir)
    if hasattr(model, "decision_function"):
        try:
            dec = model.decision_function(df)
            dec = np.asarray(dec, dtype=np.float32)
            mn, mx = float(np.min(dec)), float(np.max(dec))
            return (dec - mn) / (mx - mn + 1e-12)
        except Exception as e:
            last_err = e

    # 3) fallback: classe 0/1 (serve para KS)
    try:
        pred = model.predict(df)
        return np.asarray(pred, dtype=np.float32)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Falha ao escorar a base: {e if last_err is None else last_err}"
        )


@router.post("/adherence")
def adherence(payload: AdherencePayload):
    try:
        input_path = Path(payload.dataset_path)

        # lê bases
        df_in  = read_any_csv(input_path)
        df_ref = read_any_csv(REF_DATASET)

        # carrega modelo e define colunas esperadas
        model = load_model()
        expected = get_expected_cols(model, df_ref)

        # monta DataFrames alinhados por tipo (num x cat), SEM 'missing'
        X_in  = build_aligned_df(df_in,  expected, df_ref)
        X_ref = build_aligned_df(df_ref, expected, df_ref)

        # pontua
        s_in  = score_df(X_in,  model)
        s_ref = score_df(X_ref, model)

        # KS
        ks = ks_2samp(s_in, s_ref)

        return {
            "impl_version": IMPL_VERSION,
            "dataset_compared": str(input_path),
            "reference_dataset": str(REF_DATASET),
            "ks_statistic": float(ks.statistic),
            "p_value": float(ks.pvalue),
            "n_dataset": int(len(df_in)),
            "n_reference": int(len(df_ref)),
            "used_features_sample": expected[:10] + (["..."] if len(expected) > 10 else []),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao processar dataset de aderência. Erro: {e}")
