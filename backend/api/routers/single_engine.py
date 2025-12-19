# backend/api/routers/single_engine.py
"""
Single-engine prediction router.

Use-case:
- UI sends one engine's cycle history (rows)
- API returns one calibrated RUL prediction

Design:
- Reusing inference_engine with allow_padding=True
"""

import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.api.models.requests import SingleEngineRequest
from backend.api.models.responses import SingleEnginePredictionResponse
from backend.api.services.inference_engine import run_fd_inference
from backend.api.utils.config_reader import load_unified_model_index

router = APIRouter()


@router.post("/predict", response_model=SingleEnginePredictionResponse)
def predict_single_engine(request: SingleEngineRequest):
    try:
        fd_name = request.fd_name.upper().strip()

        if not request.rows:
            raise HTTPException(status_code=400, detail="rows list is empty")

        df = pd.DataFrame(request.rows)
        df["unit"] = int(request.unit)

        if "cycle" not in df.columns:
            df["cycle"] = list(range(1, len(df) + 1))

        # Padding enabled for single-engine 
        out = run_fd_inference(fd_name, df_input=df, allow_padding=True)

        if len(out["units"]) != 1:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 1 unit output, got {len(out['units'])}. Check provided rows.",
            )

        unified_all = load_unified_model_index()
        unified_entry = unified_all.get(fd_name, {})
        model_name = (
            str(out.get("model_name") or "")
            or str(unified_entry.get("model_name") or "")
            or str(unified_entry.get("best_model_name") or "")
            or "UNKNOWN_MODEL"
        )

        return SingleEnginePredictionResponse(
            fd_name=fd_name,
            model_name=model_name,
            unit=int(out["units"][0]),
            last_cycle=int(df["cycle"].max()),
            pred_rul_raw=float(out["pred_rul_raw"][0]),
            pred_rul_calibrated=float(out["pred_rul_calibrated"][0]),
            sequence_length=int(out["sequence_length"]),
            nasa_shift=float(out["nasa_shift"]),
            nasa_max_rul_cap=float(out["nasa_max_rul_cap"]),
            allow_padding=bool(out.get("allow_padding", True)),
        )

    except HTTPException:
        raise
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
