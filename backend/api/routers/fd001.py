# backend/api/routers/fd001.py
"""
FD001 RUL prediction router.

Responsibilities:
- Parse request into DataFrame
- Validate required columns
- Enforce strict batch behavior (>= seq_len cycles per unit)
- Delegate inference to service layer
"""

from fastapi import APIRouter, HTTPException
import pandas as pd

from backend.api.models.requests import FD001BatchRequest
from backend.api.models.responses import FD001PredictionResponse, EnginePrediction
from backend.api.services.inference_engine import run_fd_inference
from backend.api.utils.config_reader import load_fd_config

router = APIRouter()


@router.post("/fd001", response_model=FD001PredictionResponse)
def predict_fd001(request: FD001BatchRequest):
    try:
        if not request.records:
            raise HTTPException(status_code=400, detail="records list is empty")

        df = pd.DataFrame(request.records)

        for col in ("unit", "cycle"):
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Missing required column: '{col}'")

        cfg = load_fd_config("FD001")
        seq_len = int(cfg["sequence_length"])

        counts = df.groupby("unit")["cycle"].count()
        too_short = counts[counts < seq_len]
        if not too_short.empty:
            example_unit = int(too_short.index[0])
            example_len = int(too_short.iloc[0])
            raise HTTPException(
                status_code=400,
                detail=(
                    f"FD001 batch prediction requires at least {seq_len} cycles per unit. "
                    f"unit={example_unit} has only {example_len}. "
                    f"Send >= {seq_len} records for that unit, or use /single/predict for padded demo inference."
                ),
            )

        out = run_fd_inference("FD001", df_input=df, allow_padding=False)

        preds = [
            EnginePrediction(
                unit=int(unit),
                pred_rul_raw=float(raw),
                pred_rul_calibrated=float(cal),
                true_rul=None,
            )
            for unit, raw, cal in zip(out["units"], out["pred_rul_raw"], out["pred_rul_calibrated"])
        ]

        return FD001PredictionResponse(
            fd_name="FD001",
            model_name=str(out.get("model_name", "UNKNOWN_MODEL")),
            predictions=preds,
            sequence_length=int(out["sequence_length"]),
            nasa_shift=float(out["nasa_shift"]),
            nasa_max_rul_cap=float(out["nasa_max_rul_cap"]),
            allow_padding=bool(out.get("allow_padding", False)),
            units_count=len(preds),
            message="ok",
        )

    except HTTPException:
        raise
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
