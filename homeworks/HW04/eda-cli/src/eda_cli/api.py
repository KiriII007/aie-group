from __future__ import annotations

from io import StringIO
from time import perf_counter
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from .core import compute_quality_flags, missing_table, summarize_dataset


app = FastAPI(
    title="eda-cli API",
    description="HTTP-сервис качества датасетов поверх eda-cli (HW04).",
    version="0.1.0",
)


def _read_csv_upload(file: UploadFile, *, sep: str, encoding: str) -> pd.DataFrame:
    try:
        raw = file.file.read()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Cannot read upload: {exc}") from exc

    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        text = raw.decode(encoding, errors="strict")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Cannot decode file with {encoding}: {exc}") from exc

    try:
        df = pd.read_csv(StringIO(text), sep=sep)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {exc}") from exc

    return df


def _shape_dict(df: pd.DataFrame) -> Dict[str, int]:
    return {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])}


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class QualityRequest(BaseModel):
    n_rows: int = Field(..., ge=0)
    n_cols: int = Field(..., ge=0)
    max_missing_share: float = Field(..., ge=0.0, le=1.0)
    numeric_cols: int = Field(..., ge=0)
    categorical_cols: int = Field(..., ge=0)


class QualityResponse(BaseModel):
    ok_for_model: bool
    quality_score: float = Field(..., ge=0.0, le=1.0)
    message: str
    latency_ms: float = Field(..., ge=0.0)
    flags: Dict[str, Any]
    dataset_shape: Dict[str, int]


class FlagsResponse(BaseModel):
    flags: Dict[str, Any]
    latency_ms: float = Field(..., ge=0.0)
    dataset_shape: Dict[str, int]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="eda-cli", version=app.version or "unknown")


@app.post("/quality", response_model=QualityResponse)
def quality(req: QualityRequest) -> QualityResponse:
    
    t0 = perf_counter()

    too_few_rows = req.n_rows < 100
    too_many_columns = req.n_cols > 100
    too_many_missing = req.max_missing_share > 0.5

    score = 1.0
    score -= req.max_missing_share
    if too_few_rows:
        score -= 0.2
    if too_many_columns:
        score -= 0.1
    if too_many_missing:
        score -= 0.2
    score = max(0.0, min(1.0, score))

    ok_for_model = (score >= 0.5) and not too_many_missing and (req.n_rows > 0) and (req.n_cols > 0)
    msg = "OK" if ok_for_model else "Dataset looks risky for modeling"

    latency_ms = (perf_counter() - t0) * 1000.0
    flags = {
        "too_few_rows": too_few_rows,
        "too_many_columns": too_many_columns,
        "too_many_missing": too_many_missing,
    }

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=msg,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


@app.post("/quality-from-csv", response_model=QualityResponse)
def quality_from_csv(
    file: UploadFile = File(..., description="CSV-файл"),
    sep: str = Query(",", description="Разделитель CSV"),
    encoding: str = Query("utf-8", description="Кодировка CSV"),
) -> QualityResponse:
    
    t0 = perf_counter()

    df = _read_csv_upload(file, sep=sep, encoding=encoding)
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV parsed, but dataset is empty")

    summary = summarize_dataset(df)
    miss = missing_table(df)
    flags = compute_quality_flags(summary, miss, df)

    score = float(flags.get("quality_score", 0.0))
    ok_for_model = (score >= 0.5) and (not bool(flags.get("too_many_missing", False)))

    msg = "OK" if ok_for_model else "Dataset looks risky for modeling"
    latency_ms = (perf_counter() - t0) * 1000.0

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=msg,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape=_shape_dict(df),
    )


@app.post("/quality-flags-from-csv", response_model=FlagsResponse)
def quality_flags_from_csv(
    file: UploadFile = File(..., description="CSV-файл"),
    sep: str = Query(",", description="Разделитель CSV"),
    encoding: str = Query("utf-8", description="Кодировка CSV"),
) -> FlagsResponse:

    t0 = perf_counter()

    df = _read_csv_upload(file, sep=sep, encoding=encoding)
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV parsed, but dataset is empty")

    summary = summarize_dataset(df)
    miss = missing_table(df)
    flags = compute_quality_flags(summary, miss, df)

    latency_ms = (perf_counter() - t0) * 1000.0
    return FlagsResponse(flags=flags, latency_ms=latency_ms, dataset_shape=_shape_dict(df))