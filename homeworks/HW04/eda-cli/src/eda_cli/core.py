from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas.api import types as ptypes


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    is_constant: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def summarize_dataset(df: pd.DataFrame, example_values_per_column: int = 3) -> DatasetSummary:
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        is_constant = (unique == 1) and (non_null > 0)

        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None

        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                is_constant=is_constant,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    return (
        pd.DataFrame({"missing_count": total, "missing_share": share})
        .sort_values("missing_share", ascending=False)
    )


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)

    for name in candidate_cols[:max_columns]:
        s = df[name]
        non_null = int(s.notna().sum())
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty or non_null == 0:
            continue

        share = vc / non_null  # доля от всех непустых значений колонки
        table = pd.DataFrame(
            {"value": vc.index.astype(str), "count": vc.values, "share": share.values}
        )
        result[name] = table

    return result


def compute_quality_flags(
    summary: DatasetSummary,
    missing_df: pd.DataFrame,
    df: Optional[pd.DataFrame] = None,
    *,
    high_cardinality_unique_threshold: int = 50,
    high_cardinality_unique_share: float = 0.5,
    zero_share_threshold: float = 0.5,
) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}

    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    # (Новая эвристика 1) Константные колонки
    constant_columns = [c.name for c in summary.columns if c.is_constant]
    flags["constant_columns"] = constant_columns
    flags["has_constant_columns"] = len(constant_columns) > 0

    # Пропуски
    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5

    # Ниже — эвристики, которым нужен df
    high_card_cols: List[str] = []
    id_dup_cols: List[str] = []
    many_zero_cols: List[str] = []

    if df is not None and not df.empty:
        n_rows = len(df)

        # (Новая эвристика 2) Высокая кардинальность категориальных признаков
        for name in df.columns:
            s = df[name]
            if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
                non_null = int(s.notna().sum())
                if non_null == 0:
                    continue
                uniq = int(s.nunique(dropna=True))
                uniq_share = uniq / non_null
                if (uniq >= high_cardinality_unique_threshold) or (uniq_share >= high_cardinality_unique_share and uniq >= 20):
                    high_card_cols.append(name)

        flags["high_cardinality_categoricals"] = high_card_cols
        flags["has_high_cardinality_categoricals"] = len(high_card_cols) > 0

        # (Новая эвристика 3) Дубликаты в подозрительных ID-колонках
        # Простая логика: колонка называется id / *_id / *id
        for name in df.columns:
            low = name.lower()
            if low == "id" or low.endswith("_id") or low.endswith("id"):
                s = df[name]
                non_null = int(s.notna().sum())
                if non_null == 0:
                    continue
                uniq = int(s.nunique(dropna=True))
                if uniq < non_null:
                    id_dup_cols.append(name)

        flags["id_columns_with_duplicates"] = id_dup_cols
        flags["has_suspicious_id_duplicates"] = len(id_dup_cols) > 0

        # (Новая эвристика 4) Слишком много нулей в числовых
        numeric_df = df.select_dtypes(include="number")
        for name in numeric_df.columns:
            s = numeric_df[name].dropna()
            if s.empty:
                continue
            zero_share = float((s == 0).mean())
            if zero_share >= zero_share_threshold:
                many_zero_cols.append(name)

        flags["many_zero_numeric_columns"] = many_zero_cols
        flags["has_many_zero_values"] = len(many_zero_cols) > 0
    else:
        # если df не дали, то просто возвращаем пустые списки, чтобы ключи были стабильны
        flags["high_cardinality_categoricals"] = []
        flags["has_high_cardinality_categoricals"] = False
        flags["id_columns_with_duplicates"] = []
        flags["has_suspicious_id_duplicates"] = False
        flags["many_zero_numeric_columns"] = []
        flags["has_many_zero_values"] = False


    score = 1.0
    score -= max_missing_share
    if flags["too_few_rows"]:
        score -= 0.2
    if flags["too_many_columns"]:
        score -= 0.1
    if flags["has_constant_columns"]:
        score -= 0.1
    if flags["has_suspicious_id_duplicates"]:
        score -= 0.1
    if flags["has_high_cardinality_categoricals"]:
        score -= 0.05
    if flags["has_many_zero_values"]:
        score -= 0.05

    flags["quality_score"] = max(0.0, min(1.0, score))
    return flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "is_constant": col.is_constant,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)