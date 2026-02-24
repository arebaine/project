import datetime as dt
from pathlib import Path
import polars as pl


def load_symbol_folder(folder_path, start_date=None, end_date=None):
    def to_date(x):
        if x is None:
            return None
        if isinstance(x, dt.date):
            return x
        if isinstance(x, str):
            return dt.date.fromisoformat(x)
        raise TypeError(
            "start_date/end_date must be datetime.date, 'YYYY-MM-DD' string, or None"
        )

    start_date = to_date(start_date)
    end_date = to_date(end_date)

    files = sorted(Path(folder_path).glob("*.parquet"))

    def file_date(p: Path) -> dt.date:
        # expects: <symbol>_YYYY-MM-DD.parquet
        ds = p.stem.rsplit("_", 1)[-1]
        return dt.date.fromisoformat(ds)

    selected = []
    for f in files:
        d = file_date(f)
        if start_date is not None and d < start_date:
            continue
        if end_date is not None and d > end_date:
            continue
        selected.append(f)

    first_schema = pl.read_parquet_schema(selected[0])
    n_cols = len(first_schema)

    valid_files = []
    for f in selected:
        schema = pl.read_parquet_schema(f)
        if len(schema) == n_cols:
            valid_files.append(f)

    df = pl.concat([pl.read_parquet(f) for f in valid_files], how="vertical").sort(
        "ts_event"
    )
    return df
