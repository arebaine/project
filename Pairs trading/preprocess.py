import polars as pl


def filter_market_hours(df):
    return (
        df.with_columns(pl.col("ts_event").dt.convert_time_zone("US/Eastern"))
        .filter(
            (pl.col("ts_event").dt.hour() > 10)
            | (
                (pl.col("ts_event").dt.hour() == 10)
                & (pl.col("ts_event").dt.minute() >= 30)
            )
        )
        .filter(pl.col("ts_event").dt.hour() < 15)
    )


def add_midprice(df):
    return df.with_columns(
        ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2).alias("mid")
    )


def resample_1min(df):
    return (
        df.group_by_dynamic("ts_event", every="1m", closed="right")
        .agg(pl.col("mid").last())
        .drop_nulls()
    )


def build_merged(geo_data, cxw_data):
    geo = resample_1min(add_midprice(filter_market_hours(geo_data)))
    cxw = resample_1min(add_midprice(filter_market_hours(cxw_data)))

    merged = geo.join(cxw, on="ts_event", how="inner", suffix="_cxw")

    pdf = (
        merged.select(["ts_event", "mid", "mid_cxw"])
        .rename({"mid": "mid_geo"})
        .with_columns(
            pl.col("ts_event")
            .dt.convert_time_zone("US/Eastern")
            .dt.replace_time_zone(None)
            .alias("ts_event")
        )
        .to_pandas()
        .set_index("ts_event")
    )

    return pdf
