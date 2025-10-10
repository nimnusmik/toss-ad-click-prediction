import math
from typing import Iterable, Tuple
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

TRAIN_PATH = DATA_DIR / 'train.parquet'
TEST_PATH = DATA_DIR / 'test.parquet'
OUTPUT_TRAIN_PATH = PROCESSED_DIR / 'train_processed_5.parquet'
OUTPUT_TEST_PATH = PROCESSED_DIR / 'test_processed_5.parquet'


def cast_core_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure core categorical columns are numeric with sentinel fill values."""
    return df.with_columns(
        [
            pl.col("gender").cast(pl.Int16, strict=False).fill_null(-1),
            pl.col("age_group").cast(pl.Int16, strict=False).fill_null(-1),
            pl.col("inventory_id").cast(pl.Int64, strict=False).fill_null(-1),
            pl.col("day_of_week").cast(pl.Int16, strict=False).fill_null(-1),
            pl.col("hour").cast(pl.Int16, strict=False).fill_null(-1),
        ]
    )


def compute_inventory_stats(train_df: pl.DataFrame, test_df: pl.DataFrame) -> Tuple[float, float]:
    combined = pl.concat(
        [train_df.select("inventory_id"), test_df.select("inventory_id")],
        how="vertical",
        rechunk=True,
    )
    inv_log_max = (
        combined.select(
            pl.when(pl.col("inventory_id") > 0)
            .then(pl.col("inventory_id").cast(pl.Float64).log1p())
            .otherwise(0.0)
            .max()
            .alias("inv_log_max")
        )["inv_log_max"][0]
        or 1.0
    )
    inv_max = (
        combined.select(
            pl.col("inventory_id").cast(pl.Float64).max().alias("inv_max")
        )["inv_max"][0]
        or 1.0
    )
    return float(inv_log_max), float(inv_max)


def compute_age_gender_bounds(
    train_df: pl.DataFrame, test_df: pl.DataFrame
) -> Tuple[int, int, int]:
    combined = pl.concat(
        [train_df.select(["age_group", "gender"]), test_df.select(["age_group", "gender"])],
        how="vertical",
        rechunk=True,
    )
    if combined.height == 0:
        return 0, 1, 0

    age_min = combined.select(pl.col("age_group").min().alias("age_min"))["age_min"][0]
    age_max = combined.select(pl.col("age_group").max().alias("age_max"))["age_max"][0]
    gender_min = combined.select(pl.col("gender").min().alias("gender_min"))["gender_min"][0]

    age_min = int(age_min) if age_min is not None else 0
    age_max = int(age_max) if age_max is not None else age_min
    gender_min = int(gender_min) if gender_min is not None else 0
    return age_min, age_max, gender_min


def compute_seq_scalers(train_df: pl.DataFrame, test_df: pl.DataFrame) -> Tuple[float, float]:
    combined = pl.concat(
        [train_df.select("seq"), test_df.select("seq")],
        how="vertical",
        rechunk=True,
    )
    length_expr = pl.col("seq").str.count_matches(",").add(1)
    length_max = (
        combined.select(length_expr.max().alias("length_max"))["length_max"][0]
        or 1
    )
    log_max = (
        combined.select(
            length_expr.cast(pl.Float64).log1p().max().alias("length_log_max")
        )["length_log_max"][0]
        or 1.0
    )
    return float(length_max), float(log_max)


def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    week_position = (pl.col("day_of_week") - 1) * 24 + pl.col("hour")
    return df.with_columns(
        [
            (pl.col("hour") / 23.0).cast(pl.Float32).alias("hour_fraction"),
            ((pl.col("day_of_week") - 1) / 6.0).cast(pl.Float32).alias("dow_fraction"),
            (2.0 * math.pi * pl.col("hour") / 24.0).sin().cast(pl.Float32).alias("hour_sin"),
            (2.0 * math.pi * pl.col("hour") / 24.0).cos().cast(pl.Float32).alias("hour_cos"),
            (2.0 * math.pi * (pl.col("day_of_week") - 1) / 7.0)
            .sin()
            .cast(pl.Float32)
            .alias("dow_sin"),
            (2.0 * math.pi * (pl.col("day_of_week") - 1) / 7.0)
            .cos()
            .cast(pl.Float32)
            .alias("dow_cos"),
            (2.0 * math.pi * week_position / 168.0)
            .sin()
            .cast(pl.Float32)
            .alias("week_pos_sin"),
            (2.0 * math.pi * week_position / 168.0)
            .cos()
            .cast(pl.Float32)
            .alias("week_pos_cos"),
            pl.when(pl.col("day_of_week").is_in([6, 7])).then(1).otherwise(0).cast(pl.Int8).alias("is_weekend"),
            pl.when(pl.col("hour") < 6)
            .then(0)
            .when(pl.col("hour") < 10)
            .then(1)
            .when(pl.col("hour") < 14)
            .then(2)
            .when(pl.col("hour") < 18)
            .then(3)
            .when(pl.col("hour") < 22)
            .then(4)
            .otherwise(5)
            .cast(pl.Int8)
            .alias("hour_bin"),
        ]
    )


def add_inventory_features(
    df: pl.DataFrame, inv_log_max: float, inv_max: float
) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.when(pl.col("inventory_id") > 0)
            .then(pl.col("inventory_id").cast(pl.Float64).log1p() / inv_log_max)
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("inventory_id_log_norm"),
            (pl.col("inventory_id").cast(pl.Float64) / inv_max)
            .clip(0.0, 1.0)
            .cast(pl.Float32)
            .alias("inventory_id_norm"),
        ]
    )


def add_age_gender_features(
    df: pl.DataFrame,
    age_min: int,
    age_max: int,
    gender_min: int,
) -> pl.DataFrame:
    span = max(age_max - age_min, 1)
    return df.with_columns(
        [
            ((pl.col("age_group") - age_min) / span).cast(pl.Float32).alias("age_group_norm"),
            (pl.col("gender") - gender_min).cast(pl.Int8).alias("gender_id"),
            (pl.col("age_group") - age_min).cast(pl.Int16).alias("age_group_id"),
        ]
    )


def add_category_statistics(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    columns: Iterable[str],
    target_col: str = "clicked",
    alpha: float = 20.0,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    global_ctr = float(train_df.select(pl.col(target_col).mean())[target_col][0])
    total_rows = max(train_df.height, 1)

    for col in columns:
        stats = train_df.group_by(col).agg(
            [
                pl.len().alias(f"{col}_freq"),
                pl.col(target_col).mean().alias(f"{col}_ctr_raw"),
            ]
        )
        stats = stats.with_columns(
            [
                (pl.col(f"{col}_freq") / total_rows)
                .cast(pl.Float32)
                .alias(f"{col}_freq_ratio"),
                (
                    (
                        pl.col(f"{col}_ctr_raw") * pl.col(f"{col}_freq")
                        + global_ctr * alpha
                    )
                    / (pl.col(f"{col}_freq") + alpha)
                )
                .cast(pl.Float32)
                .alias(f"{col}_ctr_smooth"),
                pl.col(f"{col}_freq").cast(pl.Float32),
            ]
        )
        stats = stats.with_columns(
            pl.col(f"{col}_freq").rank("dense", descending=True).cast(pl.Int32).alias(f"{col}_freq_rank")
        )
        stats = stats.drop(f"{col}_ctr_raw")

        train_df = train_df.join(stats, on=col, how="left")
        test_df = test_df.join(stats, on=col, how="left")

        fill_exprs = [
            pl.col(f"{col}_freq").fill_null(0.0),
            pl.col(f"{col}_freq_ratio").fill_null(0.0),
            pl.col(f"{col}_ctr_smooth").fill_null(global_ctr),
            pl.col(f"{col}_freq_rank").fill_null(0),
        ]
        train_df = train_df.with_columns(fill_exprs)
        test_df = test_df.with_columns(fill_exprs)

        train_df = train_df.with_columns(
            pl.col(f"{col}_freq").log1p().cast(pl.Float32).alias(f"{col}_freq_log1p")
        )
        test_df = test_df.with_columns(
            pl.col(f"{col}_freq").log1p().cast(pl.Float32).alias(f"{col}_freq_log1p")
        )

    return train_df, test_df


def add_numeric_group_features(df: pl.DataFrame) -> pl.DataFrame:
    groupings = {
        "feat_a": [c for c in df.columns if c.startswith("feat_a_")],
        "feat_b": [c for c in df.columns if c.startswith("feat_b_")],
        "feat_c": [c for c in df.columns if c.startswith("feat_c_")],
        "feat_d": [c for c in df.columns if c.startswith("feat_d_")],
        "feat_e": [c for c in df.columns if c.startswith("feat_e_")],
        "l_feat": [c for c in df.columns if c.startswith("l_feat_")],
        "history_a": [c for c in df.columns if c.startswith("history_a_")],
        "history_b": [c for c in df.columns if c.startswith("history_b_")],
    }

    for prefix, cols in groupings.items():
        if not cols:
            continue
        mean_expr = pl.mean_horizontal([pl.col(c) for c in cols]).cast(pl.Float32)
        sum_expr = pl.sum_horizontal([pl.col(c) for c in cols]).cast(pl.Float32)
        max_expr = pl.max_horizontal([pl.col(c) for c in cols]).cast(pl.Float32)
        min_expr = pl.min_horizontal([pl.col(c) for c in cols]).cast(pl.Float32)
        df = df.with_columns(
            [
                mean_expr.alias(f"{prefix}_mean"),
                sum_expr.alias(f"{prefix}_sum"),
                max_expr.alias(f"{prefix}_max"),
                min_expr.alias(f"{prefix}_min"),
                (max_expr - min_expr).alias(f"{prefix}_range"),
            ]
        )

    history_b_cols = groupings["history_b"]
    if history_b_cols:
        history_b_cols_sorted = sorted(history_b_cols, key=lambda c: int(c.split("_")[-1]))
        recent_cols = history_b_cols_sorted[-5:]
        early_cols = history_b_cols_sorted[:5]
        df = df.with_columns(
            [
                pl.mean_horizontal([pl.col(c) for c in recent_cols])
                .cast(pl.Float32)
                .alias("history_b_recent_mean"),
                pl.mean_horizontal([pl.col(c) for c in early_cols])
                .cast(pl.Float32)
                .alias("history_b_early_mean"),
                (
                    pl.mean_horizontal([pl.col(c) for c in recent_cols])
                    - pl.mean_horizontal([pl.col(c) for c in early_cols])
                )
                .cast(pl.Float32)
                .alias("history_b_recent_shift"),
            ]
        )

    return df


def add_seq_features(
    df: pl.DataFrame,
    seq_length_max: float,
    seq_log_max: float,
    chunk_size: int = 200_000,
) -> pl.DataFrame:
    if "seq" not in df.columns:
        return df

    processed_chunks = []
    for chunk in df.iter_slices(chunk_size):
        chunk_proc = chunk.with_columns(
            [
                pl.col("seq").str.count_matches(",").add(1).alias("seq_length"),
                pl.col("seq").str.split(",").list.first().cast(pl.Int32, strict=False).alias("seq_first"),
                pl.col("seq").str.split(",").list.get(-1).cast(pl.Int32, strict=False).alias("seq_last"),
            ]
        )
        chunk_proc = chunk_proc.with_columns(
            [
                (pl.col("seq_length").cast(pl.Float64) / seq_length_max)
                .clip(0.0, 1.0)
                .cast(pl.Float32)
                .alias("seq_length_norm"),
                (pl.col("seq_length").cast(pl.Float64).log1p() / seq_log_max)
                .cast(pl.Float32)
                .alias("seq_length_log_norm"),
                pl.when(
                    pl.col("seq_first").is_not_null()
                    & pl.col("seq_last").is_not_null()
                )
                .then(pl.col("seq_last") - pl.col("seq_first"))
                .otherwise(0)
                .cast(pl.Int32)
                .alias("seq_head_tail_gap"),
                pl.when(
                    pl.col("seq_first").is_not_null()
                    & pl.col("seq_last").is_not_null()
                )
                .then((pl.col("seq_first") == pl.col("seq_last")).cast(pl.Int8))
                .otherwise(0)
                .alias("seq_head_tail_same"),
                pl.when(pl.col("seq_length") <= 50)
                .then(0)
                .when(pl.col("seq_length") <= 200)
                .then(1)
                .when(pl.col("seq_length") <= 1000)
                .then(2)
                .when(pl.col("seq_length") <= 4000)
                .then(3)
                .otherwise(4)
                .cast(pl.Int8)
                .alias("seq_length_bucket"),
                pl.col("seq_first").is_null().cast(pl.Int8).alias("seq_first_missing"),
                pl.col("seq_last").is_null().cast(pl.Int8).alias("seq_last_missing"),
            ]
        )
        chunk_proc = chunk_proc.with_columns(
            pl.when(pl.col("seq_head_tail_gap").abs() > 0)
            .then(
                (
                    pl.col("seq_length").cast(pl.Float32)
                    / pl.col("seq_head_tail_gap").abs().cast(pl.Float32)
                ).clip(0.0, 1000.0)
            )
            .otherwise(0.0)
            .alias("seq_density")
        )
        processed_chunks.append(chunk_proc)

    return pl.concat(processed_chunks, how="vertical", rechunk=True)


def add_cross_features(df: pl.DataFrame) -> pl.DataFrame:
    expressions = []
    if "seq_length" in df.columns:
        expressions.extend(
            [
                pl.when(pl.col("hour") > 0)
                .then(pl.col("seq_length").cast(pl.Float32) / pl.col("hour").cast(pl.Float32))
                .otherwise(0.0)
                .alias("seq_per_hour"),
                (
                    pl.col("seq_length_norm").cast(pl.Float32)
                    * pl.col("inventory_id_ctr_smooth").cast(pl.Float32)
                )
                .alias("seq_ctr_interaction"),
            ]
        )
    expressions.extend(
        [
            (pl.col("hour") * pl.col("day_of_week")).cast(pl.Int32).alias("hour_day_product"),
            (pl.col("hour_bin") * (pl.col("age_group") + 1)).cast(pl.Int32).alias("age_hourbin_combo"),
            (
                pl.col("gender_id").cast(pl.Int32) * 10
                + pl.col("hour_bin").cast(pl.Int32)
            )
            .cast(pl.Int32)
            .alias("gender_hour_token"),
            (
                pl.col("age_group_id").cast(pl.Int32) * 10
                + pl.col("day_of_week").cast(pl.Int32)
            )
            .alias("age_dow_token"),
        ]
    )
    return df.with_columns(expressions)


def finalize_frames(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    target_col: str = "clicked",
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    if target_col in train_df.columns:
        feature_cols = [c for c in train_df.columns if c != target_col]
        train_final = pl.concat(
            [train_df.select(feature_cols), train_df.select(target_col)],
            how="horizontal",
        )
    else:
        train_final = train_df

    return train_final, test_df


def main() -> None:
    train = pl.read_parquet(str(TRAIN_PATH))
    test = pl.read_parquet(str(TEST_PATH))

    train = cast_core_columns(train)
    test = cast_core_columns(test)

    inv_log_max, inv_max = compute_inventory_stats(train, test)
    seq_length_max, seq_log_max = compute_seq_scalers(train, test)

    train = add_time_features(train)
    test = add_time_features(test)

    train = add_inventory_features(train, inv_log_max, inv_max)
    test = add_inventory_features(test, inv_log_max, inv_max)

    age_min, age_max, gender_min = compute_age_gender_bounds(train, test)

    train = add_age_gender_features(train, age_min, age_max, gender_min)
    test = add_age_gender_features(test, age_min, age_max, gender_min)

    train, test = add_category_statistics(
        train,
        test,
        columns=["inventory_id", "hour", "day_of_week", "gender", "age_group"],
    )

    train = add_numeric_group_features(train)
    test = add_numeric_group_features(test)

    train = add_seq_features(train, seq_length_max, seq_log_max)
    test = add_seq_features(test, seq_length_max, seq_log_max)

    train = add_cross_features(train)
    test = add_cross_features(test)

    train_final, test_final = finalize_frames(train, test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_final.write_parquet(str(OUTPUT_TRAIN_PATH))
    test_final.write_parquet(str(OUTPUT_TEST_PATH))


if __name__ == "__main__":
    main()
