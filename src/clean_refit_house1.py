#!/usr/bin/env python3
"""Inspect raw REFIT House 1 readings and create a one-minute cleaned series.

Run:
    python src/clean_refit_house1.py data/REFIT_RAW_081116/RAW_House1_Part1.csv data/REFIT_RAW_081116/RAW_House1_Part2.csv

The script writes a quality report, cleaned CSV, and before/after plot under
reports/ and data/processed/. Raw input data is not modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from html import escape


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
OUTPUT_DIR = ROOT / "data" / "processed"
SENSOR_LIMIT_W = 4000.0
MAX_SHORT_GAP_MINUTES = 2

# House 1 channel mapping in the REFIT metadata / readme (channels are active
# power in watts). Appliance1 and Appliance5 are shown as representative IAMs.
PLOT_CHANNELS = ["Aggregate", "Appliance1", "Appliance5"]


def locate_time_column(columns: list[str]) -> str:
    candidates = {str(c).strip().lower(): str(c) for c in columns}
    for candidate in ("datetime", "time", "timestamp"):
        if candidate in candidates:
            return candidates[candidate]
    raise ValueError(f"Could not find a timestamp column in: {columns}")


def load_raw(paths: list[Path]) -> tuple[pd.DataFrame, str]:
    parts = []
    first_time_column = "Time"
    for path in paths:
        part = pd.read_csv(path, low_memory=False)
        time_column = locate_time_column(list(part.columns))
        if len(parts) == 0:
            first_time_column = time_column
        part = part.rename(columns={time_column: "Time"})
        part["SourceFile"] = path.name
        part["SourcePart"] = "Part2" if "part2" in path.name.lower() else (
            "Part1" if "part1" in path.name.lower() else path.stem
        )
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True, sort=False)
    frame["Time"] = pd.to_datetime(frame["Time"], errors="coerce").dt.floor("s")
    return frame, first_time_column


def summarize_raw(frame: pd.DataFrame) -> dict:
    timestamp = frame["Time"]
    valid_time = timestamp.dropna()
    deltas = valid_time.sort_values().diff().dropna().dt.total_seconds()
    numeric_columns = to_numeric_channels(frame)
    raw_missing = {
        c: int(frame[c].isna().sum()) for c in numeric_columns
    }
    nonnumeric_counts = {}
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        nonnumeric_counts[column] = int((frame[column].notna() & values.isna()).sum())
        frame[column] = values

    out_of_order = int((timestamp.diff().dt.total_seconds() < 0).sum())
    interval_summary = {
        "median_seconds": float(deltas.median()) if len(deltas) else None,
        "min_seconds": float(deltas.min()) if len(deltas) else None,
        "max_seconds": float(deltas.max()) if len(deltas) else None,
        "intervals_over_2_minutes": int((deltas > 120).sum()),
    }
    by_source = {}
    for source, part in frame.groupby("SourcePart", sort=False):
        valid = part["Time"].dropna()
        counts = valid.value_counts()
        by_source[str(source)] = {
            "rows": int(len(part)),
            "start": valid.min().isoformat() if len(valid) else None,
            "end": valid.max().isoformat() if len(valid) else None,
            "duplicate_timestamp_rows_involved": int(counts[counts > 1].sum()),
            "extra_rows_at_duplicate_timestamps": int((counts[counts > 1] - 1).sum()),
            "exact_duplicate_measurement_rows": int(part.drop(columns=["SourceFile", "SourcePart"]).duplicated().sum()),
        }
    return {
        "rows": int(len(frame)),
        "invalid_timestamps": int(timestamp.isna().sum()),
        "start": valid_time.min().isoformat() if len(valid_time) else None,
        "end": valid_time.max().isoformat() if len(valid_time) else None,
        "out_of_order_rows": out_of_order,
        "source_parts": by_source,
        "raw_missing_by_channel": raw_missing,
        "nonnumeric_values_by_channel": nonnumeric_counts,
        "timestamp_interval_seconds": interval_summary,
    }


def to_numeric_channels(frame: pd.DataFrame) -> list[str]:
    return [
        c for c in frame.columns
        if c.strip().lower() != "time"
        and not c.strip().lower().startswith("unix")
        and c.strip().lower() not in {"issues", "sourcefile", "sourcepart"}
    ]


def select_source_version(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    part2 = frame.loc[frame["SourcePart"].eq("Part2")]
    part1 = frame.loc[frame["SourcePart"].eq("Part1")]
    if part1.empty or part2.empty:
        return frame.copy(), {"rule": "No Part1/Part2 overlap rule applied", "part1_rows_excluded": 0}
    part2_start = part2["Time"].min()
    earlier_part1 = part1.loc[part1["Time"] < part2_start]
    selected = pd.concat([earlier_part1, part2], ignore_index=True, sort=False)
    return selected, {
        "rule": "Use Part1 before Part2 begins; use Part2 from its first timestamp onward because the readme identifies Part2 as the later database iteration with unavailable sensors represented as NaN",
        "part2_start": part2_start.isoformat(),
        "part1_rows_excluded_from_overlap": int(len(part1) - len(earlier_part1)),
        "part1_rows_retained_before_part2_start": int(len(earlier_part1)),
        "part2_rows_retained": int(len(part2)),
    }


def clean_and_resample(frame: pd.DataFrame, channels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = frame[["Time", *channels]].copy()
    data = data.dropna(subset=["Time"])
    data = data.sort_values("Time", kind="mergesort")

    # Remove repeated copies of the same measurement row. For timestamp
    # collisions with differing measurements, collapse each channel by mean.
    exact_duplicates = int(data.duplicated(subset=["Time", *channels]).sum())
    data = data.drop_duplicates(subset=["Time", *channels], keep="first")
    duplicate_timestamp_rows = int(data["Time"].duplicated(keep=False).sum())

    # Negative active-power readings are invalid. IAMs are rated to 4 kW;
    # replace above-limit IAM samples with missing values before aggregation.
    negative_counts = {}
    above_limit_counts = {}
    aggregate_above_20kw = 0
    for channel in channels:
        negative = data[channel] < 0
        negative_counts[channel] = int(negative.sum())
        data.loc[negative, channel] = pd.NA
        if channel.strip().lower() == "aggregate":
            aggregate_above_20kw = int((data[channel] > 20_000).sum())
        else:
            above_limit = data[channel] > SENSOR_LIMIT_W
            above_limit_counts[channel] = int(above_limit.sum())
            data.loc[above_limit, channel] = pd.NA

    # Resolve remaining timestamp collisions by the mean per channel.
    data = data.groupby("Time", as_index=False, sort=True)[channels].mean()
    data = data.set_index("Time")

    # Keep the unfilled one-minute average for the comparison plot.
    minute_means = data.resample("1min").mean()
    missing_minutes = minute_means.isna()
    missing_before_fill = {c: int(missing_minutes[c].sum()) for c in channels}

    # Carry forward at most two consecutive missing minute bins. Longer outages
    # remain NaN and are reported; they are not fabricated as continuous use.
    cleaned = minute_means.ffill(limit=MAX_SHORT_GAP_MINUTES)
    filled_counts = {
        c: int((minute_means[c].isna() & cleaned[c].notna()).sum()) for c in channels
    }
    remaining_missing = {c: int(cleaned[c].isna().sum()) for c in channels}
    cleaned.index.name = "Time"
    return minute_means, cleaned, {
        "exact_duplicate_measurement_rows_removed": exact_duplicates,
        "rows_involved_in_conflicting_timestamp_duplicates": duplicate_timestamp_rows,
        "negative_readings_set_missing": negative_counts,
        "iam_readings_above_4000w_set_missing": above_limit_counts,
        "aggregate_readings_above_20000w_flagged_and_retained": aggregate_above_20kw,
        "missing_minute_bins_before_short_fill": missing_before_fill,
        "minute_bins_filled_forward": filled_counts,
        "missing_minute_bins_left_unfilled": remaining_missing,
        "resampling": "Arithmetic mean of valid readings in each 1-minute bin",
        "duplicate_rule": "Remove exact repeated measurements; average differing readings that share a timestamp per channel",
        "short_gap_rule": "Forward-fill up to 2 consecutive missing one-minute bins per channel",
        "long_gap_rule": "Leave longer gaps missing and flag them in the report",
    }


def make_plot(before: pd.DataFrame, cleaned: pd.DataFrame, channels: list[str], output: Path) -> None:
    available = [c for c in PLOT_CHANNELS if c in channels]
    if len(available) < 3:
        available = channels[: min(3, len(channels))]
    width, panel_h, margin_left, margin_right = 1200, 210, 95, 35
    top, gap = 50, 32
    height = top + len(available) * panel_h + gap * (len(available) - 1) + 45
    plot_w = width - margin_left - margin_right
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append('<text x="40" y="30" font-family="Arial,sans-serif" font-size="21" font-weight="bold">REFIT House 1: raw and cleaned one-minute readings</text>')
    x0 = margin_left
    xmin, xmax = before.index.min(), before.index.max()
    total_seconds = max((xmax - xmin).total_seconds(), 1)

    def path_for(series: pd.Series, y_max: float, y_top: float, y_bottom: float) -> str:
        valid = series.dropna()
        if len(valid) > 2500:
            valid = valid.iloc[:: max(1, len(valid) // 2500)]
        coords = []
        for timestamp, value in valid.items():
            x = x0 + (timestamp - xmin).total_seconds() / total_seconds * plot_w
            y = y_bottom - min(max(float(value), 0), y_max) / y_max * (y_bottom - y_top)
            coords.append(f"{x:.1f},{y:.1f}")
        return " ".join(coords)

    for i, channel in enumerate(available):
        y_top = top + i * (panel_h + gap) + 12
        y_bottom = y_top + panel_h - 40
        combined = pd.concat([before[channel], cleaned[channel]]).dropna()
        y_max = max(float(combined.quantile(0.99)) if len(combined) else 1.0, 1.0)
        parts.append(f'<text x="40" y="{y_top + 12}" font-family="Arial,sans-serif" font-size="15" font-weight="bold">{escape(channel)} (W, axis to p99)</text>')
        for fraction, label in [(0, "0"), (0.5, f"{y_max/2:.0f}"), (1, f"{y_max:.0f}")]:
            y = y_bottom - fraction * (y_bottom - y_top - 20)
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/>')
            parts.append(f'<text x="{x0-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="11" fill="#64748b">{label}</text>')
        parts.append(f'<polyline points="{path_for(before[channel], y_max, y_top, y_bottom)}" fill="none" stroke="#94a3b8" stroke-width="1"/>')
        parts.append(f'<polyline points="{path_for(cleaned[channel], y_max, y_top, y_bottom)}" fill="none" stroke="#0f766e" stroke-width="1.2"/>')
    legend_y = height - 20
    parts.append(f'<line x1="{margin_left}" y1="{legend_y-4}" x2="{margin_left+25}" y2="{legend_y-4}" stroke="#94a3b8" stroke-width="2"/><text x="{margin_left+32}" y="{legend_y}" font-family="Arial,sans-serif" font-size="12">Before cleaning (raw minute means)</text>')
    parts.append(f'<line x1="{margin_left+260}" y1="{legend_y-4}" x2="{margin_left+285}" y2="{legend_y-4}" stroke="#0f766e" stroke-width="2"/><text x="{margin_left+292}" y="{legend_y}" font-family="Arial,sans-serif" font-size="12">After cleaning</text>')
    parts.append('</svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, nargs="+",
                        help="One or more raw REFIT House 1 CSV parts, in any order")
    args = parser.parse_args()
    input_paths = [path.expanduser().resolve() for path in args.input_csv]
    missing_paths = [path for path in input_paths if not path.is_file()]
    if missing_paths:
        parser.error("Input CSV does not exist: " + ", ".join(str(p) for p in missing_paths))

    raw, original_time_column = load_raw(input_paths)
    raw_summary = summarize_raw(raw)
    channels = to_numeric_channels(raw)
    selected, source_selection = select_source_version(raw)
    selected_for_plot = selected.dropna(subset=["Time"]).sort_values("Time")
    before = selected_for_plot.groupby("Time", as_index=True)[channels].mean().resample("1min").mean()
    before.index.name = "Time"
    minute_means, cleaned, cleaning_summary = clean_and_resample(selected, channels)
    cleaning_summary["source_version_selection"] = source_selection
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_path = OUTPUT_DIR / "house1_1min_clean.csv"
    cleaned.to_csv(cleaned_path, index_label="Time", date_format="%Y-%m-%d %H:%M:%S")

    plot_path = FIGURE_DIR / "house1_cleaning_comparison.svg"
    make_plot(before, cleaned, channels, plot_path)

    report = {
        "source_files": [path.name for path in input_paths],
        "household": 1,
        "timestamp_column_in_source": original_time_column,
        "units": "Watts (W), as supplied by REFIT",
        "raw_inspection": raw_summary,
        "cleaning": cleaning_summary,
        "outputs": {
            "cleaned_csv": str(cleaned_path.relative_to(ROOT)),
            "comparison_plot": str(plot_path.relative_to(ROOT)),
        },
        "notes": [
            "Aggregate values above 20 kW are counted and flagged for review, not removed automatically.",
            "The 4 kW sensor limit is applied only to individual appliance monitor (IAM) channels.",
            "The plot uses Aggregate, Appliance1 (Fridge), and Appliance5 (Washing Machine) when present.",
        ],
    }
    report_path = REPORT_DIR / "house1_raw_quality.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
