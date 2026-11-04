#!/usr/bin/env python3
"""Explore cleaned REFIT washing-machine data for Houses 1, 2, 3, and 5.

Run from the repository root:
    python src/explore_washing_machine.py

The cleaned 8-second source CSVs remain unchanged. Outputs are written under
reports/ and data/processed/washing_machine_1min/.
"""

from __future__ import annotations

import json
import argparse
from html import escape
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CLEAN_DIR = ROOT / "data" / "CLEAN_REFIT_081116"
FALLBACK_CLEAN_DIR = ROOT / "data" / "processed"
SOURCE_DIR = OFFICIAL_CLEAN_DIR if (OFFICIAL_CLEAN_DIR / "CLEAN_House1.csv").is_file() else FALLBACK_CLEAN_DIR
MINUTE_DIR = ROOT / "data" / "processed" / "washing_machine_1min"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

HOUSE_CHANNELS = {
    1: "Appliance5",
    2: "Appliance2",
    3: "Appliance6",
    5: "Appliance3",
}
ACTIVE_THRESHOLD_W = 10.0
MIN_EVENT_MINUTES = 10
MAX_GAP_MINUTES = 2
MIN_EVENT_PEAK_W = 100.0
MAX_EVENT_MINUTES = 360


def load_one_house(house: int, appliance_column: str) -> tuple[pd.DataFrame, dict]:
    path = SOURCE_DIR / f"CLEAN_House{house}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cleaned REFIT source: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = ["Time", "Aggregate", appliance_column]
    if "Issues" in header:
        required.append("Issues")
    raw = pd.read_csv(path, usecols=required, parse_dates=["Time"], low_memory=False)
    raw["Aggregate"] = pd.to_numeric(raw["Aggregate"], errors="coerce")
    raw[appliance_column] = pd.to_numeric(raw[appliance_column], errors="coerce")
    if "Issues" in raw.columns:
        raw["Issues"] = pd.to_numeric(raw["Issues"], errors="coerce")

    source_row_count = int(len(raw))
    invalid_timestamp_count = int(raw["Time"].isna().sum())
    raw_missing_wm = int(raw[appliance_column].isna().sum())
    issues_flagged_source_rows = int((raw["Issues"] > 0).sum()) if "Issues" in raw.columns else 0
    raw = raw.dropna(subset=["Time"]).sort_values("Time", kind="mergesort")
    duplicate_rows = int(raw["Time"].duplicated(keep=False).sum())
    exact_duplicates = int(raw.duplicated(keep="first").sum())
    raw = raw.drop_duplicates(keep="first")
    raw = raw.groupby("Time", as_index=True).mean(numeric_only=True)
    minute = raw.resample("1min").mean()
    minute.index.name = "Time"

    valid = minute[appliance_column].notna()
    zero_minutes = int((minute[appliance_column] == 0).sum())
    off_minutes = int((minute[appliance_column] < ACTIVE_THRESHOLD_W).sum())
    summary = {
        "house": house,
        "source_file": path.name,
        "washing_machine_channel": appliance_column,
        "source_rows": source_row_count,
        "raw_washing_machine_missing_readings": raw_missing_wm,
        "invalid_timestamps": invalid_timestamp_count,
        "timestamp_duplicate_rows_involved_before_reduction": duplicate_rows,
        "exact_duplicate_rows_removed": exact_duplicates,
        "coverage_start": minute.index.min().isoformat(),
        "coverage_end": minute.index.max().isoformat(),
        "expected_minute_bins": int(len(minute)),
        "minutes_with_no_valid_washing_machine_value": int((~valid).sum()),
        "missing_washing_machine_fraction": float((~valid).mean()),
        "exact_zero_minutes": zero_minutes,
        "exact_zero_fraction_of_valid_minutes": float((minute[appliance_column].eq(0) & valid).sum() / max(int(valid.sum()), 1)),
        "off_minutes_below_10w": off_minutes,
        "off_fraction_of_valid_minutes": float((minute[appliance_column].lt(ACTIVE_THRESHOLD_W) & valid).sum() / max(int(valid.sum()), 1)),
        "issues_flagged_source_rows": issues_flagged_source_rows,
        "issues_fraction_source_rows": float(issues_flagged_source_rows / max(source_row_count, 1)),
        "washing_machine_mean_w_when_valid": float(minute.loc[valid, appliance_column].mean()) if valid.any() else None,
        "washing_machine_max_w": float(minute[appliance_column].max()) if valid.any() else None,
    }
    return minute, summary


def bridge_short_off_gaps(active: list[bool], max_gap: int) -> list[bool]:
    result = active.copy()
    i = 0
    while i < len(result):
        if result[i]:
            i += 1
            continue
        start = i
        while i < len(result) and not result[i]:
            i += 1
        gap = i - start
        if start > 0 and i < len(result) and gap <= max_gap:
            result[start:i] = [True] * gap
    return result


def find_cycles(minute: pd.DataFrame, house: int, appliance_column: str) -> tuple[list[dict], int]:
    power = minute[appliance_column]
    active = bridge_short_off_gaps(
        (power.fillna(0) >= ACTIVE_THRESHOLD_W).tolist(), MAX_GAP_MINUTES
    )
    events = []
    rejected_long_events = 0
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        start = i
        while i < len(active) and active[i]:
            i += 1
        stop = i  # exclusive
        segment = power.iloc[start:stop]
        duration = stop - start
        peak = float(segment.max()) if segment.notna().any() else 0.0
        if duration > MAX_EVENT_MINUTES:
            rejected_long_events += 1
            continue
        if duration < MIN_EVENT_MINUTES or peak < MIN_EVENT_PEAK_W:
            continue
        energy_kwh = float(segment.fillna(0).sum() / 60_000.0)
        events.append({
            "house": house,
            "channel": appliance_column,
            "start": minute.index[start].isoformat(),
            "end": minute.index[stop - 1].isoformat(),
            "duration_minutes": int(duration),
            "active_minutes_over_threshold": int((segment >= ACTIVE_THRESHOLD_W).sum()),
            "peak_power_w": round(peak, 2),
            "energy_kwh": round(energy_kwh, 4),
        })
    return events, rejected_long_events


def svg_line_plot(path: Path, title: str, subtitle: str, index: pd.DatetimeIndex,
                  series: list[tuple[str, pd.Series, str]], y_label: str) -> None:
    width, height = 1120, 510
    left, right, top, bottom = 86, 34, 74, 76
    plot_w, plot_h = width - left - right, height - top - bottom
    y_maxima = [max(float(s.dropna().quantile(0.99)) if s.notna().any() else 1, 1) for _, s, _ in series]
    x_min, x_max = index.min(), index.max()
    seconds = max((x_max - x_min).total_seconds(), 1)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="36" y="32" font-family="Arial,sans-serif" font-size="22" font-weight="bold">{escape(title)}</text>',
           f'<text x="36" y="54" font-family="Arial,sans-serif" font-size="12" fill="#475569">{escape(subtitle)}</text>']
    for fraction in (0, .25, .5, .75, 1):
        y = top + plot_h * (1 - fraction)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        out.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="11" fill="#64748b">{y_maxima[0]*fraction:.0f}</text>')
        if len(series) > 1:
            out.append(f'<text x="{left+plot_w+8}" y="{y+4:.1f}" text-anchor="start" font-family="Arial,sans-serif" font-size="11" fill="#64748b">{y_maxima[1]*fraction:.0f}</text>')
    out.append(f'<text transform="translate(20 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="Arial,sans-serif" font-size="12">{escape(series[0][0])} (W, left axis)</text>')
    if len(series) > 1:
        out.append(f'<text transform="translate({width-10} {top+plot_h/2}) rotate(90)" text-anchor="middle" font-family="Arial,sans-serif" font-size="12">{escape(series[1][0])} (W, right axis)</text>')
    for series_index, (label, values, color) in enumerate(series):
        values = values.reindex(index)
        if len(values) > 2000:
            step = max(1, len(values) // 2000)
            values = values.iloc[::step]
        coords = []
        for timestamp, value in values.items():
            if pd.isna(value):
                continue
            x = left + (timestamp - x_min).total_seconds() / seconds * plot_w
            y_max = y_maxima[series_index]
            y = top + plot_h - min(max(float(value), 0), y_max) / y_max * plot_h
            coords.append(f"{x:.1f},{y:.1f}")
        out.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="1.6"/>')
    legend_x = left
    for label, _, color in series:
        out.append(f'<line x1="{legend_x}" y1="{height-34}" x2="{legend_x+22}" y2="{height-34}" stroke="{color}" stroke-width="2"/>')
        out.append(f'<text x="{legend_x+28}" y="{height-30}" font-family="Arial,sans-serif" font-size="12">{escape(label)}</text>')
        legend_x += 210
    for fraction, timestamp in zip((0, .5, 1), (x_min, x_min + (x_max-x_min)/2, x_max)):
        x = left + plot_w * fraction
        out.append(f'<text x="{x:.1f}" y="{height-55}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#64748b">{timestamp.strftime("%Y-%m-%d %H:%M")}</text>')
    out.append(f'<text x="{left+plot_w/2}" y="{height-8}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12">Time (cleaned REFIT local timestamp)</text>')
    out.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def svg_bars(path: Path, title: str, subtitle: str, values: list[tuple[str, float]], y_label: str) -> None:
    width, height, left, right, top, bottom = 1000, 450, 82, 36, 70, 94
    plot_w, plot_h = width - left - right, height - top - bottom
    y_max = max(max((v for _, v in values), default=0) * 1.15, 1)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="32" y="32" font-family="Arial,sans-serif" font-size="21" font-weight="bold">{escape(title)}</text>',
           f'<text x="32" y="53" font-family="Arial,sans-serif" font-size="12" fill="#475569">{escape(subtitle)}</text>']
    for fraction in (0, .25, .5, .75, 1):
        y = top + plot_h * (1 - fraction)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        out.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="11" fill="#64748b">{y_max*fraction:.1f}</text>')
    out.append(f'<text transform="translate(20 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="Arial,sans-serif" font-size="12">{escape(y_label)}</text>')
    slot = plot_w / max(len(values), 1)
    bar_w = slot * .48
    for i, (label, value) in enumerate(values):
        x = left + slot * i + (slot - bar_w) / 2
        h = float(value) / y_max * plot_h
        y = top + plot_h - h
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#0f766e"/>')
        out.append(f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12">{escape(label)}</text>')
        out.append(f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#334155">{value:.1f}</text>')
    out.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def write_24h_plot(minutes: dict[int, pd.DataFrame], events: list[dict]) -> None:
    house = 1 if 1 in minutes else next(iter(minutes))
    wm = HOUSE_CHANNELS[house]
    day = None
    house_events = [e for e in events if e["house"] == house]
    if house_events:
        middle = house_events[len(house_events) // 2]
        day = pd.Timestamp(middle["start"]).normalize()
    if day is None:
        day = minutes[house].dropna(subset=[wm]).index[len(minutes[house]) // 2].normalize()
    segment = minutes[house].loc[day: day + pd.Timedelta(hours=23, minutes=59)]
    if segment.empty:
        return
    # Normalize each trace to its own p99 so both series remain visible.
    svg_line_plot(FIGURE_DIR / "wm_24h_trace_house1.svg",
                  f"House {house}: aggregate and washing-machine power over 24 hours",
                  f"Source: cleaned REFIT House {house} · {day.date()} · one-minute mean · separate visual scales",
                  segment.index,
                  [("Aggregate power", segment["Aggregate"], "#64748b"),
                   ("Washing-machine power", segment[wm], "#0f766e")],
                  "Power (W)")


def write_cycle_plot(minutes: dict[int, pd.DataFrame], events: list[dict]) -> None:
    if not events:
        return
    event = max(events, key=lambda e: e["duration_minutes"])
    house, wm = event["house"], event["channel"]
    start = pd.Timestamp(event["start"]) - pd.Timedelta(minutes=30)
    end = pd.Timestamp(event["end"]) + pd.Timedelta(minutes=30)
    segment = minutes[house].loc[start:end]
    svg_line_plot(FIGURE_DIR / "wm_detected_cycle.svg",
                  f"House {house}: detected washing-machine cycle",
                  f"{event['start']} to {event['end']} · {event['duration_minutes']} min · peak {event['peak_power_w']:.0f} W · {event['energy_kwh']:.3f} kWh",
                  segment.index,
                  [("Aggregate power", segment["Aggregate"], "#64748b"),
                   ("Washing-machine power", segment[wm], "#0f766e")],
                  "Power (W)")


def write_hour_dow_plot(events: list[dict]) -> None:
    if not events:
        return
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    counts = [[0 for _ in range(24)] for _ in range(7)]
    for event in events:
        start = pd.Timestamp(event["start"])
        counts[start.weekday()][start.hour] += 1
    width, height, left, top = 1060, 370, 92, 82
    cell_w, cell_h = 37, 31
    max_count = max(max(row) for row in counts) or 1
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="32" y="32" font-family="Arial,sans-serif" font-size="21" font-weight="bold">Detected cycle starts by weekday and hour</text>',
           '<text x="32" y="53" font-family="Arial,sans-serif" font-size="12" fill="#475569">All selected homes combined · count of detected cycle starts · source: cleaned REFIT, one-minute resolution</text>']
    for h in range(24):
        out.append(f'<text x="{left+h*cell_w+cell_w/2}" y="{top-12}" text-anchor="middle" font-family="Arial,sans-serif" font-size="10" fill="#64748b">{h:02d}</text>')
    for d, day_name in enumerate(weekday_names):
        out.append(f'<text x="{left-12}" y="{top+d*cell_h+20}" text-anchor="end" font-family="Arial,sans-serif" font-size="11">{day_name}</text>')
        for h, value in enumerate(counts[d]):
            opacity = .08 + .84 * value / max_count
            x, y = left + h*cell_w, top + d*cell_h
            out.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="#0f766e" fill-opacity="{opacity:.2}"/>')
            if value:
                out.append(f'<text x="{x+(cell_w-2)/2}" y="{y+19}" text-anchor="middle" font-family="Arial,sans-serif" font-size="10" fill="#0f172a">{value}</text>')
    out.append('</svg>')
    (FIGURE_DIR / "wm_cycle_starts_hour_weekday.svg").write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-minute-data", action="store_true",
                        help="Recalculate events and charts from existing one-minute outputs")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MINUTE_DIR.mkdir(parents=True, exist_ok=True)
    minutes: dict[int, pd.DataFrame] = {}
    summaries, events = [], []
    prior_report = None
    if args.reuse_minute_data:
        prior_report_path = REPORT_DIR / "washing_machine_eda.json"
        if not prior_report_path.is_file():
            raise FileNotFoundError("--reuse-minute-data requires an existing reports/washing_machine_eda.json")
        prior_report = json.loads(prior_report_path.read_text(encoding="utf-8"))
    for house, channel in HOUSE_CHANNELS.items():
        if args.reuse_minute_data:
            minute_path = MINUTE_DIR / f"House{house}_washing_machine_1min.csv"
            minute = pd.read_csv(minute_path, parse_dates=["Time"], index_col="Time")
            summary = next(h.copy() for h in prior_report["homes"] if h["house"] == house)
        else:
            minute, summary = load_one_house(house, channel)
            minute.to_csv(MINUTE_DIR / f"House{house}_washing_machine_1min.csv", date_format="%Y-%m-%d %H:%M:%S")
        house_events, rejected_long = find_cycles(minute, house, channel)
        summary["detected_cycle_count"] = len(house_events)
        summary["over_6_hour_candidate_segments_excluded"] = rejected_long
        summaries.append(summary)
        events.extend(house_events)
        minutes[house] = minute
        print(f"House {house}: {len(minute):,} minute bins, {len(house_events)} detected cycles")

    event_frame = pd.DataFrame(events)
    event_frame.to_csv(REPORT_DIR / "washing_machine_cycles.csv", index=False)
    valid_events = event_frame
    cross_house = []
    for summary in summaries:
        house = summary["house"]
        house_events = event_frame.loc[event_frame["house"].eq(house)] if not event_frame.empty else pd.DataFrame()
        cross_house.append({
            "house": house,
            "channel": HOUSE_CHANNELS[house],
            "cycle_count": int(len(house_events)),
            "median_duration_minutes": float(house_events["duration_minutes"].median()) if len(house_events) else None,
            "median_peak_power_w": float(house_events["peak_power_w"].median()) if len(house_events) else None,
            "median_energy_kwh": float(house_events["energy_kwh"].median()) if len(house_events) else None,
            "cycles_per_calendar_coverage_week": float(len(house_events) / max(summary["expected_minute_bins"] / (7*24*60), 1e-9)),
        })

    event_starts = [pd.Timestamp(value) for value in event_frame["start"]] if not event_frame.empty else []
    hourly = {str(h): int(sum(ts.hour == h for ts in event_starts)) for h in range(24)}
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday = {weekday_names[d]: int(sum(ts.weekday() == d for ts in event_starts)) for d in range(7)}
    hour_peak = max(hourly, key=lambda h: hourly[h]) if event_starts else None
    day_peak = max(weekday, key=weekday.get) if event_starts else None
    all_cycle_durations = event_frame["duration_minutes"] if not event_frame.empty else pd.Series(dtype=float)
    analysis = {
        "source": "University of Strathclyde REFIT official cleaned CSV files",
        "resolution": "One-minute arithmetic mean of valid 8-second readings",
        "homes": summaries,
        "cycle_detection": {
            "active_threshold_w": ACTIVE_THRESHOLD_W,
            "minimum_duration_minutes": MIN_EVENT_MINUTES,
            "short_off_gaps_merged_minutes": MAX_GAP_MINUTES,
            "minimum_peak_w": MIN_EVENT_PEAK_W,
            "maximum_duration_minutes": MAX_EVENT_MINUTES,
            "rule_plain_language": "A cycle is a run of 10 to 360 minutes where the washing-machine channel is at least 10 W, bridging off-gaps up to 2 minutes, and with at least one minute reaching 100 W. Longer segments are excluded as likely sensor or data anomalies.",
        },
        "total_detected_cycles": int(len(event_frame)),
        "cycle_hour_counts": hourly,
        "cycle_weekday_counts": weekday,
        "most_common_start_hour": hour_peak,
        "most_common_start_weekday": day_peak,
        "median_cycle_duration_minutes_all_homes": float(all_cycle_durations.median()) if len(all_cycle_durations) else None,
        "cross_house_comparison": cross_house,
        "figures": [
            "reports/figures/wm_24h_trace_house1.svg",
            "reports/figures/wm_detected_cycle.svg",
            "reports/figures/wm_cycle_starts_hour_weekday.svg",
            "reports/figures/wm_cross_house_duration.svg",
            "reports/figures/wm_cross_house_energy.svg",
        ],
    }
    (REPORT_DIR / "washing_machine_eda.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    write_24h_plot(minutes, events)
    write_cycle_plot(minutes, events)
    write_hour_dow_plot(events)
    svg_bars(FIGURE_DIR / "wm_cross_house_duration.svg",
             "Median detected washing-machine cycle duration by house",
             "Detected cycles only · source: cleaned REFIT · one-minute means",
             [(f"House {x['house']}", x["median_duration_minutes"] or 0) for x in cross_house],
             "Median duration (minutes)")
    svg_bars(FIGURE_DIR / "wm_cross_house_energy.svg",
             "Median washing-machine energy per detected cycle by house",
             "Cycle energy = sum of one-minute mean power / 60,000 · source: cleaned REFIT",
             [(f"House {x['house']}", x["median_energy_kwh"] or 0) for x in cross_house],
             "Median energy (kWh)")

    print(json.dumps({"total_detected_cycles": len(event_frame), "cross_house": cross_house,
                      "most_common_start_hour": hour_peak, "most_common_start_weekday": day_peak}, indent=2))


if __name__ == "__main__":
    main()
