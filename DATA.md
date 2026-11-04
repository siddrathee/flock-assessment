# House 1 raw data inspection and cleaning

## Source and current status

The assignment specifies the University of Strathclyde REFIT raw measurements,
House 1. The available files are `RAW_House1_Part1.csv` and
`RAW_House1_Part2.csv`; the supplied `CLEAN_READ_ME_081116.txt` states that
readings are active power in watts, normally sampled about every 8 seconds.
It also explains that Part 1 encodes unavailable sensors as zero, while Part 2
uses NaN, and that the two parts are different database iterations.

The two CSV parts are in `data/REFIT_RAW_081116/`; the cleaned readme is in
`data/raw/`. Run:

```bash
python src/clean_refit_house1.py data/REFIT_RAW_081116/RAW_House1_Part1.csv data/REFIT_RAW_081116/RAW_House1_Part2.csv
```

The script uses Part 1 before the earliest Part 2 timestamp, then Part 2 for
the overlapping and later period. This follows the readme's description of
Part 2 as a later database iteration and avoids averaging its NaNs with Part 1
placeholder zeros. It records the overlap selection and discarded Part 1
rows in the quality report. Other household files are not needed for this
House 1 task.

The script accepts `DateTime`, `Time`, or `Timestamp` as the time column. It
writes the cleaned one-minute CSV to `data/processed/`, the inspection report
to `reports/house1_raw_quality.json`, and a before/after plot to
`reports/figures/`.

## Cleaning decisions

- Parse timestamps, count invalid values and out-of-order rows, and sort by
  timestamp before processing.
- Remove exact repeated measurement rows. When readings at the same timestamp
  differ, average them per channel so they do not receive extra weight.
- Treat negative active power as invalid. Set it missing and report the count.
- Treat IAM values above 4,000 W as invalid because the individual appliance
  monitor sensor limit is 4 kW. Set those values missing and report the count.
- Report unusually high aggregate readings above 20 kW for review, but retain
  them automatically because an aggregate spike cannot be judged from the
  sensor limit for individual appliance monitors.
- Resample to 1-minute arithmetic means. For a missing minute, carry the last
  value forward for at most two consecutive minutes. Leave longer outages
  missing and report them rather than inventing a continuous trace.

The script reports raw timestamp intervals, duplicate timestamps, missing and
nonnumeric values, invalid readings, and missing minute bins before and after
short-gap handling. It outputs an SVG comparison plot so no plotting package
is required.

## Initial House 1 findings

- The two files contain 8,518,763 rows across 2013-10-09 14:06:17 to
  2015-07-10 11:56:32. The retained series uses Part 1 through
  2014-06-27 21:57:49, then Part 2 from 2014-06-27 21:57:50 onward.
- Part 1 has 1,211,004 rows involved in repeated timestamps; 321,692 exact
  repeated measurement rows are removed after source-version selection.
  Three rows in the combined raw file are out of timestamp order.
- The median difference between ordered raw timestamps is 6 seconds, with 530
  intervals over two minutes and a maximum gap of about 41.6 days. The readme
  notes readings are polled at roughly 6-8 second intervals and that real
  outages occur.
- No negative values or nonnumeric channel values were found. There are 104
  aggregate readings above 20 kW; these are flagged and retained for review.
  IAM readings above 4 kW are set missing, with counts recorded per channel in
  `reports/house1_raw_quality.json`.
- The readme maps Appliance1 to the fridge and Appliance5 to the washing
  machine; these are the two appliance channels shown with the aggregate in
  the comparison plot.
- After resampling, long missing periods remain unfilled. For example, the
  aggregate channel has 118,151 missing one-minute bins after filling gaps of
  at most two minutes. See the JSON report for each channel's counts.

These findings use the supplied cleaned readme to interpret the two raw-file
iterations. Part 1 uses zero for unavailable sensors, so true off readings and
unavailable readings cannot always be distinguished in that earlier period.
