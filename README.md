# Flock Energy REFIT take-home

This repository contains a reproducible prototype for inspecting REFIT
household power data, exploring washing-machine use, and estimating
washing-machine power from whole-home aggregate readings. The work is a
research prototype, not a utility-ready customer service.

## Data setup

REFIT is a University of Strathclyde dataset of aggregate and appliance active
power from UK homes. The source readings are approximately 8 seconds apart and
in watts. The dataset is licensed CC BY 4.0. The project uses the official raw
and cleaned REFIT releases; large data files are excluded from version control.

Expected local files:

- Raw House 1 parts: `data/REFIT_RAW_081116/RAW_House1_Part1.csv` and
  `data/REFIT_RAW_081116/RAW_House1_Part2.csv`.
- Cleaned REFIT files: `CLEAN_House1.csv`, `CLEAN_House2.csv`,
  `CLEAN_House3.csv`, and `CLEAN_House5.csv` in
  `data/CLEAN_REFIT_081116/`. Alternatively, place these four files directly
  in `data/processed/`.
- `data/raw/CLEAN_READ_ME_081116.txt` for source-format and appliance-channel
  notes.

Install the only required Python packages if they are not already available:

```bash
python -m pip install pandas numpy
```

## Run the analysis

From the repository root:

```bash
python src/clean_refit_house1.py data/REFIT_RAW_081116/RAW_House1_Part1.csv data/REFIT_RAW_081116/RAW_House1_Part2.csv
python src/explore_washing_machine.py
python src/model_washing_machine.py
```

The first command writes House 1 cleaning outputs to `data/processed/` and
`reports/`. The second creates one-minute series for Houses 1, 2, 3, and 5,
cycle events, quality summaries, and EDA figures. The third fits and evaluates
the model, writes metrics and predictions, and creates an actual-versus-
predicted plot.

## Main outputs

- `DATA.md`: House 1 raw coverage, quality checks, and cleaning decisions.
- `RESULTS.md`: washing-machine EDA, cycle rule, model evaluation, plots, and
  interpretation.
- `Recommendation.md`: utility-facing summary and practical implications.
- `reports/house1_raw_quality.json`: raw House 1 inspection report.
- `reports/washing_machine_eda.json` and
  `reports/washing_machine_cycles.csv`: EDA metrics and detected events.
- `reports/model_metrics.json`: holdout metrics, split dates, and model details.

## Assumptions and trade-offs

- House 1 uses Part 1 before the first Part 2 timestamp and Part 2 from its
  start onward. The readme identifies Part 2 as the later database iteration,
  with unavailable sensors represented as NaN instead of zero.
- Raw House 1 duplicate measurements are deduplicated if identical; conflicting
  readings at the same timestamp are averaged by channel. IAM readings above
  4,000 W are set missing. Short gaps of up to two minutes are forward-filled;
  longer gaps remain missing.
- EDA resamples the official cleaned data to one-minute means and leaves
  missing minutes unfilled. The `Issues` flag is retained and reported for
  EDA. Model training, validation, and test exclude minute bins with
  `Issues > 0`.
- Washing-machine cycles use a transparent threshold heuristic. They are not
  verified cycle labels. A simple ridge regression is used because the task
  values explainable choices and reproducibility over model complexity.
- The model uses a chronological 70/10/20 split within each home. It pools
  training rows across the four homes and caps the fitting sample at 300,000
  rows. The held-out test is not used to choose model settings.
- The always-off baseline has lower MAE than the regression model. Ridge
  regression reduces RMSE modestly but has low active-minute F1, so it is not
  suitable for individual customer alerts.

## References

- Murray, D., Stankovic, L. and Stankovic, V. “An electrical load measurements
  dataset of United Kingdom households from a two-year longitudinal study.”
  *Scientific Data* 4, 160122 (2017). [doi:10.1038/sdata.2016.122](https://doi.org/10.1038/sdata.2016.122).
- Murray, D. et al. *REFIT: Electrical Load Measurements*. University of
  Strathclyde. [Dataset page](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/).

## AI tools

OpenAI Codex assisted with code and documentation. The data-handling rules,
cycle definition, split design, and reported limitations are documented in
the repository; model performance is measured on chronological held-out data.
