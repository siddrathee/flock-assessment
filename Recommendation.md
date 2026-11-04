# Utility-facing summary: washing-machine load insights

## What this prototype does

The prototype estimates washing-machine power from one-minute whole-home
electricity readings. It uses REFIT data from four UK households and a simple
ridge regression. This can support analysis of appliance-use patterns, but
the current model is not accurate enough for individual customer alerts or
billing decisions.

On the held-out periods, the always-off baseline had an MAE of 18.68 W and an
RMSE of 158.97 W. Ridge regression reduced RMSE to 152.48 W, but MAE rose to
28.49 W. Its active-minute F1 was 0.149, and it estimated 73.1% of actual
washing-machine energy. The improvement in RMSE is modest and does not make
the model dependable at the household level.

## Practical opportunity to test

An opt-in pilot could offer households reminders to move flexible washing
loads to a lower-cost or lower-carbon period. REFIT does not include tariff
prices or carbon intensity, so this opportunity has not been measured in
this analysis.

Measure the pilot with a randomized comparison: track verified washing-cycle
start times and washing-machine kWh before and during the pilot, and compare
the change in off-peak cycle share and total washing energy between households
receiving reminders and a control group. Validate detections with appliance
plugs or participant confirmation before sending personalized prompts.

## Limits and deployment needs

- Other appliances can create similar whole-home power patterns. Missing
  periods and REFIT `Issues` flags also weaken labels and model evaluation.
- The four households are a small UK sample. Appliance models, household
  routines, tariff structures, and climate differ in India; local data and
  validation are required before applying results there.
- At 15- or 30-minute readings, washer peaks and pauses are averaged together.
  Cycle start times, duration, and separate wash/heating/spin stages become
  much harder to distinguish, so both event detection and power estimates
  would likely be less precise.
- The current model is suitable for a research prototype and aggregate
  exploration only. It should not drive customer-facing recommendations until
  it demonstrates reliable event detection and energy estimates in a larger,
  locally representative validation set.
