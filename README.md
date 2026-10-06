# Modelling of Essential and Vital Worker Requirements in Catastrophic Pandemics & Rapid Scale-up of Filtration

This repository is intended to allow others to use this work to estimate national PPE stockpile requirements, model the scale-up of alternative transmission-suppressing interventions, highlight global inequalities in pandemic response measures, or otherwise work towards pandemic resilience.

This repo contains two main models that may be useful for other researchers:

1. `src/essential_workers.py` estimates essential worker counts, vital worker counts, indoor essential worker counts, and indoor vital worker counts for all 196 UN member states and 21 additional territories, alongisde equivalent clean air delivery rates (eCADRs) for each grouping. It also provides a sector breakdown for many of these countries so users can see indoor essential workers (or equivalent) in the food sector, healthcare sector, manual sector, etc. This may be useful for analysis on potential PPE stockpiles or scaling up other transmission reducing interventions.
2. `src/filtration_scale_up_model.py` models the scale-up of commercial portable air cleaners, DIY Corsi-Rosenthal boxes, and DIY coal baghouse filtration units during the first 6 months of a pandemic as transmissible as measles. This is provided globally and by UN region. We also provide country-level estimates, but recommend that these are not used unless the numbers are verified with national data.

---



## Project structure


| Path | Purpose |
| --- | --- |
| `data/essential_workers/` | ILO, O*NET, poll, crosswalk, labour force, JEM, ASHRAE-241 room types, viral loads and `parameters_essential_workers.csv` |
| `data/filtration_scale_up/` | `parameters_filtration_scale_up.csv`, coal baghouse airflow, cached World Bank MVA, Comtrade prices |
| `src/essential_workers.py` | Essential and vital workers per country and occupational group |
| `src/ecadr_requirements.py` | ASHRAE-241 eCADR scaled to the pathogen and mask efficiency, per worker |
| `src/essential_workers_validation.py` | Checks against the ILO, indoor-method sensitivity and GDP correlations |
| `src/filtration_scale_up_model.py` | Methods 2.3 scale-up: PACs, CR boxes, coal baghouse filters |
| `src/processing/` | Shared code: input loading (`preprocessing.py`), file paths and the parameters reader (`paths.py`), the regressions (`linear_models.py`) and Monte Carlo samplers (`mc_distributions.py`) |
| `scripts/` | Walkthrough notebooks for each model (`essential_workers_walkthrough`, `filtration_scale_up_walkthrough`) |
| `scripts/visualization/` | ALLFED matplotlib figure scripts |
| `results/essential_workers/` | Worker CSVs, plus `validation/` and `visualizations/` |
| `results/filtration_scale_up/` | `pacs_prioritized/`, `cr_boxes_prioritized/`, `linear_models/` and `visualizations/` |
| `tests/` | pytest suite, run on the real data |


---



## Installation

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

```bash
git clone https://github.com/ALLFED/vital-workers-filtration-scale-up.git
cd vital-workers-filtration-scale-up
uv sync --extra dev --extra notebooks
```

That creates `.venv`, installs the pinned dependencies and installs this
repository in editable mode, so `import essential_workers` and
`import processing` work from anywhere.

Without uv:

```bash
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` is generated from the lockfile, so regenerate it after
changing dependencies in `pyproject.toml`:

```bash
uv lock
uv export --format requirements-txt --no-hashes --no-emit-project -o requirements.txt
```

---



## Running the models

The filtration model needs the essential-worker outputs and the fitted
regressions, so run them in this order from the repository root:

```bash
python src/essential_workers.py              # core worker CSVs
python src/essential_workers_validation.py   # results/essential_workers/validation/
python src/processing/linear_models.py       # fits the regressions into the parameters file
python src/filtration_scale_up_model.py      # all three scenarios, both prioritisation orders
python scripts/visualization/plot_essential_workers.py
python scripts/visualization/plot_workers_vs_gdp.py
python scripts/visualization/plot_group_composition.py
python scripts/visualization/plot_filtration_coverage.py   # --scenario sets the stacked supply figure
```

`linear_models.py` writes `baghouse_gradient` and `baghouse_intercept_l_per_s`
into `parameters_filtration_scale_up.csv`, with the R² and sample size each
came from. It writes the MVA exponent `mva_exponent_b` as a normal
distribution whose 90% bounds are the pooled slope and the PRODCOM slope. Its
plots go to `results/filtration_scale_up/linear_models/`.

Both scripts refuse to run on incomplete inputs: `linear_models.py` stops while
`coal_plant_airflow.csv` is empty, and `filtration_scale_up_model.py` stops
while any uncertain parameter still has `low == high == 0`, naming each one.
On the first run the filtration model downloads manufacturing value added from
the World Bank. Check the printed country count is roughly 190-200; a much
lower number means the download failed and a stale cache was used.

The figure scripts write 300 DPI PNGs and fetch the ALLFED style sheet and
Natural Earth country polygons from the internet on first run. Maps use the
Winkel Tripel projection and [cmasher](https://cmasher.readthedocs.io/)
colormaps.

---



## Parameters

Each model reads one CSV with the columns
`name, value, low, high, distribution, units, source, notes`. Rows with a
`value` are fixed settings. Rows with `low`, `high` and `distribution` are
uncertain parameters sampled in the Monte Carlo (filtration model only).

- `data/essential_workers/parameters_essential_workers.csv` — the indoor-fraction method, group overlaps, pathogen, viral load percentile, mask efficiencies and the outdoor-airflow credit.
- `data/filtration_scale_up/parameters_filtration_scale_up.csv` — production, prices, ramp-up times, CR box and baghouse inputs, number of draws and the uncertainty interval. `adjust_MVA_by_cost` switches on the Comtrade price adjustment.

---



## Essential worker method

The ILO (2023, *World Employment and Social Outlook: The value of essential
work*) counts a worker as essential if they are in a key **occupation**
(ISCO-08) **and** a key **industry** (ISIC Rev.4). The ILO uses worker-level
microdata. We only have ILO employment by ISCO-08 level-2 code for each
country (`ilo_isco08_employment.csv`), so we approximate the intersection:

1. **Group overlaps.** For each occupational group (Food, Health, Retail,
   Security, Transport, Manual, Cleaning, Tech, Armed Forces), the overlap is
   the global share of its workers who are also in a key industry (ILO
   Figure A1). Armed Forces uses 0.40 from Blueprint Biosecurity, as the ILO
   excludes uniformed services. These are the `group_overlap_*` rows of the
   parameters file.
2. **Calibration.** For each country with ILO occupation data and an ILO
   published essential share, one scalar `x` between 0 and 1 moves all groups
   except Armed Forces together, toward 1 when the model is below the ILO and
   toward 0 when it is above. Countries without data take the mean of similar
   countries (`similar_countries.csv`), or the global overlaps if none of
   those were calibrated.
3. **Vital workers.** Team members rated each ISCO-08 4-digit occupation as
   vital or not (`isco08_opinion_poll_census.xlsx`). The ratings are averaged
   to level 2 and use the same calibrated overlaps.
4. **Indoor workers.** Essential and vital weights are multiplied by the
   indoor fraction of each occupation (0–1), from the job exposure matrix
   (`jem_binary`, the default, or `jem_partial`) or O*NET (`onet_max`,
   `onet_banded`). Set it with `indoor_context_method`.
5. **Counts.** Employment that the ILO lists as "not elsewhere classified" is
   shared out in proportion to the coded occupations. Shares are multiplied by
   the World Bank 2024 labour force, or by ILO total employment where that is
   missing.
6. **Filtration requirements.** Each group is matched to an ASHRAE-241 room
   type. Its eCADR per person is scaled up from SARS-CoV-2 to the pathogen
   (measles by default) by the ratio of their viral loads at
   `viral_load_percentile`, adjusted for mask efficiency, and reduced by the
   room's existing outdoor airflow. The COVID columns keep ASHRAE-241 as is.

`onsite_housing_worker_requirements.csv` leaves out ISCO 61 and 63 (farmers),
who are assumed to live where they work.

---



## Using the essential worker estimates

Most users only need `results/essential_workers/essential_workers_by_country.csv`.
To rerun with different assumptions, edit `value` in
`parameters_essential_workers.csv` (for example `indoor_context_method`,
`ashrae_pathogen` or `u_new_other`) and run `python src/essential_workers.py`,
or call the model from Python:

```python
import essential_workers as ew

results = ew.estimate(indoor_context_method="jem_binary")
by_country = results["by_country"]   # also by_group, by_region, onsite_housing
ew.write_results(results)            # writes the CSVs below
```

`estimate(parameters_file=...)` accepts a copy of the parameters file, so
alternative assumptions can be kept side by side.

Columns starting with `%` are fractions of the labour force (0–1, not 0–100).
Worker columns are numbers of people.

**`essential_workers_by_country.csv`** (one row per country or territory)

| Column | Units | Meaning |
| --- | --- | --- |
| `Country Name`, `Country Code`, `Region` | | Name, ISO3 code and UN subregion |
| `Labour Force (2024)` | people | World Bank labour force, or ILO total employment where missing |
| `%Essential Workers`, `%Vital Workers` | fraction | Share of the labour force |
| `%Indoor Essential Workers`, `%Indoor Vital Workers` | fraction | The same, weighted by the indoor fraction of each occupation |
| `Essential Workers`, `Vital Workers`, `Indoor Essential Workers`, `Indoor Vital Workers` | people | Share × labour force |
| `Indoor Essential CADR Requirement (L/s)`, `Indoor Vital CADR Requirement (L/s)` | L/s | Total eCADR needed for the pathogen in the parameters file |
| `Indoor Essential CADR Requirement COVID (L/s)`, `Indoor Vital CADR Requirement COVID (L/s)` | L/s | The same for a COVID-like pathogen |
| `Scaled ECA Essential (L/s/person)`, `Scaled ECA Vital (L/s/person)` | L/s per person | Requirement divided by indoor workers |

**`essential_workers_by_group.csv`** has one row per country and occupational
group for the 144 countries with ILO occupation data, with the
four worker counts, the four requirement columns, `Scaled ECA (L/s/person)`
and the ASHRAE-241 room type (`occupancy_group`, `occupancy_category`).

**`essential_workers_by_region.csv`** has the by-country columns summed over
each UN subregion, with shares recomputed from the sums.

**`onsite_housing_worker_requirements.csv`**: `Essential Workers` and
`Vital Workers` (people), and the same without ISCO 61 and 63 in the
`(Housing Requirement)` columns.

**`group_composition_global.csv`** and **`group_composition_by_region.csv`**:
the share (fraction) of each workforce in each occupational group.

**`ashrae241_scaled_table1.csv`**: per group, the ASHRAE-241 room type,
eCADR (L/s per person), eACH (air changes per hour), volume per occupant (m³),
the scaling factor `k` and the scaled eCADR and eACH.
**`ashrae241_scaled_by_mask_efficiency.csv`** gives the scaled eCADR and eACH
for each mask efficiency in `mask_efficiencies`, with negative values shown as
zero.

---



## Results

- `results/essential_workers/` — the CSVs described above.
  - `validation/` — the model and calibrated shares against the ILO, the overlap calibration for each country, the indoor-method sensitivity, and worker shares against GDP per capita with their correlations.
  - `visualizations/` — country maps of the four worker shares, the occupational make-up of each workforce and two GDP scatter plots.
- `results/filtration_scale_up/pacs_prioritized/` — the default run, where panel filters stay with PACs. For each scenario:
  - `weekly_ecadr_by_country_*` — median cumulative eCADR (L/s) by country and week.
  - `ecadr_by_channel_*` — median global eCADR (L/s) by supply channel and week.
  - `coverage_{vital,essential}[_covid]_*` — the median and uncertainty interval of the share (fraction) of the requirement met, by region and week.
  - `requirements_by_region.csv` — the eCADR (L/s) each region is measured against.
- `results/filtration_scale_up/cr_boxes_prioritized/` — the same outputs when panel filters are diverted to CR boxes.
- `results/filtration_scale_up/linear_models/` — plots of the two fitted regressions.
- `results/filtration_scale_up/visualizations/` — scenario coverage, mask-efficiency coverage, stacked supply by channel for both prioritisation orders, and regional maps at 3 and 6 months.

---



## Tests

```bash
pytest
```
