# Grid Optimizer Suite

## Description of Code Package

This grid optimizer suite contains code used to model and optimize modern energy grids. The primary functions addressed are:

* Collecting and collating energy input data
* Modeling grid and battery behavior
* Optimizing energy resource mixes
* Visualizing and interpreting results

This code was intentionally not packaged into a graphical user interface (GUI). The intent is for users to modify and adapt the code to meet their own energy modeling objectives. The framework can be adapted to model virtually any electric grid, and new optimization methods or visualizations can be incorporated as needed.

---

# Table of Contents

1. [Overview of Code Use](#overview-of-code-use)
2. [Data Collection and Collation](#data-collection-and-collation)

   * [Solar Data Collection and Collation](#solar-data-collection-and-collation)
   * [Wind Data Collection and Collation](#wind-data-collection-and-collation)
   * [EV Data Collection and Collation](#ev-data-collection-and-collation)
   * [Electric Vehicle Battery Wear](#electric-vehicle-battery-wear)
   * [Supply, Demand, Imports, Natural Gas, and Hydropower Collection](#supply-demand-imports-natural-gas-and-hydropower-collection)
   * [The Final Dataset](#the-final-dataset)
3. [Running the Simulation](#running-the-simulation)

   * [Random Sample with Pruning Optimization](#random-sample-with-pruning-optimization)
   * [Cleaning the Optimized Scenarios](#cleaning-the-optimized-scenarios)
   * [Sensitivity Analysis](#sensitivity-analysis)
   * [Grid Building](#grid-building)

---

# Overview of Code Use

The workflow begins by collecting and collating the required input data.

Typical inputs include:

* Solar generation data from the NREL NSRDB API
* Wind data from the ECMWF ERA5 API
* Grid demand and generation data from the EIA Hourly Grid Monitor and EIA API

Data may be obtained from any source, provided that a complete year of hourly observations is available.

After collection, the data are processed:

* Solar data are adjusted to account for single-axis tracking.
* Wind speeds are converted into expected wind power output.
* Historical generation data are aligned with modeled outputs.

The model is then tuned to account for factors such as:

* Transmission losses
* Installed capacity that is unavailable or operating below rated output

A series of Monte Carlo simulations are used to align modeled generation with observed generation.

Once the model is tuned, optimization can be performed. The primary optimization method implemented in this package is a **Discretized Random Sample with Pruning** approach. Users may substitute alternative optimization methods if desired.

The workflow is:

1. Generate a sample space of candidate grid configurations.
2. Randomly sample configurations.
3. Evaluate system performance.
4. Eliminate non-viable configurations through pruning logic.
5. Continue until optimal solutions are identified.

This approach allows exploration of extremely large solution spaces while requiring far fewer evaluations than exhaustive search methods.

---

# Data Collection and Collation

## Solar Data Collection and Collation

Solar data were collected using the NREL NSRDB API.

Documentation is available at:

https://developer.nrel.gov/docs/solar/nsrdb/

An API key is required.

The included `Solar_API_Grab.py` script retrieves data from selected locations. The default configuration downloads data from five locations in California, but users may modify the locations as needed.

Output is written to:

```text
SOL_API_OUT/
```

Because multiple locations and many years of data may be requested, execution can take several minutes due to API limits.

The downloaded data should then be processed using PVLib to simulate single-axis tracking systems. This process:

* Converts irradiance data into expected solar generation
* Calculates hourly means
* Calculates hourly standard deviations
* Produces CSV files suitable for Monte Carlo simulation

---

## Wind Data Collection and Collation

Wind data were collected using ECMWF ERA5 reanalysis data.

Data can be obtained from:

https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels

Users should obtain wind speeds at approximately 100 m elevation, which is representative of modern utility-scale wind turbines.

Two scripts are provided:

### `Wind_Extractor.py`

Extracts the required data from downloaded GRIB files.

Dependencies:

```bash
pip install cfgrib
```

### `Wind_Power_Converter.py`

Converts wind speed data into power output for a representative 3.2 MW wind turbine.

Outputs include:

* Mean hourly generation
* Hourly standard deviation

These values are used as Monte Carlo simulation inputs.

---

## EV Data Collection and Collation

Electric vehicle charging and driving behavior were synthesized from two datasets:

1. AAA Driving Survey
2. Caltech EV Charging Station Dataset

Because comprehensive public datasets linking driving and charging behavior are limited, these data sources were combined.

The synthesis process used:

### AAA Driving Survey

Provides:

* Seasonal driving patterns
* Hourly driving patterns

These data determine vehicle energy consumption behavior.

### Caltech Charging Dataset

Provides:

* Charging behavior
* Charging timing patterns

These data determine charging availability and charging habits.

The resulting dataset simulates hourly charging and discharging behavior for representative EVs throughout the year.

Because EV datasets vary substantially in structure and format, no generalized data-processing scripts are included for this step.

Any future adjustments will need hand collating

---

## Electric Vehicle Battery Wear

Use:

```text
Baseline_BatteryWear_Minutely.py
```

This script establishes baseline EV battery degradation.

Inputs include:

* Temperature data
* Minute-resolution charging data
* Minute-resolution driving data

The script estimates battery lifetime based on the time required to reach:

```text
80% remaining battery capacity
```

This baseline value is later compared against battery wear resulting from:

* Smart charging
* Vehicle-to-grid (V2G) operation

---

## Supply, Demand, Imports, Natural Gas, and Hydropower Collection

Data can be obtained through the EIA Hourly Grid Monitor:

https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48

Alternatively, data may be collected through the EIA API.

The following scripts are provided:

### `EIA_API_Grab.py`

Retrieves raw EIA data.

### `EIA_Data_Collator.py`

Processes and organizes the collected data.

### `Cal_Out.py`

Generates cleaned simulation inputs suitable for Monte Carlo modeling.

Outputs include hourly means and standard deviations for:

* Demand
* Generation
* Imports
* Natural gas generation
* Hydropower generation

---

## The Final Dataset

The final simulation input should be a CSV file containing hourly means and standard deviations for:

* Demand
* Actual generation
* Imports
* Solar generation sites
* Wind generation sites
* Electric vehicle behavior
* Natural gas generation
* Hydropower generation

The simulation requires valid data for all modeled hours.

In the included dataset, January 31 is omitted because some API datasets did not contain complete observations for the final hours of the year. Removing the entire day improved reproducibility.

Any data source may be used, provided:

* Time alignment is maintained
* All variables contain matching numbers of observations

---

# Running the Simulation

Once the final dataset has been created (or the included dataset is used), simulations can be performed.

Two primary workflows are available:

1. Random Sample with Pruning optimization
2. Grid build-out simulation

---

## Random Sample with Pruning Optimization

### Generating the Sample Space

The optimization process requires a predefined sample space.

Use:

```text
Generate_Sample_Space.py
```

This file contains:

* Parameter bounds
* Step sizes
* Candidate resource levels

Users should adjust these values to suit their modeling goals.

The practical size of the sample space is limited by:

* Available RAM
* Processing power
* Acceptable runtime

---

### Running the Optimization

Load the generated sample-space file by setting:

```python
File_Path =
```

within `Optimizer.py`.

The optimizer randomly samples candidate grid configurations and evaluates each through Monte Carlo simulation.

### Failure Logic

If a configuration fails any of the 10 simulation runs:

* LCOE is not calculated.
* Pruning logic is applied.

If more than two simulations fail:

* All configurations with fewer resources in every category are removed from the sample space.

### Cost-Based Pruning

If a configuration passes:

1. LCOE is calculated.
2. Pruning logic is applied.

Configurations with very high LCOE values can be eliminated to avoid unnecessary computation.

For this work, a pruning threshold of:

```text
110 $/MWh
```

was used.

---

## Cleaning the Optimized Scenarios

After optimization is complete, the best scenarios should be validated using:

```text
Verify_2050_Optimized_Case.py
```

This script runs each candidate scenario:

```text
50 times
```

and calculates:

* Mean LCOE
* Standard deviation of LCOE

Scenarios within three standard deviations of the best optimizer result should then be selected and saved to a CSV file.

These scenarios can then be analyzed using:

```text
Top_Scenario_Analysis.py
```

This script:

* Runs each scenario 50 times
* Calculates summary statistics
* Evaluates robustness

For stochastic robustness, only scenarios passing:

```text
48 / 50 simulations
```

were retained.

This corresponds approximately to a 66% probability of passing any individual 10-of-10 validation sequence.

---

## Sensitivity Analysis

Sensitivity analysis is performed using:

```text
MakeTornadoData.py
```

Create a scenario definition file similar to:

```text
TornadoRawSame.csv
```

Desired parameters can then be varied by specified percentages.

After simulation is complete, results can be visualized using:

```text
Tornado_Plots.py
```

---

## Grid Building

Two grid build-out models are included.

### `Build_Grid_6810.py`

Evaluates scenarios requiring:

* 6 of 10 passing simulations
* 8 of 10 passing simulations
* 10 of 10 passing simulations

and performs several single-point analyses.

### `Build_Grid_50Iterations.py`

Performs a more rigorous analysis by:

* Simulating every four years through 2050
* Running 50 Monte Carlo iterations per evaluation year
* Requiring 48 of 50 successful runs

Outputs include:

* LCOE statistics
* Reliability statistics
* Box-and-whisker plots for each modeled year

Resource additions are controlled within the:

```python
for year in modeled_years:
```

section of the code.

The values for:

```python
Wind_Installed +=
Solar_Installed +=
...
```

must be manually adjusted by the user to produce build-out pathways that resemble optimized scenarios.

---

# Notes

This code was developed as a research framework and is intended to be modified and extended by users. The package emphasizes transparency and flexibility over ease of use. Users are encouraged to adapt the data sources, optimization methods, and modeling assumptions to suit their specific research objectives.
