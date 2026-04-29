# LTspice Python Simulation Pipeline

A lightweight Python framework for automating LTspice simulations, collecting raw simulation data, and parsing device operating point information for analysis and visualization.

This project builds on PyLTSpice and provides a structured workflow for:

- Running parametric sweeps
- Parsing .raw and .log files
- Extracting traces, .meas results, and semiconductor operating points
- Organizing results into structured Python objects
- Plotting and analyzing simulation data

The goal is to make large simulation sweeps reproducible, scriptable, and easier to analyze from Python.

## Features
Run LTspice simulations from Python

Perform parameter sweeps programmatically

Extract:
- waveform traces
- .meas results
- device operating point parameters

Filter device parameters and traces

Collect results into structured dataclasses

Generate quick plots from simulation data

Parallel simulation support via PyLTSpice

## Descriptions

| File             | Purpose                                           |
| ---------------- | ------------------------------------------------- |
| `spice.py`       | Wrapper around PyLTspice for running simulations  |
| `callbacks.py`   | Callback used to collect `.raw` and `.log` data   |
| `parser.py`      | Converts simulation output into structured data   |
| `helpers.py`     | Utility functions for printing stats and plotting |
| `runconfig.py`   | Central configuration for simulations             |
| `script.py`      | Example pipeline execution                        |
| `model_file.txt` | Example MOSFET model                              |

The simulation workflow follows four main stages.

### 1. Run Simulation

A Spice object wraps the PyLTspice simulation runner and does the following:
- creates the .net file
- launches LTspice
- handles parameter updates
- executes simulations

```
spice = Spice(
    exe_path,
    asc_path,
    output_folder,
    callback_proc=CallbackGetAllData
)
```

### 2. Callback Data Collection

Each simulation run returns a structured result using a callback.

The callback loads:
- .raw waveform data
- .log measurements
- semiconductor operating point data

and returns a SimResult object.

```
class SimResult:
    file_id
    raw_data
    log_data
    semi_ops
```

### 3. Parsing Simulation Data

The Parser converts SimResult objects into ParsedSimResult objects containing:
- traces
- .meas values
- device parameters

```
parser.parse(
    results,
    trace_names=["v(vy)", "Id(M7)"],
    device_names=["M6"],
    device_values=["Model", "Id", "Vgs", "Vdsat"]
)
```

### 4. Analysis

Utility functions help summarize and visualize results.
```
Helpers.print_stats(results)
Helpers.plot_op_points(results, "V(vx)", "Id(M7)")
```
## Example Usage

Example parameter sweep:
```
values = ["0.3u", "0.5u", "1.0u"]

for v in values:
    spice.set_parameter("l", v)
    spice.simulate()
```

After simulations complete:

```
raw_results = list(spice.sim_runner)

parsed_results = list(parser.parse(
    raw_results,
    config.trace_names,
    config.meas_names,
    config.device_names,
    config.device_values
))
```

## Extracting MOSFET Operating Point Data

MOSFET operating point parameters can also be extracted from the LTspice `.log` file.  
These values are parsed automatically and stored in the `semi_ops` field of each
`ParsedSimResult`.

The devices and parameters to extract are configured in `RunConfig`:

```python
self.device_names = ["M6"]
self.device_values = ["Model", "Id", "Vgs", "Vdsat"]```
After parsing, the data can be accessed directly from the results:

```python
for r in parsed_results:
    mos = r.semi_ops["MOSFETS"]["M6"]
    print(mos["Id"], mos["Vgs"], mos["Vdsat"])
```



## Configuration

Simulation settings are controlled through RunConfig.

Example parameters:

```
self.trace_names = ["v(vy)", "Id(M7)"]
self.device_names = ["M6"]
self.device_values = ["Model", "Id", "Vgs", "Vdsat"]
```

## Requirements

Python 3.10+
LTspice
PyLTSpice
NumPy
Matplotlib

Install dependencies:

```pip install numpy matplotlib PyLTSpice```

## Motivation

LTspice is powerful but difficult to automate for large design-space exploration tasks. This project provides a clean Python interface for:
- analog design sweeps
- device characterization
- operating point extraction
- automated analysis pipelines

## Future Improvements

Potential extensions include:
- gm/Id curve extraction
- automated device characterization
- Monte Carlo sweep support
- multi-parameter sweeps
- better plotting utilities
- exporting results to Pandas / CSV
