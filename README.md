# RotaABM

**WARNING: This repository is currently under development and is NOT ready for use!**

This repository contains code for the rotavirus genetic diversity project.

## Installation

`pip install -e .`

This installs Rotasim as an importable module.

## Running the code

To run the code, import the module and call the `run` function. For example:

```python
import rotasim

sim = rotasim.Sim()
sim.run()
```

Alternatively, run the simple.py script in the tests directory which writes results to the `results` folder. File names are set based on parameter values. Run `plot_results.py` to plot key results.