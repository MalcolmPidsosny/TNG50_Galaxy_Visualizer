# TNG50_Galaxy_Visualizer
This algorithm is designed to visualize selected galaxies from the IllustrisTNG50 simulation.

# TNG Data Visualizer

A Python tool for visualizing the properties and spatial distribution of individual subhalos from the IllustrisTNG simulations.

The TNG Data Visualizer takes the raw particle data of a single IllustrisTNG subhalo and converts the stellar, gas, and dark matter particle distributions onto a common, uniform 3D grid. The resulting density fields can then be visualized and compared across different matter components.

## Features

* Load particle data for individual IllustrisTNG subhalos
* Process **stellar**, **gas**, and **dark matter** particle data
* Convert particle distributions onto a uniform 3D Cartesian grid
* Calculate mass densities on the grid
* Produce 2D visualizations of the resulting density distributions
* Use a consistent spatial grid for direct comparison between matter components

## Example Output

The visualizer produces density maps showing the spatial distribution of the different matter components within a selected subhalo.

<img width="1085" height="920" alt="TNG-0_gas_density_Projection_z_density" src="https://github.com/user-attachments/assets/62c21c0a-b105-49a2-9af2-9ba898493bce" />
<img width="1085" height="920" alt="TNG-0_stars_density_Projection_z_density" src="https://github.com/user-attachments/assets/6b9bc2eb-e1a5-4788-b155-2103386d14cd" />
<img width="1085" height="920" alt="TNG-0_dm_density_Projection_z_density" src="https://github.com/user-attachments/assets/cb9d7a95-1957-49ce-b403-bc409a7e93c0" />


## Requirements

* Python ≥ 3.11
* NumPy
* SciPy
* Astropy
* yt
* scikit-learn
* `illustris_python`
* tomllib

The IllustrisTNG simulation data itself is **not included** with this repository and must be obtained separately.

## Installation

Clone the repository:

```bash
git clone https://github.com/MalcolmPidsosny/tng50-data-visualizer.git
cd tng50-data-visualizer
```

Install the required dependencies:

```bash
pip install .
```

## Usage

The visualizer currently operates on a single subhalo at a time.

Specify the desired configuration file, or run the example configuration:

```bash
python -W ignore tests/test_case.py config-Example.toml
```

The program loads the relevant particle data, constructs a common uniform grid, calculates the density of each matter component, and generates the corresponding plots.

### Data Processing

For a selected subhalo, the program:

1. Loads the particle data from IllustrisTNG.
2. Extracts the spatial coordinates and masses of the relevant particles.
3. Places the particles onto a common 3D Cartesian grid.
4. Calculates the mass density within each grid cell.
5. Generates plots of the resulting density fields.

The resulting grid allows the stellar, gas, and dark matter distributions to be examined using the same spatial resolution and coordinate system.

## Matter Components

### Stars

The stellar particle distribution is converted into a 3D mass-density field on the uniform grid.

### Gas

Gas cells are mapped onto the same uniform grid and used to calculate the gas density distribution.

### Dark Matter

Dark matter particles are mapped onto the uniform grid to produce a corresponding dark matter density field.

## Units

The visualizer maintains physical units throughout the data-processing and plotting pipeline. Density fields are represented as mass per unit volume, while spatial coordinates are represented in kiloparsecs.

## Project Status

This project is currently under development.

At present, the primary functionality is the visualization of the **stellar, gas, and dark matter surface density distributions** of individual subhalos.

Additional analysis capabilities may be added in the future.

## Future Development

Potential future additions include:

* Additional particle/cell properties
* Improved visualization options
* Multiple subhalo comparison
* Radial density profiles
* Surface-density calculations
* Kinematic properties
* Galaxy morphology measurements
* Automated analysis of subhalo properties
* Additional visualization and plotting methods

## Data

This project is designed to work with data from the **IllustrisTNG** simulations.

For information about the simulations and access to the simulation data, see the official IllustrisTNG documentation.

## License

[Add your chosen license here.]

## Author

**Malcolm Pidsosny**

This project was developed as part of ongoing work with IllustrisTNG simulation data and computational astrophysics.
