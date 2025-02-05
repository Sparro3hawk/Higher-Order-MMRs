# Higher-Order-MMRs
There is now a growing sample of observed second or higher-order resonances (e.g. TOI-1136, TOI-178, TRAPPIST-1). We carried out an ambitious set of ~20,000 N-body disk migration simulations with REBOUNDx spanning a wide range of disk properties and initial planetary architectures. This repository includes the Python file used to run these simulations and the Jupyter Notebook with analysis code.

In this repository, the following pieces of information are available. 
1. Generalized_Migration_Runs_v10.py: A Python file to conduct N-body simulations with REBOUND and REBOUNDx.
2. Population_Synthesis_Analysis.ipynb: Analysis code for the ~20000 simulations run using Generalized_Migration_Runs_v10.py.
3. Exoplanet Datasets: A folder of datasets used by Generalized_Migration_Runs_v10.py and Population_Synthesis_Analysis.ipynb.
4. Exoplanet_Archive_Examiner.py: A Python file with a couple helpful functions for manipulating data from the Exoplanet Archive.
5. mr_forecast.py: The main file from the Forecaster package (Chen and Kipping 2016). 
6. fitting_parameters.h5: A file called by mr_forecast.py.
7. output_plots: A folder of plots created when running Population_Synthesis_Analysis.ipynb.
8. output_variables: A folder of CSVs and numpy variables created when running the first section of Population_Synthesis_Analysis.ipynb.
9. save: Folders of numpy arrays for the two example systems analyzed in Population_Synthesis_Analysis.ipynb.
