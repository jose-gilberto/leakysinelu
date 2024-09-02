

# Semi-Periodic Activation for Time Series Classification

This repository maintain the source code used to produce all the experiments, results and figures of the paper approved on BRACIS 24.

## Authors

- José Gilberto Barbosa de Medeiros Júnior (gilberto.barbosa@usp.br)
- André Guarnier de Mitri (andremitri@usp.br)
- Diego Furtado Silva (diegofsilva@icmc.usp.br)

## Abstract

This paper investigates the lack of research on activation functions for neural network models in time series tasks. It highlights the need to identify essential properties of these activations to improve their effectiveness in specific domains. To this end, the study comprehensively analyzes properties, such as bounded, monotonic, nonlinearity, and periodicity, for activation in time series neural networks. We propose a new activation that maximizes the coverage of these properties, called LeakySineLU. We empirically evaluate the LeakySineLU against commonly used activations in the literature using 112 benchmark datasets for time series classification, obtaining the best average ranking in all comparative scenarios. 

## Requirements

```
torch == 2.0.1
numpy == 1.24.3
pandas == 1.5.3
matplotlib == 3.7.2
aeon == 0.8.1
lightning == 2.2.1
```

## Folder Structure

```
|- notebooks/ # notebooks used to produce figures
|- experiments/ # all the scripts to run the experiments
    |- fcn_ucr.py
    |- mlp_ucr.py
    |- utils.py
```

- `fcn_ucr.py` Runs all experiments over the 112 UCR equal-length datasets (all datasets without Vary as length on https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) using a FCN.
- `mlp_ucr.py` Do the same for the MLP network.
- `utils.py` Contains all the utility functions to load the data and prepare the Torch-like structures.

## Future Works

- Study the impact of dead neurons;
- Upgrade the experiments to include ResNet, InceptionTime and LITE;
- Propose a new LeakySineLU based on the PReLU, with adaptative weights on the negative slope.
