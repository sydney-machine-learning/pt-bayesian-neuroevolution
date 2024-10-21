# Bayesian neuroevolution using distributed swarm optimisation and tempered MCMC
A synergy of Neuro-evolution and Bayesian neural learning by using Particle Swarm Optimization for forming efficient proposals in parallel tempering MCMC. The architecture is implemented in a parallel computing environment for improving computational efficiency.

## Requirements
This project was tested on python 3.8 and the required packages can be installed using the provided `requirements.txt`:

```
    pip install -r requirements.txt
```

## Running Evolutionary Parallel Tempering
```
    cd pt_pso
    python pt_pso_mcmc_pop.py --problem <problem-name>
```

The script supports following additional command line arguments:
```
usage: evo_pt_mcmc.py [-h] [--problem PROBLEM] [--num-chains NUM_CHAINS]
                      [--population-size POPULATION_SIZE]
                      [--num-samples NUM_SAMPLES]
                      [--swap-interval SWAP_INTERVAL] [--burn-in BURN_IN]
                      [--max-temp MAX_TEMP] [--topology TOPOLOGY]
                      [--run-id RUN_ID] [--root ROOT]
                      [--train-data TRAIN_DATA] [--test-data TEST_DATA]
                      [--config-file CONFIG_FILE]

Run Evolutionary Parallel Tempering

optional arguments:
  -h, --help            show this help message and exit
  --problem PROBLEM     Problem to be used for Evolutionary PT: "synthetic",
                        "iris", "ions", "cancer", "bank", "penDigit", "chess", "TicTacToe"
  --num-chains NUM_CHAINS
                        Number of Chains for Parallel Tempering
  --population-size POPULATION_SIZE
                        Population size for G3PCX Evolutionary algorithm.
  --num-samples NUM_SAMPLES
                        Total number of samples (all chains combined).
  --swap-interval SWAP_INTERVAL
                        Number of samples between each swap.
  --burn-in BURN_IN     Ratio of samples to be discarded as burn-in samples.
                        Value 0.1 means 10 percent samples are discarded
  --max-temp MAX_TEMP   Temperature to be assigned to the chain with maximum
                        temperature.
  --topology TOPOLOGY   String representation of network topology. Eg:-
                        "input,hidden,output"
  --run-id RUN_ID       Unique Id to identify run.
  --root ROOT           path to root directory (evolutionary-pt).
  --train-data TRAIN_DATA
                        Path to the train data
  --test-data TEST_DATA
                        Path to the test data
  --config-file CONFIG_FILE
                        Path to data config yaml file
```

*Note: The default values of these values can be changed from config.py and data.yaml file*

## DataSets - Classification
The approach is tested on the following datasets:
1. [Iris](https://archive.ics.uci.edu/ml/datasets/iris)
2. [Ionosphere](https://archive.ics.uci.edu/ml/datasets/ionosphere)
3. [Breast Cancer](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29)
4. [Pen Digit](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits)

## Cite this work
```
@article{kapoor2022bayesian,
  title={Bayesian neuroevolution using distributed swarm optimization and tempered MCMC},
  author={Kapoor, Arpit and Nukala, Eshwar and Chandra, Rohitash},
  journal={Applied Soft Computing},
  volume={129},
  pages={109528},
  year={2022},
  publisher={Elsevier}
}
```
