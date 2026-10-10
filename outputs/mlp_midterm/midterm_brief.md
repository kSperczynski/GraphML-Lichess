# MLP baseline: midterm brief

## Research question
How much rating information can be recovered from a player's own aggregate features, without using graph edges?

## Method
The pilot uses 7,571 players and 14 numeric features. Player IDs and target ratings are excluded from inputs. A fixed random player split contains 5,299 training, 1,136 validation, and 1,136 test players. Infinite feature values are treated as missing and filled with training-set medians. Feature and target scalers are fitted only on the training set.

The MLP has hidden dimensions 64 and 32, ReLU activations, dropout 0.1, and a linear output. It uses MSE loss and Adam with learning rate 0.001, up to 200 epochs, and early-stopping patience 20. Each run restores the checkpoint with minimum validation MSE. Training seeds are 42, 43, and 44. Baselines are the training-set mean rating and Ridge regression with alpha 1.0.

## Results
| Split | Model | MAE (rating points) | RMSE (rating points) |
|---|---|---:|---:|
| validation | Training mean | 240.20 | 303.90 |
| validation | Ridge | 153.12 | 193.83 |
| validation | MLP | 145.82 +/- 0.08 | 183.81 +/- 0.34 |
| test | Training mean | 236.45 | 296.53 |
| test | Ridge | 155.65 | 193.48 |
| test | MLP | 149.51 +/- 1.24 | 186.38 +/- 1.26 |

MLP values are mean +/- sample standard deviation across three independent training seeds on one fixed player split. They describe training variability, not uncertainty across players or data splits. Baselines are deterministic and evaluated once.

## Interpretation
On the held-out test set, MLP has a mean RMSE of 186.38 rating points, compared with 193.48 for Ridge and 296.53 for the training-mean predictor. This is a 3.67% RMSE reduction relative to Ridge and a 37.15% reduction relative to the mean predictor. The pilot shows predictive information in player features and a modest additional benefit from the nonlinear model. The comparison does not establish a graph-model advantage or an improvement over Glicko-2. A fair GNN comparison still requires matching players, labels, available information, preprocessing, and model-selection rules.

## Limitations
This pilot estimates the rating recorded for a player's latest sampled game from aggregate features. It is not yet an experiment using the first k games to predict a future converged rating. Some features include lifetime statistics and the sampled windows are not aligned to a common prediction cutoff. Random player splitting does not address these temporal issues. The rating is a proxy target rather than verified true skill. Sampling players with analyzed games can introduce selection bias. Graph models and Glicko-2 have not been evaluated in this notebook.

## Next steps
Agree on a shared temporal input window and label definition, reproduce the baselines on that dataset, compare against teammates' graph models, and join spectral subgroup labels to saved player predictions. Complete the assigned literature contributions for the final paper.

## Short presentation script
I implemented a non-graph MLP baseline to measure how much rating information is available in player features alone. We use training-only preprocessing, a shared player split, and validation-based checkpoint selection. We compare against a constant mean predictor and Ridge regression, and repeat MLP training with three seeds. The current results are a pilot on aggregate features; our next step is to align the temporal task and compare the graph models under the same protocol.
