# Order Delay Prediction with a Decision Tree

This project uses simulated orders to compare an unrestricted decision tree with a tree tuned by cross-validation. It shows how the choice of tree complexity affects which deliveries are flagged for review.

## Run locally

From the repository folder on Linux or WSL, run these commands with Python 3.12.

```bash
python -m pip install -r requirements.txt
python model.py train
python model.py predict --data data/prediction_example.csv
```

Training reads `data/training.csv` by default and writes `artifacts/model.joblib` and `artifacts/metrics.json`. Prediction writes `artifacts/predictions.csv` with the original columns and two model outputs. Use `--data path/to/file.csv` to supply your own data and `--help` after either subcommand to see its options.

Only load Joblib files you trust. Loading a Joblib file can execute code.

## Data and evaluation

Both CSVs contain simulated logistics data.

| Column | Meaning |
| --- | --- |
| `distance_km` | Delivery distance in kilometers, finite and nonnegative |
| `queue_size` | Number of waiting orders as a nonnegative integer |
| `rain` | Rain indicator, 0 or 1 |
| `weekend` | Weekend indicator, 0 or 1 |
| `delayed` | Actual delay label, 0 or 1, required only for training |

Stratified cross-validation tunes the tree on the training split. The held-out test set compares balanced accuracy, delay F1, and confusion matrices across the models.

The `delay_risk` output is the class-weighted share of delayed orders in the predicted leaf and can be used to rank orders in the demonstration. The `predicted_delay` output is the predicted class, 0 or 1. A real deployment would require checking calibration and evaluating the cost of missed delays and false alarms.

## Demonstration results

The test set contains 400 simulated orders, including 81 delays.

| Model | Balanced accuracy | Delay F1 |
| --- | ---: | ---: |
| Majority-class baseline | 0.500 | 0.000 |
| Unrestricted tree | 0.625 | 0.403 |
| Tuned tree | 0.703 | 0.495 |

The tuned tree detected 54 of the 81 delays and flagged 83 orders that arrived on time.

## Notebook

Open [analysis.ipynb](analysis.ipynb) in VS Code with the Jupyter extension, select a Python kernel with the project dependencies installed, and run all cells. The notebook shows the confusion matrix and the top levels of the tuned tree.
