# Turing Reservoir

This repository contains a simple reservoir computing setup for a command/label prediction task.

## Files

- `main.py`: Training script.
- `test.py`: Testing script.
- `training_generator.py`: Data generation module and helper functions.
- `saved_reservoir/`: Directory where trained weight files are stored.

## Overview

- `main.py` trains a reservoir model using generated sequences of commands.
- `test.py` loads the trained reservoir and readout weights, generates new test data, and evaluates model accuracy.
- `training_generator.py` creates command sequences, label sequences, and includes helper utilities for debugging and accuracy measurement.

## What `main.py` does

`main.py` is the training entry point. It:

1. Generates training data with `training_generator.load_data()`.
2. Initializes a `Reservoir` with random input and recurrent weights.
3. Runs training sequences through the reservoir to collect reservoir states.
4. Trains a linear readout weight matrix `Wout` using least squares regression.
5. Computes training accuracy via `training_generator.get_accuracy()`.
6. Saves the trained weights to `saved_reservoir/`:
   - `Win.npy`
   - `W.npy`
   - `Wout.npy`

## What `test.py` does

`test.py` is the evaluation entry point. It:

1. Generates new test data with `training_generator.load_data()`.
2. Loads the saved reservoir weights from `saved_reservoir/`.
3. Uses the loaded reservoir and readout weights to run the test sequences.
4. Computes and prints accuracy using `training_generator.get_accuracy()`.

When `debug=True`, `test.py` also prints sequences where the model prediction does not exactly match the true labels. This helps identify failure cases.

## What `training_generator.py` does

`training_generator.py` is the data generator and helper module. It provides:

- `generate_sequence()`: construct random command sequences with writes, reads, and no-ops.
- `load_data()`: generate many sequences for training or testing.
- `command_to_word()` / `commands_to_words()`: convert commands to readable names.
- `predictions_to_labels()`: convert predicted label indices into one-hot label vectors.
- `get_accuracy()`: run sequences through the reservoir and compute overall accuracy, with optional debug output.

It also defines the command and label encodings used by the model.

## Running

1. Install dependencies:

```bash
pip install numpy
```

2. Train the model:

```bash
python main.py
```

3. Test the model:

```bash
python test.py
```

## Notes

- The model uses reservoir computing with a fixed recurrent layer and a trained readout.
- `main.py` saves weights so `test.py` can evaluate the same trained model.
- `training_generator.py` ensures each sequence contains at least one `READ` command and computes labels based on the current tape value.
