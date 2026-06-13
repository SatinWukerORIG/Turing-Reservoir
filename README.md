# Turing Reservoir

This repository contains reservoir computing experiments for simulating Turing machines. Two variants explore different levels of complexity:

- **single_tape/**: A basic single-tape Turing machine with WRITE_0, WRITE_1, and READ commands.
- **multitape/**: An extended multi-tape Turing machine with MOVE_LEFT and MOVE_RIGHT commands for tape head navigation.

## Directory Structure

```
.
├── single_tape/
│   ├── train.py
│   ├── test.py
│   ├── training_generator.py
│   └── saved_reservoir/          # Trained weights
├── multitape/
│   ├── train.py
│   ├── test.py
│   ├── test_training_generator.py
│   ├── training_generator.py
│   └── saved_reservoir/          # Trained weights
├── LICENSE
└── README.md
```

## Common Pattern

Both `single_tape/` and `multitape/` follow the same workflow:

### `train.py`

Training entry point. It:

1. Generates training data with `training_generator.load_data()`.
2. Initializes a `Reservoir` with random input and recurrent weights.
3. Optionally loads pre-trained weights if `TRAIN_FROM_SAVED_WEIGHTS` is True.
4. Runs command sequences through the reservoir to collect states.
5. Trains a linear readout weight matrix `Wout` using least squares regression.
6. Computes training accuracy.
7. Saves trained weights to `saved_reservoir/`:
   - `Win.npy` (input-to-reservoir weights)
   - `W.npy` (recurrent weights)
   - `Wout.npy` (readout weights)

### `test.py`

Testing entry point. It:

1. Generates new test data with `training_generator.load_data()`.
2. Loads saved reservoir and readout weights.
3. Evaluates accuracy on test data using `training_generator.get_accuracy()`.
4. When `debug=True`, prints sequences where predictions don't match true labels.

### `training_generator.py`

Data generation and helper module providing:

- **Single-tape version:**
  - `generate_sequence()`: Create random command sequences (WRITE_0, WRITE_1, READ, NOP).
  - `load_data()`: Generate multiple training/test sequences.
  - `get_accuracy()`: Evaluate accuracy with optional debug output.

- **Multi-tape version:**
  - `simulate_tape_command()`: Simulate Turing machine operations on a tape with head position tracking.
  - `generate_sequence()`: Create random command sequences (WRITE_0, WRITE_1, READ, MOVE_LEFT, MOVE_RIGHT).
  - `load_data()`: Generate multiple training/test sequences.
  - `get_accuracy()`: Evaluate accuracy with optional debug output.

## Single-Tape Variant

Commands are 3-dimensional:
- `WRITE_0`: Write 0 to the tape
- `WRITE_1`: Write 1 to the tape
- `READ`: Read the current tape value

The tape is a single value (0 or 1) that can be updated by WRITE commands. READ outputs the current value.

## Multi-Tape Variant

Commands are 5-dimensional:
- `WRITE_0`: Write 0 to current tape position
- `WRITE_1`: Write 1 to current tape position
- `READ`: Read the current tape position
- `MOVE_LEFT`: Move tape head left (extends tape if needed)
- `MOVE_RIGHT`: Move tape head right (extends tape if needed)

The tape is a list of bits with a head pointer. MOVE commands can extend the tape dynamically.

## Running

1. Install dependencies:

```bash
pip install numpy
```

2. Train the model (choose variant):

```bash
cd single_tape
python train.py
```

or

```bash
cd multitape
python train.py
```

3. Test the model:

```bash
python test.py
```

Set `debug=True` in `test.py` to see failure cases.

## Technical Details

- **Reservoir computing**: The recurrent layer is randomly initialized and fixed. Only the readout weights are trained.
- **Spectral radius**: Recurrent weights are scaled to control stability (0.9 for single-tape, 0.95 for multi-tape).
- **Training**: Linear readout weights are solved using least squares regression on collected reservoir states.
- **Supervision**: Labels are one-hot vectors indicating the output of READ operations or "empty" for write/move operations.
