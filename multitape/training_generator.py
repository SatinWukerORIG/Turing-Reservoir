import random

import numpy as np
from numpy.typing import NDArray

import train


WRITE_0 = np.array([1, 0, 0, 0, 0], dtype=np.float32)
WRITE_1 = np.array([0, 1, 0, 0, 0], dtype=np.float32)
READ    = np.array([0, 0, 1, 0, 0], dtype=np.float32)
MOVE_LEFT = np.array([0, 0, 0, 1, 0], dtype=np.float32)
MOVE_RIGHT = np.array([0, 0, 0, 0, 1], dtype=np.float32)
NOP   = np.array([0, 0, 0, 0, 0], dtype=np.float32)  # Distractor command

LABEL_EMPTY = np.array([1, 0, 0, 0, 0], dtype=np.float32)
LABEL_READ_0 = np.array([0, 1, 0, 0, 0], dtype=np.float32)
LABEL_READ_1 = np.array([0, 0, 1, 0, 0], dtype=np.float32)

map_label = {
    -1: LABEL_EMPTY,
    0: LABEL_READ_0,
    1: LABEL_READ_1
}

def predictions_to_labels(predicted_indices):
    """Convert raw prediction indices to one-hot label vectors."""

    # Model outputs use argmax over three scores, giving indices 0, 1, 2.
    predicted_indices = np.atleast_1d(predicted_indices)

    # Those indices map to label keys -1, 0, 1 in map_label
    lookup_table = np.array([map_label[-1], map_label[0], map_label[1]])

    # Initialized, untrained model tends to output numbers other than 0, 1, and 2
    # np.clip forces the indices to be inside the range
    safe_indices = np.clip(predicted_indices, 0, 2)

    return lookup_table[safe_indices]


ALL_COMMANDS = [WRITE_0, WRITE_1, READ, MOVE_LEFT, MOVE_RIGHT, NOP]


def simulate_tape_command(cmd, tape, tape_idx) -> NDArray:
    """
    Simulate a single Turing machine command on the tape and return the
    corresponding supervision label.

    Args:
        cmd: One-hot command vector from ALL_COMMANDS.
        tape: Mutable list representing the tape contents.
        tape_idx: Current position of the tape head.

    Returns:
        A one-hot label vector indicating the expected output for this step.
        For WRITE and MOVE commands the label is LABEL_EMPTY. For READ, the
        label reports the bit under the tape head as LABEL_READ_0 or LABEL_READ_1.
    """
    if np.array_equal(cmd, WRITE_0):
        tape[tape_idx] = 0
        return LABEL_EMPTY

    elif np.array_equal(cmd, WRITE_1):
        tape[tape_idx] = 1
        return LABEL_EMPTY

    elif np.array_equal(cmd, READ):
        return map_label[tape[tape_idx]]

    elif np.array_equal(cmd, MOVE_LEFT):
        tape_idx -= 1
        if tape_idx < 0:
            tape_idx = 0
            tape.insert(0, 0)  # Extend tape to the left with a blank cell

        return LABEL_EMPTY

    elif np.array_equal(cmd, MOVE_RIGHT):
        tape_idx += 1
        if tape_idx >= len(tape):
            tape.append(0)  # Extend tape to the right with a blank cell

        return LABEL_EMPTY

    elif np.array_equal(cmd, NOP):
        return LABEL_EMPTY

    return None

def generate_sequence(min_length=2, max_length=10):
    """
    Returns:
        commands: (length, 5)
        labels: (length,)
    """

    length = random.randint(min_length, max_length)

    # The lists to be returned
    commands = []
    labels = []

    # Force the first command to be a write
    first_write = random.choice([WRITE_0, WRITE_1])
    commands.append(first_write)
    labels.append(LABEL_EMPTY)

    # Initialize tape state
    tape = [0 if np.array_equal(first_write, WRITE_0) else 1]
    tape_idx = 0


    for _ in range(length - 1):

        cmd = random.choice(ALL_COMMANDS)
        commands.append(cmd)

        labels.append(simulate_tape_command(cmd, tape, tape_idx))

    # Ensure at least one READ exists
    if not any(np.array_equal(cmd, READ) for cmd in commands):

        idx = random.randint(1, length - 1)
        commands[idx] = READ

        # Recompute labels
        tape = [0 if np.array_equal(first_write, WRITE_0) else 1]
        tape_idx = 0
        labels = []

        for cmd in commands:

            labels.append(simulate_tape_command(cmd, tape, tape_idx))

    return np.array(commands), np.array(labels)


def labels_to_indices(label_vectors) -> NDArray:
    """
    Convert one-hot label vectors (LABEL_EMPTY, LABEL_READ_0, LABEL_READ_1)
    back to their readable indices: -1 (empty), 0, or 1.

    Args:
        label_vectors: array-like of shape (N, 5) or (5,) containing one-hot label vectors.

    Returns:
        numpy.ndarray of shape (N,) with values in {-1, 0, 1}.
    """
    labels_arr = np.atleast_2d(label_vectors).astype(np.float32)

    # Known label vectors in the same order as map_label keys [-1,0,1]
    known = [LABEL_EMPTY, LABEL_READ_0, LABEL_READ_1]

    result = []
    for v in labels_arr:
        # choose nearest known label by L2 distance
        dists = [np.linalg.norm(v - k) for k in known]
        k = int(np.argmin(dists))
        if k == 0:
            result.append(-1)
        else:
            result.append(k - 1)

    return np.array(result, dtype=int)


def command_to_word(command):
    if np.array_equal(command, WRITE_0):
        return "WRITE_0"
    if np.array_equal(command, WRITE_1):
        return "WRITE_1"
    if np.array_equal(command, READ):
        return "READ"
    if np.array_equal(command, NOP):
        return "NOP"
    if np.array_equal(command, MOVE_LEFT):
        return "MOVE_LEFT"
    if np.array_equal(command, MOVE_RIGHT):
        return "MOVE_RIGHT"
    return "UNKNOWN"


def commands_to_words(commands):
    """Convert a sequence of one-hot command vectors to readable names."""
    return [command_to_word(command) for command in commands]


def load_data(num_sequences=1000, min_length=2, max_length=10):
    commands = []
    labels = []

    for _ in range(num_sequences):
        cmd, lbl = generate_sequence(min_length=min_length, max_length=max_length)
        commands.append(cmd)
        labels.append(lbl)

    return commands, labels

def get_accuracy(test_commands, test_labels, reservoir, Wout, debug=False):
    correct = 0
    total = len(test_commands)

    for commands, labels in zip(test_commands, test_labels):
        readable_commands = commands_to_words(commands)
        predicted_labels = train.predict_sequence(reservoir, commands, Wout)

        predicted_labels = predictions_to_labels(predicted_labels)

        if np.array_equal(predicted_labels, labels):
            correct += 1

        else:
            if debug:
                # Convert one-hot label vectors to readable indices (-1, 0, 1) for debugging
                true_indices = labels_to_indices(labels)
                predicted_indices = labels_to_indices(predicted_labels)
                print("Commands:    ", readable_commands)
                print("True labels: ", true_indices.tolist())
                print("Predictions: ", predicted_indices.tolist())

    accuracy = correct / total
    print(f"accuracy: {correct}/{total} = {accuracy:.4f}")

