import main
import numpy as np
import random

WRITE_0 = np.array([1, 0, 0], dtype=np.float32)
WRITE_1 = np.array([0, 1, 0], dtype=np.float32)
READ    = np.array([0, 0, 1], dtype=np.float32)
NOP   = np.array([0, 0, 0], dtype=np.float32)  # Distractor command

LABEL_EMPTY = np.array([1, 0, 0], dtype=np.float32)
LABEL_READ_0 = np.array([0, 1, 0], dtype=np.float32)
LABEL_READ_1 = np.array([0, 0, 1], dtype=np.float32)

map_label = {
    -1: LABEL_EMPTY,
    0: LABEL_READ_0,
    1: LABEL_READ_1
}

def predictions_to_labels(predicted_indices):
    """Convert raw prediction indices to one-hot label vectors."""

    # Model outputs use argmax over three scores, giving indices 0, 1, 2.
    # Those indices map to label keys -1, 0, 1 in map_label, so subtract 1.
    predicted_indices = np.atleast_1d(predicted_indices)
    return np.array([map_label[int(idx) - 1] for idx in predicted_indices])


ALL_COMMANDS = [WRITE_0, WRITE_1, READ, NOP]


def generate_sequence(min_length=2, max_length=10):
    """
    Returns:
        commands: (length, 3)
        labels: (length,)
    """

    length = random.randint(min_length, max_length)

    commands = []

    # Initial tape state
    tape = 0

    labels = []

    # Force the first command to be a write
    first_write = random.choice([WRITE_0, WRITE_1])
    commands.append(first_write)

    tape = 0 if np.array_equal(first_write, WRITE_0) else 1
    labels.append(LABEL_EMPTY)

    for _ in range(length - 1):

        cmd = random.choice(ALL_COMMANDS)
        commands.append(cmd)

        if np.array_equal(cmd, WRITE_0):
            tape = 0
            labels.append(LABEL_EMPTY)

        elif np.array_equal(cmd, WRITE_1):
            tape = 1
            labels.append(LABEL_EMPTY)

        elif np.array_equal(cmd, READ):
            labels.append(map_label[tape])
        
        elif np.array_equal(cmd, NOP):
            labels.append(LABEL_EMPTY)

    # Ensure at least one READ exists
    if not any(np.array_equal(cmd, READ) for cmd in commands):

        idx = random.randint(1, length - 1)
        commands[idx] = READ

        # Recompute labels
        tape = 0
        labels = []

        for cmd in commands:

            if np.array_equal(cmd, WRITE_0):
                tape = 0
                labels.append(LABEL_EMPTY)

            elif np.array_equal(cmd, WRITE_1):
                tape = 1
                labels.append(LABEL_EMPTY)

            elif np.array_equal(cmd, READ):
                labels.append(map_label[tape])

            elif np.array_equal(cmd, NOP):
                labels.append(LABEL_EMPTY)

    return np.array(commands), np.array(labels)


def command_to_word(command):
    if np.array_equal(command, WRITE_0):
        return "WRITE_0"
    if np.array_equal(command, WRITE_1):
        return "WRITE_1"
    if np.array_equal(command, READ):
        return "READ"
    if np.array_equal(command, NOP):
        return "NOP"
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
        predicted_labels = main.predict_sequence(reservoir, commands, Wout)

        predicted_labels = predictions_to_labels(predicted_labels)

        if np.array_equal(predicted_labels, labels):
            correct += 1

        else:
            if debug:
                print("Commands:    ", readable_commands)
                print("True labels: ", labels.tolist())
                print("Predictions: ", predicted_labels.tolist())

    accuracy = correct / total
    print(f"accuracy: {correct}/{total} = {accuracy:.4f}")

