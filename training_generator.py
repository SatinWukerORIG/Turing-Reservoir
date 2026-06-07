import numpy as np
import random

WRITE_0 = np.array([1, 0, 0], dtype=np.float32)
WRITE_1 = np.array([0, 1, 0], dtype=np.float32)
READ    = np.array([0, 0, 1], dtype=np.float32)

ALL_COMMANDS = [WRITE_0, WRITE_1, READ]


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
    labels.append(-1)

    for _ in range(length - 1):

        cmd = random.choice(ALL_COMMANDS)
        commands.append(cmd)

        if np.array_equal(cmd, WRITE_0):
            tape = 0
            labels.append(-1)

        elif np.array_equal(cmd, WRITE_1):
            tape = 1
            labels.append(-1)

        elif np.array_equal(cmd, READ):
            labels.append(tape)

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
                labels.append(-1)

            elif np.array_equal(cmd, WRITE_1):
                tape = 1
                labels.append(-1)

            elif np.array_equal(cmd, READ):
                labels.append(tape)

    return np.array(commands), np.array(labels)


def command_to_word(command):
    if np.array_equal(command, WRITE_0):
        return "WRITE_0"
    if np.array_equal(command, WRITE_1):
        return "WRITE_1"
    if np.array_equal(command, READ):
        return "READ"
    return "UNKNOWN"


def commands_to_words(commands):
    """Convert a sequence of one-hot command vectors to readable names."""
    return [command_to_word(command) for command in commands]


def load_data(num_sequences=1000):
    commands = []
    labels = []

    for _ in range(num_sequences):
        cmd, lbl = generate_sequence()
        commands.append(cmd)
        labels.append(lbl)

    return commands, labels

