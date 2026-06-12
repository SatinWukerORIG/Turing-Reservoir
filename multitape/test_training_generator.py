import unittest
import numpy as np
import training_generator

COMMAND_WORDS = {
    "WRITE_0": training_generator.WRITE_0,
    "WRITE_1": training_generator.WRITE_1,
    "READ": training_generator.READ,
    "MOVE_LEFT": training_generator.MOVE_LEFT,
    "MOVE_RIGHT": training_generator.MOVE_RIGHT,
    "NOP": training_generator.NOP,
}


def words_to_commands(command_words):
    return np.array([COMMAND_WORDS[word] for word in command_words], dtype=np.float32)


class TestTrainingGenerator(unittest.TestCase):
    def assert_commands_produce_labels(self, command_words, expected_indices):
        commands = words_to_commands(command_words)
        tape = [0]
        tape_idx = 0
        actual_indices = []

        for command in commands:
            label, tape_idx = training_generator.simulate_tape_command(command, tape, tape_idx)
            actual_indices.append(int(training_generator.labels_to_indices(label)[0]))

        self.assertEqual(actual_indices, expected_indices)

    def test_user_example_sequence(self):
        self.assert_commands_produce_labels(
            [
                "WRITE_1",
                "MOVE_LEFT",
                "WRITE_1",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
                "MOVE_RIGHT",
                "NOP",
                "READ",
                "MOVE_RIGHT",
                "MOVE_LEFT",
            ],
            [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1],
        )

    def test_write_then_read_same_cell(self):
        self.assert_commands_produce_labels(
            ["WRITE_0", "MOVE_RIGHT", "WRITE_1", "MOVE_LEFT", "READ"],
            [-1, -1, -1, -1, 0],
        )

    def test_left_edge_extend_then_read(self):
        self.assert_commands_produce_labels(
            ["MOVE_LEFT", "READ"],
            [-1, 0],
        )


if __name__ == "__main__":
    unittest.main()
