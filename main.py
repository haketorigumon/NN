#!/usr/bin/env python3
"""
This script implements a character-level Recurrent Neural Network (RNN) from scratch
using NumPy. It can be used to train a model on a text file and generate text in the
style of the training data.
"""

import argparse
from dataclasses import dataclass
import numpy as np


@dataclass
class Hyperparameters:
    """
    Stores the hyperparameters for the RNN model.

    Attributes:
        vocab_size: The number of unique characters in the training data.
        hidden_size: The number of neurons in the hidden layer.
        seq_length: The number of time steps to unroll the RNN for.
        learning_rate: The learning rate for the Adagrad optimizer.
    """
    vocab_size: int
    hidden_size: int = 100
    seq_length: int = 25
    learning_rate: float = 1e-1


@dataclass
class Gradients:
    """
    Stores the gradients for the RNN model parameters.

    Attributes:
        dWxh: Gradient for the input-to-hidden weight matrix.
        dWhh: Gradient for the hidden-to-hidden weight matrix.
        dWhy: Gradient for the hidden-to-output weight matrix.
        dbh: Gradient for the hidden bias vector.
        dby: Gradient for the output bias vector.
    """
    dWxh: np.ndarray
    dWhh: np.ndarray
    dWhy: np.ndarray
    dbh: np.ndarray
    dby: np.ndarray


@dataclass
class LossFunOutput:
    """
    Stores the output of the loss function.

    Attributes:
        loss: The calculated loss value.
        grads: The gradients of the model parameters.
        hprev: The last hidden state.
    """
    loss: float
    grads: Gradients
    hprev: np.ndarray


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Computes the numerically stable softmax of a vector.

    Args:
        x: A NumPy array.

    Returns:
        A NumPy array with the softmax applied.
    """
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


class RNN:
    """
    A character-level Recurrent Neural Network (RNN) model.
    """
    def __init__(self, hp: Hyperparameters):
        """
        Initializes the RNN model.

        Args:
            hp: An object containing the model's hyperparameters.
        """
        self.hp = hp

        # Model parameters
        self.Wxh = np.random.randn(hp.hidden_size, hp.vocab_size) * 0.01
        self.Whh = np.random.randn(hp.hidden_size, hp.hidden_size) * 0.01
        self.Why = np.random.randn(hp.vocab_size, hp.hidden_size) * 0.01
        self.bh = np.zeros((hp.hidden_size, 1))
        self.by = np.zeros((hp.vocab_size, 1))

        # Adagrad memory
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)

    def _forward_pass(
        self, inputs: list[int], targets: list[int], hprev: np.ndarray
    ) -> tuple[float, dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
        """
        Performs the forward pass of the RNN.

        Args:
            inputs: A list of character indices.
            targets: A list of target character indices.
            hprev: The previous hidden state.

        Returns:
            A tuple containing the loss, and dictionaries for the inputs,
            hidden states, outputs, and probabilities.
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0.0
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.hp.vocab_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t - 1] + self.bh)
            ys[t] = self.Why @ hs[t] + self.by
            ps[t] = softmax(ys[t])
            loss += -np.log(ps[t][targets[t], 0] + 1e-12)
        return loss, xs, hs, ys, ps

    def _backward_pass(self, targets: list[int], xs: dict[int, np.ndarray], hs: dict[int, np.ndarray], ps: dict[int, np.ndarray]) -> Gradients:
        """
        Performs the backward pass of the RNN.

        Args:
            targets: A list of target character indices.
            xs: A dictionary of one-hot encoded inputs.
            hs: A dictionary of hidden states.
            ps: A dictionary of probabilities.

        Returns:
            An object containing the gradients of the model parameters.
        """
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(targets))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += dy @ hs[t].T
            dby += dy
            dh = self.Why.T @ dy + dhnext
            dhraw = (1 - hs[t] ** 2) * dh
            dbh += dhraw
            dWxh += dhraw @ xs[t].T
            dWhh += dhraw @ hs[t - 1].T
            dhnext = self.Whh.T @ dhraw
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        return Gradients(dWxh=dWxh, dWhh=dWhh, dWhy=dWhy, dbh=dbh, dby=dby)

    def lossFun(
        self,
        inputs: list[int],
        targets: list[int],
        hprev: np.ndarray,
    ) -> LossFunOutput:
        """
        Calculates the loss and gradients for a sequence of characters.

        Args:
            inputs: A list of character indices.
            targets: A list of target character indices.
            hprev: The previous hidden state.

        Returns:
            An object containing the loss, gradients, and the last hidden state.
        """
        loss, xs, hs, ys, ps = self._forward_pass(inputs, targets, hprev)
        grads = self._backward_pass(targets, xs, hs, ps)
        return LossFunOutput(loss=loss, grads=grads, hprev=hs[len(inputs) - 1])

    def sample(self, h: np.ndarray, seed_ix: int, n: int) -> list[int]:
        """
        Samples a sequence of characters from the model.

        Args:
            h: The initial hidden state.
            seed_ix: The index of the first character.
            n: The number of characters to sample.

        Returns:
            A list of sampled character indices.
        """
        x = np.zeros((self.hp.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []

        for _ in range(n):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            p = softmax(y)
            ix = np.random.choice(range(self.hp.vocab_size), p=p.ravel())
            x = np.zeros((self.hp.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)

        return ixes

    def update(self, grads: Gradients) -> None:
        """
        Updates the model parameters using the Adagrad optimization algorithm.

        Args:
            grads: An object containing the gradients of the model parameters.
        """
        for param, dparam, mem in zip(
            [self.Wxh, self.Whh, self.Why, self.bh, self.by],
            [grads.dWxh, grads.dWhh, grads.dWhy, grads.dbh, grads.dby],
            [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby],
        ):
            mem += dparam * dparam
            param += -self.hp.learning_rate * dparam / np.sqrt(mem + 1e-8)


class Trainer:
    """
    A class to handle the training of the RNN model.
    """
    def __init__(self, rnn: RNN, data: str, char_to_ix: dict[str, int], ix_to_char: dict[int, str]):
        """
        Initializes the Trainer.

        Args:
            rnn: The RNN model to train.
            data: The training data.
            char_to_ix: A dictionary mapping characters to indices.
            ix_to_char: A dictionary mapping indices to characters.
        """
        self.rnn = rnn
        self.data = data
        self.char_to_ix = char_to_ix
        self.ix_to_char = ix_to_char

    def train(self, num_iterations: int):
        """
        Trains the RNN model.

        Args:
            num_iterations: The number of training iterations.
        """
        p = 0
        smooth_loss = -np.log(1.0 / self.rnn.hp.vocab_size) * self.rnn.hp.seq_length
        hprev = np.zeros((self.rnn.hp.hidden_size, 1))

        for n in range(num_iterations):
            if p + self.rnn.hp.seq_length + 1 >= len(self.data):
                hprev = np.zeros((self.rnn.hp.hidden_size, 1))
                p = 0

            inputs = [self.char_to_ix[ch] for ch in self.data[p: p + self.rnn.hp.seq_length]]
            targets = [self.char_to_ix[ch] for ch in self.data[p + 1: p + self.rnn.hp.seq_length + 1]]

            if n % 100 == 0:
                sample_ix = self.rnn.sample(hprev, inputs[0], 200)
                txt = "".join(self.ix_to_char[ix] for ix in sample_ix)
                print(f"----\n{txt}\n----")

            loss_output = self.rnn.lossFun(inputs, targets, hprev)
            self.rnn.update(loss_output.grads)

            smooth_loss = smooth_loss * 0.999 + loss_output.loss * 0.001
            if n % 100 == 0:
                print(f"iter {n}, loss: {smooth_loss:.4f}")

            hprev = loss_output.hprev
            p += self.rnn.hp.seq_length


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a character-level RNN on a text file.")
    parser.add_argument("input_file", help="The input text file for training.")
    parser.add_argument("--num_iterations", type=int, default=100000, help="The number of training iterations.")
    parser.add_argument("--hidden_size", type=int, default=100, help="The size of the hidden layer.")
    parser.add_argument("--seq_length", type=int, default=25, help="The sequence length.")
    parser.add_argument("--learning_rate", type=float, default=1e-1, help="The learning rate.")
    return parser.parse_args()


def main() -> None:
    """
    The main function of the script.
    """
    args = parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = f.read()

    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print(f"Data has {data_size} characters, {vocab_size} unique.")

    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    hp = Hyperparameters(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        seq_length=args.seq_length,
        learning_rate=args.learning_rate,
    )
    rnn = RNN(hp)
    trainer = Trainer(rnn, data, char_to_ix, ix_to_char)
    trainer.train(args.num_iterations)


if __name__ == "__main__":
    main()
