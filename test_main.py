import unittest
import numpy as np
from main import RNN, Hyperparameters, Gradients, Trainer, softmax


class TestRNN(unittest.TestCase):
    def setUp(self):
        self.hp = Hyperparameters(vocab_size=10, hidden_size=20, seq_length=5, learning_rate=0.1)
        self.rnn = RNN(self.hp)

    def test_softmax(self):
        x = np.array([1, 2, 3])
        s = softmax(x)
        self.assertAlmostEqual(np.sum(s), 1.0)
        self.assertTrue(np.all(s >= 0))

    def test_forward_pass(self):
        inputs = [0, 1, 2, 3, 4]
        targets = [1, 2, 3, 4, 5]
        hprev = np.zeros((self.hp.hidden_size, 1))
        loss, _, _, _, _ = self.rnn._forward_pass(inputs, targets, hprev)
        self.assertIsInstance(loss, float)

    def test_backward_pass(self):
        inputs = [0, 1, 2, 3, 4]
        targets = [1, 2, 3, 4, 5]
        hprev = np.zeros((self.hp.hidden_size, 1))
        _, xs, hs, _, ps = self.rnn._forward_pass(inputs, targets, hprev)
        grads = self.rnn._backward_pass(targets, xs, hs, ps)
        self.assertIsInstance(grads, Gradients)
        self.assertEqual(grads.dWxh.shape, self.rnn.Wxh.shape)

    def test_update(self):
        grads = Gradients(
            dWxh=np.random.randn(*self.rnn.Wxh.shape),
            dWhh=np.random.randn(*self.rnn.Whh.shape),
            dWhy=np.random.randn(*self.rnn.Why.shape),
            dbh=np.random.randn(*self.rnn.bh.shape),
            dby=np.random.randn(*self.rnn.by.shape),
        )
        wxh_before = self.rnn.Wxh.copy()
        self.rnn.update(grads)
        self.assertFalse(np.all(wxh_before == self.rnn.Wxh))


if __name__ == "__main__":
    unittest.main()
