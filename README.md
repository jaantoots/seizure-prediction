# Seizure prediction using a LSTM recurrent neural network

Previous uses of LSTM RNNs for time series classification:

- [Learning to Diagnose with LSTM Recurrent Neural Networks](http://arxiv.org/abs/1511.03677)

## Architecture

- Input, $\mathbf{x}^{(t)}$
- $N$ layers of LSTM RNN of 128 inputs
- Fully connected layer
- Output, $\mathbf{y}^{(t)}$
- Softmax loss applied at each time step
- Prediction from average, last, or weighted average
