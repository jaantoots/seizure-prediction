# Seizure prediction using a LSTM recurrent neural network

This design implements preliminary convolutional layers to reduce the
input length. This is used as input for the recurrent layers. The
design hopes that the convolutional layers serve mainly to downsample
and extract features that can then be modelled in time series using
the recurrent layers. It is however possible that a simpler
convolutional network may ultimately perform better.

Previous uses of LSTM RNNs for time series classification:

- [Learning to Diagnose with LSTM Recurrent Neural Networks](http://arxiv.org/abs/1511.03677)

