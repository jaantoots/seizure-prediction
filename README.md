# Seizure prediction using a convolutional neural network

Use a convolutional neural network to classify time series data. This
convolutional network was used after an initial idea of using an RNN
failed to produce any good results.

This approach still suffers from a number of problems: no batch
normalisation and possibly suboptimal initialisation. Additionally,
the network design is _a priori_ not very promising nor scalable.
