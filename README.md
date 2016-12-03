# Seizure prediction using a convolutional neural network

Use a convolutional neural network to classify time series data. This
convolutional network was used after an initial idea of using an RNN
failed to produce any good results.

This approach still suffers from a number of problems: no batch
normalisation and possibly suboptimal initialisation. Additionally,
the network design is _a priori_ not very promising nor scalable.

---

The results after training turned out to be quite underwhelming. Area
under ROC in validation reached above 0.8 after approximately 1000
iterations using a batch size of 20 on a Tesla K80 but the test data
scores from the leader board were much lower than expected. Possibly
due to careless validation split.

Overall, did not have time to work on this enough.
