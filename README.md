# Federated Learning with Flowe in Python

Flower is an open-source framework for federated learning (FL) designed with a client-server (hub-and-spoke) topology: a single server coordinates training across many distributed clients. 
More on Flower [here](https://flower.ai).

In this implementation, a **fixed set** of federated clients train their local neural network on their own dataset, satisfying **data locality** principles.
A **server** aggregates new model weights, following the **Federated Averaging** algorithm, for a total of 10 rounds.
