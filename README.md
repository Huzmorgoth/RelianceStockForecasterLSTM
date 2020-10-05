# RelianceStockForecasterLSTM

# STOCK SCREENING

- Webscraping for relevant information on the stocks
- Find highest ranking stocks based on the daily trading volumes

# FORECASTING

- Forecast on RELIANCE using LSTM and other algorithms

# Algorithm (LSTM)

LSTM's are a special type of Recurrent neural networks (RNN). Recurrent neural networks (RNN) are a special type of neural network in which the output of a layer is fed back to the input layer multiple times in order to learn from the past data. Basically, the neural network is trying to learn data that follows a sequence. However, since the RNNs utilize past data, they can become computationally expensive due to storing large amouts of data in memory. The LSTM mitigates this issue, using gates. It has a cell state, and 3 gates; forget, imput and output gates.

The cell state is essentially the memory of the network. It carries information throughtout the data sequence processing. Information is added or removed from this cell state using gates. Information from the previous hidden state and current input are combined and passed through a sigmoid function at the forget gate. The sigmoid function determines which data to keep or forget. The transformed values are then multipled by the current cell state.

Next, the information from the previous hidden state combined with the input is passed through a sigmoid function to again determine important information, and also a tanh function to transform data between -1 and 1. This transformation helps with the stability of the network and helps deal with the vanishing/exploding gradient problem. These 2 outputs are multiplied together, and the output is added to the current cell state with the sigmoid function applied to it to give us our new cell state for the next time step.

Finally, the information from the hidden state combined with the current input are combined and a sigmoid function applied to it. The new cell state is passed through a tanh function to transform the values and both outputs are multiplied to determine the new hidden state for the next time step.
