# Tetris Reinforcement Learning

This project uses reinforcement learning to train an AI to play the game of Tetris. The AI is implemented with a Deep Q-Network (DQN) and PyTorch. The primary components of the project are a training script (`train.ipynb`), a testing script (`test.ipynb`), and the game environment (`src/tetris.py`) and DQN model (`src/deep_q_network.py`) modules.

## Files Description

### 1. train.ipynb
This Jupyter notebook contains the main script for training the AI to play Tetris. It begins by initializing the Tetris game environment and the DQN model, then iterates over a number of epochs, during which it performs actions and learns from the results. It uses a replay memory to store and recall past actions and Q-values. The tracking of the model is implemented with Tensorboard.

### 2. test.ipynb
This Jupyter notebook is used to test the trained AI. It loads the trained model and runs a game of Tetris, making moves based on the AI's predictions, and records the game to a video file.

### 3. src/tetris.py
This Python file defines the game environment for Tetris. It includes functions for rotating and moving pieces, checking for collisions, clearing lines, and calculating the score.

### 4. src/deep_q_network.py
This Python file defines the Deep Q-Network (DQN) model. It includes the structure of the neural network and the forward pass function.

## Installation and Usage
The project requires PyTorch, numpy, PIL, OpenCV, and matplotlib. Use `pip install -r requirements.txt`.

Otherwise they can be installed via pip:

```bash
pip install torch numpy pillow opencv-python matplotlib
```

To train the model, run the `train.ipynb` notebook. Be aware that training can take a long time. The trained model will be saved in the `trained_models` directory.

To test the model, run the `test.ipynb` notebook. You can specify the output video file in the `Options` instance.

## Acknowledgements
This code is inspired by the following project: https://github.com/uvipen/Tetris-deep-Q-learning-pytorch
