# Flappy Bird Reinforcement Learning with DQN

This project is a simple implementation of the **Flappy Bird game** using **Deep Q-Networks (DQN)** for reinforcement learning. The goal is to train an AI agent to play Flappy Bird using Q-learning and improve its performance over time.

## Project Structure

The project is organized as follows:

flappy-bird-pygame/ │ ├── dqn_model.py # Defines the DQN model used for Q-learning ├── flappy_bird.py # Defines the game environment ├── train.py # Main script to train the model using reinforcement learning ├── requirements.txt # List of Python dependencies ├── flappy_bird_dqn.pth # Saved model after training (if available) ├── README.md # This file

bash
Sao chép
Chỉnh sửa

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/hieuchaydii/flappy-bird-pygame.git
cd flappy-bird-pygame
2. Create a virtual environment (optional but recommended)
bash
Sao chép
Chỉnh sửa
python -m venv myenv
source myenv/bin/activate   # For Linux/macOS
myenv\Scripts\activate      # For Windows
3. Install dependencies
Install the required Python libraries by running:

bash
Sao chép
Chỉnh sửa
pip install -r requirements.txt
4. Dependencies
This project requires the following Python packages:

pygame: For creating the Flappy Bird game environment.

torch: For building and training the neural network model.

numpy: For numerical operations.

You can find the complete list of dependencies in requirements.txt.

How to Train the Model
To train the model, run the train.py script:

bash
Sao chép
Chỉnh sửa
python train.py
This will start the training process. The agent will interact with the environment, try different actions, and gradually improve by learning from the rewards it receives. The model will be saved every 500 episodes in a file called flappy_bird_dqn.pth.

Model Details
The model is a simple Deep Q-Network (DQN) consisting of a neural network. It takes the state of the game as input (including bird's position, velocity, and the position of pipes) and outputs Q-values for each action (jump or don't jump). The model is updated after each episode using the Q-learning algorithm.

How to Play the Game
Once the model is trained, you can load the trained model and have the agent play the game. Simply run:

bash
Sao chép
Chỉnh sửa
python play.py
This will allow the agent to play using the trained model and demonstrate its learned behavior.

Contributing
If you'd like to contribute to the project, feel free to submit a pull request. You can also report any issues or bugs by opening an issue in the GitHub repository.

License
This project is open source and available under the MIT License.