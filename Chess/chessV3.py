import chess
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# Define piece values (positive for white, negative for black)
piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 4,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 50  # King's value set to 50
}

# DQN Network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Chess AI class for DQN
class ChessAI:
    def __init__(self, piece_values, epsilon=0.1, gamma=0.99, lr=0.001, batch_size=32):
        self.board = chess.Board()
        self.piece_values = piece_values
        self.epsilon = epsilon  # Exploration probability
        self.gamma = gamma  # Discount factor
        self.lr = lr
        self.batch_size = batch_size

        self.memory = deque(maxlen=10000)  # Experience replay
        self.model = DQN(64, 64)  # Input: 64 (8x8 board), Output: 64 legal moves
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def state_to_tensor(self, board):
        # Convert the board into a tensor (flattened 64-size array)
        state = np.zeros(64)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                state[square] = piece_values.get(piece.piece_type, 0) * (1 if piece.color == chess.WHITE else -1)
        return torch.tensor(state, dtype=torch.float32)

    def get_legal_moves(self):
        return list(self.board.legal_moves)

    def choose_action(self, state_tensor):
        if random.random() < self.epsilon:
            # Exploration: choose a random action
            legal_moves = self.get_legal_moves()
            return random.choice(legal_moves)
        else:
            # Exploitation: choose the best action from the Q-network
            with torch.no_grad():
                q_values = self.model(state_tensor)  # Get all Q-values from the model
                legal_moves = self.get_legal_moves()

                # For each legal move, calculate the Q-value
                legal_q_values = []
                for move in legal_moves:
                    # Map move to a 64-dimensional Q-value space
                    move_index = move.from_square * 64 + move.to_square
                    legal_q_values.append(q_values[move_index].item())

                # Get the index of the move with the highest Q-value
                best_move_index = np.argmax(legal_q_values)
                return legal_moves[best_move_index]

    def update_q_values(self, state_tensor, action, reward, next_state_tensor, done):
        # Get the Q value for the chosen action
        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)
        
        # Q-learning update
        target = reward
        if not done:
            target += self.gamma * torch.max(next_q_values)
        
        # Update Q value for the chosen action
        action_index = action.from_square * 64 + action.to_square
        loss = nn.MSELoss()(q_values[action_index], target)

        # Backpropagate and update the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calculate_captured_score(self, previous_board):
        white_score = 0
        black_score = 0
        
        # Loop through all moves and check for captures
        for move in self.board.move_stack:
            if move in previous_board.move_stack:
                continue  # Skip moves that are already in the previous board
            piece = self.board.piece_at(move.to_square)
            if piece:
                if piece.color == chess.WHITE:  # Black captures a white piece
                    black_score += piece_values.get(piece.piece_type, 0)
                elif piece.color == chess.BLACK:  # White captures a black piece
                    white_score += piece_values.get(piece.piece_type, 0)

        return white_score, black_score

    def get_move_reward(self, previous_board):
        white_score, black_score = self.calculate_captured_score(previous_board)

        # Update the total reward with the captured piece values
        reward = 0
        if white_score > black_score:
            reward = white_score - black_score  # Positive reward for white's capture
        elif black_score > white_score:
            reward = black_score - white_score  # Negative reward for black's capture

        # Account for losing pieces
        for move in self.board.move_stack:
            piece = self.board.piece_at(move.to_square)
            if piece:
                # Subtract piece values for lost pieces
                if piece.color == chess.WHITE and move.from_square not in previous_board.piece_map():
                    reward -= piece_values.get(piece.piece_type, 0)
                elif piece.color == chess.BLACK and move.from_square not in previous_board.piece_map():
                    reward += piece_values.get(piece.piece_type, 0)

        return reward

    def train(self):
        # Train the DQN using experience replay
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from memory
        batch = random.sample(self.memory, self.batch_size)
        for state_tensor, action, reward, next_state_tensor, done in batch:
            self.update_q_values(state_tensor, action, reward, next_state_tensor, done)

    def play_game(self):
        total_reward = 0
        while not self.board.is_game_over():
            # Get the current state
            state_tensor = self.state_to_tensor(self.board)
            
            # Choose the action
            action = self.choose_action(state_tensor)

            # Apply the action
            previous_board = self.board.copy()
            self.board.push(action)
            
            # Get the reward
            reward = self.get_move_reward(previous_board)

            # Get the next state
            next_state_tensor = self.state_to_tensor(self.board)

            # Check if the game is over
            done = self.board.is_game_over()

            # Store in memory
            self.memory.append((state_tensor, action, reward, next_state_tensor, done))

            # Train the model
            self.train()

            total_reward += reward

            if done:
                break
        
        return total_reward

# Start the training loop for AI vs AI
def train_chess_ai(episodes=1000):
    ai = ChessAI(piece_values)
    rewards = []

    for episode in range(episodes):
        print(f"Training episode {episode+1}/{episodes}")
        episode_reward = ai.play_game()
        rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1} completed. Total reward: {episode_reward}")

    return rewards

# Function to display the board in a textual format
def display_board(board):
    print(board)

# Run training and plot rewards
if __name__ == "__main__":
    print("Starting the chess AI training!")
    
    # Train the AI and collect rewards for plotting
    rewards = train_chess_ai(episodes=1000)

    # Plotting the rewards over time
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('AI Training Rewards Over Episodes')
    plt.show()
