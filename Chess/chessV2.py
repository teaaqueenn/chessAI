import pygame
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

# Initialize Pygame
pygame.init()

# Constants for the board
SQUARE_SIZE = 100
BOARD_SIZE = SQUARE_SIZE * 8
WIDTH, HEIGHT = BOARD_SIZE, BOARD_SIZE
LIGHT_SQUARE_COLOR = (235, 236, 208)
DARK_SQUARE_COLOR = (119, 148, 85)
SELECTED_SQUARE_COLOR = (0, 255, 0, 128)
LEGAL_MOVE_COLOR = (0, 0, 255)

# Hyperparameters for DQN
ALPHA = 0.001  # Learning rate
GAMMA = 0.99   # Discount factor
EPSILON_START = 1.0  # Start epsilon (exploration rate)
EPSILON_END = 0.1    # End epsilon (exploitation rate)
EPSILON_DECAY = 0.995 # Decay rate for epsilon
BATCH_SIZE = 64  # Experience replay batch size
REPLAY_MEMORY_SIZE = 10000  # Replay buffer size
TARGET_UPDATE_FREQUENCY = 10  # Frequency of target network updates

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Game")

# Initialize chess board
board = chess.Board()

# Load images for chess pieces
def load_images():
    images = {}
    for color in ["white", "black"]:
        for piece in ["rook", "knight", "bishop", "queen", "king", "pawn"]:
            image_path = f"images/{color}_{piece}.png"  # Update the path accordingly
            image = pygame.image.load(image_path)
            image = pygame.transform.scale(image, (SQUARE_SIZE // 2, SQUARE_SIZE // 2))
            images[f"{color}_{piece}"] = image
    return images

# Drawing Functions
def draw_board():
    for row in range(8):
        for col in range(8):
            color = LIGHT_SQUARE_COLOR if (row + col) % 2 == 0 else DARK_SQUARE_COLOR
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces(images):
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        piece_color = "white" if piece.color == chess.WHITE else "black"
        piece_names = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
        piece_str = f"{piece_color}_{piece_names[piece.piece_type]}"
        piece_image = images[piece_str]
        screen.blit(piece_image, (col * SQUARE_SIZE + (SQUARE_SIZE - piece_image.get_width()) // 2, row * SQUARE_SIZE + (SQUARE_SIZE - piece_image.get_height()) // 2))

def draw_selected_square(selected_square):
    if selected_square is not None:
        row, col = divmod(selected_square, 8)
        pygame.draw.rect(screen, SELECTED_SQUARE_COLOR, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

def draw_legal_moves(legal_moves):
    for move in legal_moves:
        row, col = divmod(move.to_square, 8)
        pygame.draw.circle(screen, LEGAL_MOVE_COLOR, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), 10)

# Player Input and Move Handling
# Global variable to store move history
move_history = []

def handle_player_input():
    selected_square = None
    legal_moves = []
    images = load_images()  # Load images once here

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                col, row = pos[0] // SQUARE_SIZE, pos[1] // SQUARE_SIZE
                clicked_square = chess.square(col, row)

                if selected_square is None:
                    if board.piece_at(clicked_square):
                        selected_square = clicked_square
                        legal_moves = [move for move in board.legal_moves if move.from_square == selected_square]
                else:
                    move = chess.Move(selected_square, clicked_square)
                    if move in legal_moves:
                        board.push(move)
                        # Append the move to move history in descriptive format
                        move_history.append(board.san(move))  # Use SAN (Standard Algebraic Notation)
                        selected_square = None
                        legal_moves = []

        # Redraw the board and pieces
        draw_board()
        draw_pieces(images)
        draw_selected_square(selected_square)
        draw_legal_moves(legal_moves)

        # Display the move history
        draw_move_history()

        pygame.display.flip()

def draw_move_history():
    # Display the move history on the side of the screen
    font = pygame.font.Font(None, 36)
    y_offset = 10
    for move in move_history:
        text = font.render(move, True, (255, 255, 255))
        screen.blit(text, (WIDTH + 20, y_offset))  # Position the text on the right side
        y_offset += 40  # Increase the vertical offset for each new move

# Define the neural network for DQN
class ChessDQN(nn.Module):
    def __init__(self):
        super(ChessDQN, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 64)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def add(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)  # Return the length of the buffer

# Epsilon-greedy policy and Q-learning update
def epsilon_greedy_policy(model, state, legal_moves, epsilon):
    if random.random() < epsilon:
        # Choose a random move from the legal moves
        return random.choice(legal_moves)
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = model(state_tensor)  # Get the Q-values from the model
        # Find the Q-value corresponding to each legal move
        legal_move_indices = [legal_moves.index(move) for move in legal_moves]
        q_values_for_legal_moves = q_values[0, legal_move_indices]
        best_move_index = torch.argmax(q_values_for_legal_moves).item()
        return legal_moves[best_move_index]

def update_q_values(model, target_model, buffer, optimizer):
    if len(buffer) < BATCH_SIZE:
        return
    batch = buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, done_flags = zip(*batch)
    
    # Convert lists of states and next_states to numpy arrays first
    states = np.array(states, dtype=np.float32)  # Convert list to a numpy array
    next_states = np.array(next_states, dtype=np.float32)  # Convert list to a numpy array
    
    # Then convert to torch tensors
    states = torch.tensor(states, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    done_flags = torch.tensor(done_flags, dtype=torch.float32)

    q_values = model(states)
    next_q_values = target_model(next_states)
    max_next_q_values = next_q_values.max(1)[0]

    q_target = rewards + GAMMA * max_next_q_values * (1 - done_flags)

    loss = nn.functional.mse_loss(q_values.gather(1, actions.unsqueeze(1)), q_target.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate state and reward from the board
def get_state_from_board(board):
    state = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type
        piece_color = piece.color
        channel = piece_type - 1 if piece_color == chess.WHITE else piece_type + 5
        state[channel, row, col] = 1.0
    return state

def get_reward(board, previous_board):
    reward = 0
    if board.is_checkmate():
        reward += 20
    elif previous_board.is_checkmate():
        reward -= 20

    for square, piece in previous_board.piece_map().items():
        if not board.piece_at(square):
            piece_value = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 20}
            if piece.color == chess.WHITE:
                reward -= piece_value[piece.piece_type]
            else:
                reward += piece_value[piece.piece_type]
    return reward

# Training and Self-play
def self_play(model, target_model, optimizer, replay_buffer, epsilon, episode_rewards):
    board = chess.Board()
    previous_board = chess.Board()
    done = False
    total_reward = 0
    moves = []

    while not done:
        state = get_state_from_board(board)
        legal_moves = list(board.legal_moves)  # Get the legal moves
        action = epsilon_greedy_policy(model, state, legal_moves, epsilon)  # Pass legal moves to the policy
        
        move = action  # Now the action is a move object
        moves.append(move)
        previous_board.set_fen(board.fen())
        board.push(move)
        
        reward = get_reward(board, previous_board)
        done = board.is_game_over()

        replay_buffer.add((state, legal_moves.index(action), reward, get_state_from_board(board), done))

        update_q_values(model, target_model, replay_buffer, optimizer)

        if len(replay_buffer.buffer) % TARGET_UPDATE_FREQUENCY == 0:
            target_model.load_state_dict(model.state_dict())

        total_reward += reward

    episode_rewards.append(total_reward)

    return moves, episode_rewards

def plot_performance(rewards):
    plt.plot(rewards)
    plt.title('Agent Performance Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.show()

def train(model, target_model, optimizer, replay_buffer, episodes):
    epsilon = EPSILON_START
    episode_rewards = []

    for episode in range(episodes):
        moves, episode_rewards = self_play(model, target_model, optimizer, replay_buffer, epsilon, episode_rewards)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        print(f"Episode {episode + 1}, Total Reward: {episode_rewards[-1]}")

        if episode % 10 == 0:
            plot_performance(episode_rewards)

# Main function
if __name__ == "__main__":
    model = ChessDQN()
    target_model = ChessDQN()
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=ALPHA)
    replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)

    train(model, target_model, optimizer, replay_buffer, 1000)
