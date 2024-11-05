import chess
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import numpy as np
import random
import matplotlib.pyplot as plt

# Chessboard constants
BOARD_SIZE = 8
SQUARE_SIZE = 64
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (185, 122, 87)
GREEN = (0, 255, 0)

# Neural network model
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Chess environment
class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def step(self, action):
        # Convert action (e.g., "c4") to a valid UCI move
        source_square = None  # You might need to generate a source square if needed
        dest_square = chess.parse_square(action)

        move = chess.Move(source_square, dest_square)
        self.board.push(move)

        # Calculate reward based on captured or lost pieces
        reward = 0
        captured_piece = self.board.peek()  # Get the captured piece
        if captured_piece:
            if captured_piece.piece_type == chess.PAWN:
                reward += 1 if captured_piece.color == chess.BLACK else -1
            elif captured_piece.piece_type == chess.KNIGHT or captured_piece.piece_type == chess.BISHOP:
                reward += 2 if captured_piece.color == chess.BLACK else -2
            elif captured_piece.piece_type == chess.ROOK:
                reward += 5 if captured_piece.color == chess.BLACK else -5
            elif captured_piece.piece_type == chess.QUEEN:
                reward += 9 if captured_piece.color == chess.BLACK else -9
            elif captured_piece.piece_type == chess.KING:
                reward += 20 if captured_piece.color == chess.BLACK else -20

        # Check for checkmate or stalemate
        done = self.board.is_checkmate() or self.board.is_stalemate()

        return self.board.fen(), reward, done

    def reset(self):
        self.board.reset()
        return self.board.fen()

# State representation
def state_to_vector(fen):
    board = chess.Board(fen)
    vector = np.zeros(128)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.piece_type
            color = piece.color
            vector[square * 6 + color] = 1
    return vector

# Q-learning algorithm
def q_learning(model, env, episodes=1000, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, gamma=0.99):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    rewards = []
    losses = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            if np.random.rand() < epsilon:
                # Randomly choose a valid chess square
                action = random.choice(list(chess.SQUARES))
                print("current action", action)
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state_to_vector(state)).float())
                action_index = torch.argmax(q_values).item()
                action = chess.SQUARE_NAMES[action_index]

            # Ensure action is a valid chess square name
            if not isinstance(action, str) or len(action) != 2:
                raise ValueError("Invalid action format. Use a chess square name (e.g., 'e4', 'c7').")

            next_state, reward, done = env.step(action)
            with torch.no_grad():
                target = reward + gamma * torch.max(model(torch.tensor(state_to_vector(next_state)).float())).item()
            target = torch.tensor([target]).float()

            optimizer.zero_grad()
            loss = criterion(target, q_values)
            loss.backward()
            optimizer.step()

            total_reward += reward
            state = next_state
            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)
        losses.append(loss.item())
        print(f"Episode: {episode}, Total Reward: {total_reward}")

    # Plot rewards and losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title("Losses Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.show()

    return rewards

# Pygame visualization
def draw_board(screen, board):
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            color = BROWN if (row + col) % 2 == 0 else WHITE
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            image = pygame.image.load(f"images/{piece.symbol()}.png")
            rect = image.get_rect()
            rect.center = (chess.SQUARE_NAMES[square][0] * SQUARE_SIZE + SQUARE_SIZE // 2, chess.SQUARE_NAMES[square][1] * SQUARE_SIZE + SQUARE_SIZE // 2)
            screen.blit(image, rect)

def main():
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE * SQUARE_SIZE, BOARD_SIZE * SQUARE_SIZE))
    pygame.display.set_caption("Chess Reinforcement Learning")

    env = ChessEnv()
    model = ChessNet()
    rewards = q_learning(model, env)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_board(screen, env.board)
        pygame.display.flip()

if __name__ == "__main__":
    main()