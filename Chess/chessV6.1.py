import chess
import tkinter as tk
from tkinter import messagebox
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


GREEN = f'#395631'
BEIGE = f'#d1b281'

# Initialize the chess board
board = chess.Board()

# Colors for the chess pieces
white_pieces = {
    chess.PAWN: "♙", chess.KNIGHT: "♘", chess.BISHOP: "♗", chess.ROOK: "♖",
    chess.QUEEN: "♕", chess.KING: "♔"
}
black_pieces = {
    chess.PAWN: "♟", chess.KNIGHT: "♞", chess.BISHOP: "♝", chess.ROOK: "♜",
    chess.QUEEN: "♛", chess.KING: "♚"
}

# Tkinter setup
root = tk.Tk()
root.title("Chess Game")

# Create a canvas to draw the chessboard
canvas = tk.Canvas(root, width=400, height=400)
canvas.pack()

# Define the Deep Q-Network (DQN) model
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        # Input layer: 12 * 8 * 8 = 768 (flattened input)
        self.fc1 = nn.Linear(12 * 8 * 8, 512)  # First hidden layer with 512 units
        self.fc2 = nn.Linear(512, 512)         # Second hidden layer with 512 units
        self.fc3 = nn.Linear(512, 256)         # Third hidden layer with 256 units
        self.fc4 = nn.Linear(256, 256)         # Fourth hidden layer with 256 units
        self.fc5 = nn.Linear(256, 128)         # Fifth hidden layer with 128 units
        self.fc6 = nn.Linear(128, 128)         # Sixth hidden layer with 128 units
        self.fc7 = nn.Linear(128, 64)          # Seventh hidden layer with 64 units
        self.fc8 = nn.Linear(64, 64)           # Eighth hidden layer with 64 units
        self.fc9 = nn.Linear(64, 64)           # Ninth hidden layer with 64 units
        self.fc10 = nn.Linear(64, 4672)        # Output layer: 4672 (number of possible moves)

    def forward(self, x):
        # Passing the input through all the layers with ReLU activation
        x = torch.relu(self.fc1(x))  # First hidden layer
        x = torch.relu(self.fc2(x))  # Second hidden layer
        x = torch.relu(self.fc3(x))  # Third hidden layer
        x = torch.relu(self.fc4(x))  # Fourth hidden layer
        x = torch.relu(self.fc5(x))  # Fifth hidden layer
        x = torch.relu(self.fc6(x))  # Sixth hidden layer
        x = torch.relu(self.fc7(x))  # Seventh hidden layer
        x = torch.relu(self.fc8(x))  # Eighth hidden layer
        x = torch.relu(self.fc9(x))  # Ninth hidden layer
        x = self.fc10(x)             # Output layer (no activation function here)
        return x


# Define the ChessRLAI class
class ChessRLAI:
    def __init__(self, model, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, gamma=0.99):
        self.model = model
        self.epsilon = epsilon  # Initial epsilon
        self.epsilon_min = epsilon_min  # Minimum value of epsilon
        self.epsilon_decay = epsilon_decay  # Decay factor
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()  # Loss function for Q-values
        self.gamma = gamma  # Discount factor for future rewards
    
    def select_action(self, state):
        """
        Selects an action using epsilon-greedy strategy, with epsilon decay.
        """
        if random.random() < self.epsilon:
            legalMoves = list(board.legal_moves)
            action = random.choice(legalMoves)
        else:
            with torch.no_grad():
                q_values = self.model(state)  # Get Q-values for all possible moves
                action = torch.argmax(q_values).item()  # Select the move with the highest Q-value
        
        # Decay epsilon after every action to reduce exploration over time
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        return action

# Function to display the board in Tkinter
def display_board():
    # Clear the canvas
    canvas.delete("all")
    
    square_size = 50
    for row in range(8):
        for col in range(8):
            # Determine the color of the square
            color = GREEN if (row + col) % 2 == 0 else BEIGE
            canvas.create_rectangle(col * square_size, row * square_size,
                                    (col + 1) * square_size, (row + 1) * square_size,
                                    fill=color)
            
            # Get the piece at the current square
            piece = board.piece_at(chess.square(col, 7 - row))  # Flip y-axis for correct orientation
            if piece:
                piece_char = ""
                if piece.color == chess.WHITE:
                    piece_char = white_pieces.get(piece.piece_type, "")
                else:
                    piece_char = black_pieces.get(piece.piece_type, "")
                canvas.create_text(col * square_size + square_size / 2,
                                   row * square_size + square_size / 2,
                                   text=piece_char, font=("Arial", 24))

    # Draw row labels (1-8) on the left side of the board
    for row in range(8):
        canvas.create_text(-10, row * square_size + square_size / 2,
                           text=str(8 - row), font=("Arial", 14))  # 8-1, 7-2, ..., 1-8

    # Draw column labels (a-h) at the bottom of the board
    for col in range(8):
        canvas.create_text(col * square_size + square_size / 2,
                           8 * square_size + 10,  # Position below the board
                           text=chr(ord('a') + col), font=("Arial", 14))


def display_title():
    # Clear the canvas
    canvas.delete("all")
    
    # Add the title text at the top of the canvas
    canvas.create_text(200, 50, text="Welcome to Chess!", font=("Arial", 24, "bold"), fill="black")

    # Add a description text below the title
    description = (
        "Choose a game mode to start: \n\n"
        "1. Player vs Player\n"
        "2. Player vs RLAI\n"
        "3. RLAI vs RLAI (not implemented)"
    )
    canvas.create_text(200, 150, text=description, font=("Arial", 14), fill="black", justify="center")
    
    # Ensure canvas updates
    root.update()  # This forces an update of the canvas and GUI elements


# Function to handle a player's move from the console
def make_move(move_uci):
    global board
    try:
        # Apply the move using UCI notation
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            board.push(move)
            print(f"Move made: {move_uci}")
            display_board()
        else:
            print("Invalid move!")
    except ValueError:
        print("Invalid UCI format or move.")


def board_to_tensor(board):
    # Create a tensor to represent the board state (12 layers, 8x8 grid)
    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)  # 12 layers (6 piece types for white and black)

    # Iterate over each square on the chessboard (64 squares in total)
    for square in range(64):
        piece = board.piece_at(square)

        if piece:  # If there is a piece at this square
            row, col = divmod(square, 8)  # Get the row and column from the square number

            # Determine the piece type and color
            piece_type = piece.piece_type
            color = piece.color

            # Determine the index in the tensor: for white pieces, use positive values
            # For black pieces, use negative values and the index to determine layer
            layer = piece_type - 1  # piece_type 1 corresponds to layer 0, etc.

            if color == chess.BLACK:
                layer += 6  # Black pieces are in the second half of the tensor (layers 6 to 11)

            # Add the piece's value to the appropriate position in the tensor
            tensor[layer, row, col] = 1  # We are simply marking the presence of a piece, so use a value of 1

    return tensor

def print_board_tensor():
    tensor = board_to_tensor(board)
    print("Tensor shape:", tensor.shape)
    print(tensor)

# Function to ask the player for the game mode
def select_game_mode():
    print("Select Game Mode:")
    print("Type 1 for Player vs Player (PvP)")
    print("Type 2 for Player vs RLAI (PvRLAI)")
    print("Type 3 for RLAI vs RLAI (not implemented yet)")
    mode = input("Enter the number of your choice: ")
    return mode

# Function to handle the Player vs Player mode
def play_pvp():
    while not board.is_game_over():
        display_board()

        print(" ")
        # Print all legal moves
        print("Legal moves: ")
        print(board.legal_moves)

        # Get the move from the player (example: "e2e4")
        move_uci = input("Enter your move (e.g. 'e2e4'): ")
        print_board_tensor()

        # Try to make the move
        make_move(move_uci)

    print("Game Over!")
    print("Result: " + board.result())

# Function to handle the Player vs RLAI mode
# Function to handle the Player vs RLAI mode
def play_pvrla():
    # Initialize the ChessRLAI agent for the AI
    rla_agent = ChessRLAI(model=DQN())

    while not board.is_game_over():
        display_board()

        print(" ")
        # Print all legal moves
        print("Legal moves: ")
        print(board.legal_moves)

        # Get the move from the player (example: "e2e4")
        move_uci = input("Enter your move (e.g. 'e2e4'): ")
        print_board_tensor()

        # Try to make the player's move
        make_move(move_uci)

        # If it's the RLAI's turn (assume AI plays after player)
        if not board.is_game_over():
            state = board_to_tensor(board).unsqueeze(0)  # Add batch dimension
            action = rla_agent.select_action(state)  # Let the RLAI select its move

            # Ensure that 'action' is one of the legal moves
            legal_moves = list(board.legal_moves)

            # Check if the selected action (move) is a valid legal move
            if action in legal_moves:
                move = action  # Get the corresponding move from legal_moves
                print(f"RLAI plays: {move}")
                make_move(move.uci())
            else:
                print("Invalid action selected by RLAI!")

    print("Game Over!")
    print("Result: " + board.result())


# Main function to start the game based on the selected mode
def start_game():
    display_title()
    
    mode = select_game_mode()

    if mode == "1":
        print("Starting Player vs Player mode...")
        play_pvp()
    elif mode == "2":
        print("Starting Player vs RLAI mode...")
        play_pvrla()
    elif mode == "3":
        print("RLAI vs RLAI mode is not implemented yet.")
    else:
        print("Invalid choice. Please restart the game and select a valid mode.")

# Start the game
start_game()

# Initial board display
display_board()
