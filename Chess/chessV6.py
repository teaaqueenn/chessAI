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

# Define piece values (positive for white, negative for black)
piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # King is never captured in normal play
}

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

# Create labels for the score
white_score_label = tk.Label(root, text="White Score: 0")
white_score_label.pack(side="left")
black_score_label = tk.Label(root, text="Black Score: 0")
black_score_label.pack(side="right")

# Variables to store the captured pieces' scores
white_score = 0
black_score = 0

class ChessAI:
    def __init__(self, model, epsilon=0.1):
        self.model = model  # The DQN model
        self.epsilon = epsilon  # Exploration rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def select_move(self, state_tensor):
        # Epsilon-greedy strategy for exploration vs exploitation
        if random.random() < self.epsilon:
            # Exploration: Choose a random legal move
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
        else:
            # Exploitation: Choose the best move according to the model
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                q_values = self.model(state_tensor)  # Get Q-values for all possible moves
            
            # Debugging: Check the shape of q_values
            print("q_values shape:", q_values.shape)  # This will show if q_values is a scalar or has the expected shape

            q_values = q_values.squeeze().cpu().numpy()  # Convert to numpy array

            # Get legal moves
            legal_moves = list(board.legal_moves)

            # Create a list to store the Q-values for the legal moves
            legal_move_q_values = []

            for move in legal_moves:
                # Add the Q-value corresponding to the "from_square" of the move
                legal_move_q_values.append(q_values[move.from_square])

            # Get the index of the best move (highest Q-value)
            best_move_index = np.argmax(legal_move_q_values)

            # Select the best move
            move = legal_moves[best_move_index]

        return move




    def train(self, transition, gamma=0.99):
        state, action, reward, next_state = transition
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        
        # Compute Q-value for current state and action
        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)
        target_q_value = reward + gamma * torch.max(next_q_values)
        
        # Compute loss and update the model
        target_q_values = q_values.clone()
        target_q_values[action] = target_q_value

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(8 * 8 * 12, 128)  # Flatten the board tensor into a vector
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 1)  # Output a Q-value for each possible action

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  # Output Q-values for each possible move
        return x


# Function to display the board in Tkinter
def display_board():
    global white_score, black_score
    
    # Clear the canvas
    canvas.delete("all")
    
    # Loop over the 8x8 board to draw the pieces
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

    # Update the score labels
    white_score_label.config(text=f"White Score: {white_score}")
    black_score_label.config(text=f"Black Score: {black_score}")


# Function to handle a player's move from the console
def make_move(move_uci):
    global board
    try:
        # Apply the move using UCI notation
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            board.push(move)
            print(f"Move made: {move_uci}")
            #update_scores()
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
            tensor[layer, row, col] = piece_values.get(piece_type, 0)

    return tensor

def print_board_tensor():
    tensor = board_to_tensor(board)
    print("Tensor shape:", tensor.shape)
    print(tensor)


# Function to play the game (with console input for moves)
def play_game():
    # Assume white is the human player and black is the AI
    ai = ChessAI(QNetwork(), epsilon=0.2)  # Initialize AI with the Q-network model
    while not board.is_game_over():
        display_board()

        if board.turn == chess.WHITE:  # Player's turn
            move_uci = input("Enter your move (e.g. 'e2e4'): ")
            make_move(move_uci)

        else:  # AI's turn
            state_tensor = board_to_tensor(board)  # Convert the board to a tensor
            ai_move = ai.select_move(state_tensor)  # AI selects a move
            make_move(ai_move.uci())  # Make the AI's move
        
        print_board_tensor()

    print("Game Over!")
    print("Result: " + board.result())

def train_ai(ai, experience_buffer, batch_size=32):
    if len(experience_buffer) < batch_size:
        return
    
    # Sample a batch of experiences
    batch = random.sample(experience_buffer, batch_size)
    
    for state, action, reward, next_state in batch:
        ai.train((state, action, reward, next_state))

class ExperienceReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def add(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)



def start_game_thread():
    threading.Thread(target=play_game, daemon=True).start()

# Initial board display
display_board()

# Start the game (in a thread)
start_game_thread()

# Main loop
root.mainloop()