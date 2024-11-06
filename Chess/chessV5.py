import chess
import tkinter as tk
from tkinter import messagebox
import threading
import torch
import numpy as np


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
    while not board.is_game_over():
        display_board()

        print(" ")
        # Print all legal moves
        print("Legal moves:")
        print(board.legal_moves)

        # Get the move from the player (example: "e2e4")
        move_uci = input("Enter your move (e.g. 'e2e4'): ")
        print_board_tensor()

        # Try to make the move
        make_move(move_uci)
        
    print("Game Over!")
    print("Result: " + board.result())
    print(f"White's Total Score: {white_score}")
    print(f"Black's Total Score: {black_score}")


def start_game_thread():
    threading.Thread(target=play_game, daemon=True).start()

# Initial board display
display_board()

# Start the game (in a thread)
start_game_thread()

# Main loop
root.mainloop()