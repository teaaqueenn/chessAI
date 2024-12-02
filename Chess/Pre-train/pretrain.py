import chess
import chess.engine
import numpy as np
import torch
import os
from tqdm import tqdm  # Import tqdm for the loading bar

board = chess.Board()

def board_to_tensor(board):
    """Convert the chess board to a tensor representation."""
    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)  # 12 layers (6 piece types for white and black)
    
    for square in range(64):
        piece = board.piece_at(square)
        
        if piece:  # If there is a piece at this square
            row, col = divmod(square, 8)  # Get the row and column from the square number

            # Determine the piece type and color
            piece_type = piece.piece_type
            color = piece.color

            # Determine the index in the tensor
            layer = piece_type - 1  # piece_type 1 corresponds to layer 0, etc.

            if color == chess.BLACK:
                layer += 6  # Black pieces are in the second half of the tensor (layers 6 to 11)

            tensor[layer, row, col] = 1  # Mark the presence of a piece

    return tensor.view(-1)  # Flatten the tensor to 768-length vector

# Path to your Stockfish engine
engine_path = r"C:\Users\Grace\Documents\GitHub\chessAI\Chess\Pre-train\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"

# Initialize Stockfish engine
try:
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    board = chess.Board()
    info = engine.analyse(board, chess.engine.Limit(time=1.0))
    print(info)  # If this prints, the engine is working
    engine.quit()
except Exception as e:
    print("Error with Stockfish engine:", e)

def load_existing_games():
    """Load previously saved games from 'chess_games.pt' if it exists."""
    if os.path.exists('chess_games.pt'):
        print("Loading existing game data...")
        return torch.load('chess_games.pt', weights_only=True)  # Load existing games
    else:
        print("No previous game data found. Starting fresh.")
        return []

def save_games_as_tensor(games):
    """Save the game data as tensors."""
    tensor_data = []

    # Using tqdm to create a progress bar for saving the games as tensors
    for game in tqdm(games, desc="Saving games", ncols=100, unit="game"):
        board = chess.Board()  # Reset board for each game
        for move in game:
            board_tensor = board_to_tensor(board)
            tensor_data.append((board_tensor, move))
            board.push(move)
    
    # Save tensor data to a file
    torch.save(tensor_data, 'chess_games.pt')

def generate_games(num_games=5000):
    """Generate new chess games and combine with existing data."""
    # Load previously saved games if they exist
    existing_games = load_existing_games()
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    new_games = []

    # Using tqdm to create a progress bar for generating games
    for game_idx in tqdm(range(num_games), desc="Generating games", ncols=100, unit="game"):
        board = chess.Board()
        game_moves = []
        while not board.is_game_over():
            result = engine.play(board, chess.engine.Limit(time=1.0))
            game_moves.append(result.move)
            board.push(result.move)

        # Append the generated game to the new_games list
        new_games.append(game_moves)

        # Save progress after each game
        board_tensor = board_to_tensor(board)  # Convert the board to tensor
        tensor_data = [(board_tensor, result.move)]  # Save the current move and board tensor
        existing_games.append(tensor_data[0])  # Append the current board state and move
        torch.save(existing_games, 'chess_games.pt')  # Save to the tensor file after each game

        # Print progress message every 10 games
        if (game_idx + 1) % 10 == 0:
            print(f"10 games completed. {game_idx + 1}/{num_games} games generated.")

    engine.quit()

    # Combine the new games with the existing games and return
    all_games = existing_games + new_games
    return all_games

# Generate and save the dataset
games = generate_games()
save_games_as_tensor(games)