import chess
import tkinter as tk
from tkinter import messagebox
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import copy
from tkinter import simpledialog
import os
import random
import torch
import chess
import chess.engine
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import ChessDQN as DQN

class ChessRLAI:
    def __init__(self, model = DQN(), epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, gamma=0.99):
        self.model = model
        self.epsilon = epsilon  # Initial epsilon
        self.epsilon_min = epsilon_min  # Minimum value of epsilon
        self.epsilon_decay = epsilon_decay  # Decay factor
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()  # Loss function for Q-values
        self.gamma = gamma  # Discount factor for future rewards

        self.total_reward = 0.0
        self.sum_reward = 0.0
        self.total_loss = 0.0
        self.turn_reward = 0.0
        self.turn_q_values = []

        self.load_pretrained_weights(r"C:\Users\Grace\Documents\GitHub\chessAI\chess_ai_model_v1.pth")

    def load_model(model):
        filename = r"C:\Users\Grace\Documents\GitHub\chessAI\chess_ai_model_v1.pth"
        model.load_state_dict(torch.load(filename))
        model.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {filename}")
    
    def board_to_tensor(self, board):
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

        # Flatten the tensor to match the input shape expected by the neural network
        tensor = tensor.view(-1)  # Flatten the 12x8x8 tensor into a 768-length vector
        return tensor


    def load_pretrained_weights(self, path):
        self.model.load_state_dict(torch.load(path))

    def update_model_after_game(self, game_moves, game_rewards):
        """
        Update the model after the entire game using Q-learning.

        Parameters:
            game_moves (list of chess.Move): The list of moves made during the game.
            game_rewards (list of float): The rewards received for each move in the game.
        """
        # Initialize total loss for the game
        total_loss = 0

        for move, reward in zip(game_moves, game_rewards):
            # Convert the current board state to tensor
            state_tensor = self.board_to_tensor(move.board()).unsqueeze(0)  # Add batch dimension

            # Get the Q-values for the current state
            current_q_values = self.model(state_tensor).squeeze(0)

            # Simulate the move by copying the board and making the move on the clone
            cloned_board = copy.deepcopy(move.board())  # Clone the board
            cloned_board.push(move)  # Make the move on the cloned board

            # Convert the new state after the move to tensor
            next_state_tensor = self.board_to_tensor(cloned_board).unsqueeze(0)  # Get the next state tensor after the move

            # Get the Q-values for the next state
            next_q_values = self.model(next_state_tensor).squeeze(0)

            # Calculate the maximum Q-value for the next state
            max_next_q_value = torch.max(next_q_values).item()

            # Find the index of the move in the model's Q-value predictions
            legal_moves = list(move.board().legal_moves)
            move_index = legal_moves.index(move)

            # Get the current Q-value for the move
            current_q_value = current_q_values[move_index].item()

            # Calculate the new Q-value using the Q-learning equation
            updated_q_value = current_q_value + self.learning_rate * (reward + self.gamma * max_next_q_value - current_q_value)

            # Create a tensor for the updated Q-value
            updated_q_value_tensor = torch.tensor([updated_q_value]).unsqueeze(0)  # Add batch dimension

            # Prepare the target Q-values for the loss calculation
            target_q_values = current_q_values.clone()
            target_q_values[move_index] = updated_q_value_tensor.item()

            # Calculate the loss (using MSE)
            loss = self.loss_fn(current_q_values, target_q_values)

            # Zero gradients
            self.optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Optimize the model
            self.optimizer.step()

            # Update total loss for the game
            total_loss += loss.item()

        # After the game ends, you can track the total loss, or any other metrics
        self.total_loss += total_loss
        print(f"Total loss for the game: {total_loss}")

    def save_model(self, filename='chess_ai_model_updated.pth'):
        """
        Save the current state of the model to a file.

        Parameters:
            filename (str): The name of the file where the model will be saved.
        """
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved as '{filename}'")

    def is_fork(self, board, square):
        """
        Check if the piece at `square` is performing a fork.
        A fork happens when one piece attacks two or more pieces of the opponent.
        """
        piece = board.piece_at(square)
        if piece is None or piece.color != chess.BLACK:
            return False

        # Get all attacked squares
        attacked_squares = board.attacks(square)
        
        # Count how many opponent pieces are being attacked
        attacked_pieces = [board.piece_at(sq) for sq in attacked_squares if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE]
        
        # If two or more opponent pieces are attacked, it's a fork
        return len(attacked_pieces) >= 2

    def is_double_attack(self, board, square):
        """
        Check if the piece at `square` is attacking two opponent pieces at the same time.
        A double attack occurs when one piece attacks two opponent pieces simultaneously.
        """
        piece = board.piece_at(square)
        if piece is None or piece.color != chess.BLACK:
            return False

        # Get all attacked squares
        attacked_squares = board.attacks(square)
        
        # Count how many opponent pieces are being attacked
        attacked_pieces = [board.piece_at(sq) for sq in attacked_squares if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE]
        
        # If two or more opponent pieces are attacked simultaneously, it's a double attack
        return len(attacked_pieces) >= 2
    
    def is_back_rank_threat(self, board):
        """
        Check if there is a back rank threat against the opponent's king.
        A back rank threat occurs when the opponent's king is on the back rank
        and there is a potential check from a black piece (typically a rook or queen).
        """
        # Get the location of the white king (since we're checking for black's threat)
        white_king_square = board.king(chess.WHITE)
        
        # Check if the white king is on the 1st rank (back rank for white)
        if white_king_square not in chess.SQUARES[0:8]:  # First rank is 0-7 (row 1)
            return False  # White king is not on the back rank
        
        # Check if the opponent's back rank is blocked or if there are escape squares
        for square in chess.SQUARES[0:8]:
            piece = board.piece_at(square)
            if piece and piece.color == chess.WHITE:
                # If there's a white piece on the back rank, the back rank is "blocked"
                # and may provide a potential check from a black piece.
                return True
        
        # If there's a black piece threatening the back rank (typically rooks or queens)
        for square in chess.SQUARES[0:8]:
            # Check if a black piece is on the same rank as the white king
            piece = board.piece_at(square)
            if piece and piece.color == chess.BLACK:
                if piece.piece_type in [chess.ROOK, chess.QUEEN]:  # Corrected line
                    # If a rook or queen is attacking on the back rank, it's a threat
                    if board.is_check():
                        return True
        
        return False
    
    def is_piece_vulnerable(self, board, square):
        # Determine which color is playing
        color = board.turn
        
        # Iterate over all opponent's pieces
        opponent_color = not color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == opponent_color:
                # Check if the opponent can move to the target square
                legal_moves = board.legal_moves
                for move in legal_moves:
                    if move.to_square == square:
                        return True
        return False


    def reward_of_move(self, move, board) -> float:
        """
        Calculate the reward for a given move, considering both tactical motifs and material factors.
        """
        reward = 0.0

        # Define piece values for each type of piece (based on standard chess piece values)
        piece_value = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,  # Adjusted value for rooks (standard: 5, but more dynamic based on board context)
            chess.QUEEN: 9.0,
            chess.KING: 0.0  # King does not have a direct value in terms of material
        }

        # Importance scale for different chess concepts (scaled from 1-10)
        importance_scale = {
            "material_gain_loss": 9.5,
            "check_checkmate": 9.45,
            "king_safety": 9.4,
            "piece_development": 8.95,
            "control_of_center": 8.9,
            "threats_and_tactics": 8.85,
            "pawn_structure": 8.75,
            "piece_activity": 8.6,
            "attacks_on_king": 8.55,
            "rook_queen_activity": 8.5,
            "pawn_breaks": 8.45,
            "passed_pawns": 8.2,
            "creating_plan": 8.15,
            "knight_outposts": 8.1,
            "piece_protection": 8.05,
            "avoiding_hanging_pieces": 8.0,
            "promotion_threats": 7.95,
            "opposite_color_bishops": 7.75,
            "back_rank_weaknesses": 7.6,
            "isolated_pawns": 7.55,
            "doubled_pawns": 7.5,
            "connected_pawns": 7.4,
            "backward_pawns": 7.3,
            "weak_squares": 7.2,
            "en_passant": 6.0
        }

        # Create a copy of the board to simulate the move
        board_copy = board.copy()

        # Step 1: Material gain or loss (scaled reward based on piece value)
        if board_copy.is_capture(move):
            captured_piece = board_copy.piece_at(move.to_square)
            if captured_piece:
                piece_type = captured_piece.piece_type
                reward += importance_scale["material_gain_loss"] * piece_value.get(piece_type, 0.0)

        # Step 2: Check or checkmate (scaled reward)
        board_copy.push(move)
        if board_copy.is_checkmate():
            reward += importance_scale["check_checkmate"]
        elif board_copy.is_check():
            reward += importance_scale["check_checkmate"] * 0.5  # Slightly smaller reward for check
        board_copy.pop()

        # Step 3: King safety (penalize if king becomes exposed)
        if board_copy.is_check():
            reward -= importance_scale["king_safety"] * 0.5  # Penalize for putting the king in check

        # Step 4: Piece development (reward if piece is developed)
        move_piece = board_copy.piece_at(move.from_square)
        if move_piece:
            piece_type = move_piece.piece_type
            if piece_type == chess.PAWN:
                if move.to_square in [chess.C3, chess.F3, chess.C6, chess.F6]:  # Example: central pawn pushes
                    reward += importance_scale["piece_development"]
            elif piece_type in [chess.KNIGHT, chess.BISHOP]:
                if move.to_square in [chess.D3, chess.E3, chess.D6, chess.E6]:  # Example: key squares for piece development
                    reward += importance_scale["piece_development"]

        # Step 5: Control of the center (reward for controlling central squares)
        if move.to_square in [chess.D4, chess.E4, chess.D5, chess.E5]:  # Example: controlling the center
            reward += importance_scale["control_of_center"]

        # Step 6: Threats and tactics (reward for tactical moves like forks, pins) with multipliers
        if self.is_fork(board_copy, move.to_square):
            reward += importance_scale["threats_and_tactics"] * 1.5  
        if board.is_pinned(chess.BLACK, move.to_square):
            reward += importance_scale["threats_and_tactics"] * 1.0
        if self.is_double_attack(board_copy, move.to_square):
            reward += importance_scale["threats_and_tactics"] * 1.2
        if self.is_back_rank_threat(board_copy):
            reward += importance_scale["threats_and_tactics"] * 0.5

        if move_piece == chess.QUEEN:
            if board_copy.is_check():  # Threatening the opponent's king with the queen
                reward += importance_scale["threats_and_tactics"]

        # Step 7: Pawn structure (penalize if structure is weakened)
        if move_piece == chess.PAWN:
            if board_copy.is_isolated(move.to_square):
                reward -= importance_scale["pawn_structure"] * 0.5  # Penalize isolated pawns

        # Step 8: Piece activity (reward if piece becomes more active)
        if move_piece == chess.ROOK:
            if move.to_square in [chess.D1, chess.E1, chess.D8, chess.E8]:  # Example: activating rooks
                reward += importance_scale["rook_queen_activity"]

        # Step 9: Attacks on the opponent's king (reward for attacking moves)
        if board_copy.is_checkmate():
            reward += importance_scale["attacks_on_king"]

        # Step 10: Passed pawns (reward if the move advances a passed pawn)
        if move_piece == chess.PAWN:
            if board_copy.is_passed_pawn(move.to_square):
                reward += importance_scale["passed_pawns"]

        # Step 11: Creating a plan (reward for structured, purposeful moves)
        if move_piece == chess.KNIGHT:
            reward += importance_scale["creating_plan"]

        # Step 12: Knight outposts (reward for placing knights on outposts)
        if move_piece == chess.KNIGHT:
            if move.to_square in [chess.D4, chess.E5, chess.D5, chess.E4]:
                reward += importance_scale["knight_outposts"]

        # Step 13: Piece protection (reward for moves that protect valuable pieces)
        if move_piece in [chess.KNIGHT, chess.ROOK, chess.BISHOP]:
            reward += importance_scale["piece_protection"]

        # Step 14: Avoiding hanging pieces (penalize if a piece is left unprotected)
        if move_piece in [chess.KNIGHT, chess.ROOK, chess.BISHOP, chess.QUEEN]:
            if board_copy.is_hanging(move.to_square):
                reward -= importance_scale["avoiding_hanging_pieces"]

        # Step 15: Promotion threats (reward for advancing a pawn that threatens promotion)
        if move_piece == chess.PAWN:
            if move.to_square in [chess.A8, chess.H8]:
                reward += importance_scale["promotion_threats"]

        # Step 16: Opposite-color bishops (reward if there is a strategic imbalance created)
        if move_piece == chess.BISHOP:
            if board_copy.is_opposite_color_bishops():
                reward += importance_scale["opposite_color_bishops"]

        # Step 17: Back-rank weaknesses (penalize for leaving a back-rank weakness)
        if move_piece == chess.PAWN:
            if board_copy.is_back_rank_weak():
                reward -= importance_scale["back_rank_weaknesses"]

        # Step 18: Isolated pawns (penalize isolated pawns)
        if move_piece == chess.PAWN:
            if board_copy.is_isolated_pawn(move.to_square):
                reward -= importance_scale["isolated_pawns"]

        # Step 19: Doubled pawns (penalize for doubled pawns)
        if move_piece == chess.PAWN:
            if board_copy.is_doubled_pawn(move.to_square):
                reward -= importance_scale["doubled_pawns"]

        # Step 20: Connected pawns (reward for connected pawns)
        if move_piece == chess.PAWN:
            if board_copy.is_connected_pawn(move.to_square):
                reward += importance_scale["connected_pawns"]

        # Step 21: Backward pawns (penalize if move creates a backward pawn)
        if move_piece == chess.PAWN:
            if board_copy.is_backward_pawn(move.to_square):
                reward -= importance_scale["backward_pawns"]

        # Step 22: Weak squares (penalize if a weak square is created)
        if move_piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
            if board_copy.is_weak_square(move.to_square):
                reward -= importance_scale["weak_squares"]

        # Step 23: En passant capture (reward for en passant)
        if board_copy.is_en_passant(move):
            reward += importance_scale["en_passant"]

        # Step 24: Attacking opponent’s weak piece (reward for attacking vulnerable pieces)
        if move_piece in [chess.QUEEN, chess.ROOK, chess.KNIGHT]:
            if board_copy.is_attacking_weak_piece(move.to_square):
                reward += importance_scale["material_gain_loss"] * 0.75

        # Step 25: Counter-attacks (reward for moves that create counter threats)
        if board_copy.is_check():
            reward += importance_scale["threats_and_tactics"] * 0.5  # Reward for counter-threats

        
        if self.is_piece_vulnerable(board_copy, move.to_square):
            captured_piece = board_copy.piece_at(move.to_square)
            if captured_piece:
                piece_type = captured_piece.piece_type
                reward -= importance_scale["material_gain_loss"] * piece_value.get(piece_type, 0.0) * 0.2

        reward -= 0.05

        return reward


    def find_best_move_with_q_values(self, board, gamma=0.99, alpha=0.1):
        global best_move_q_value
        """
        Calculate the Q-values for each legal move, update the Q-value using Q-learning,
        and return the move with the highest Q-value.

        Parameters:
            board (chess.Board): The current state of the board.
            legal_moves (list): A list of legal moves (chess.Move objects).
            reward (float): The reward for making the move.
            gamma (float): The discount factor for future rewards.
            alpha (float): The learning rate for updating Q-values.

        Returns:
            chess.Move: The move with the highest Q-value (in UCI notation).
        """
        # Initialize a list to store the Q-values for each legal move
        q_values = []

        legal_moves = list(board.legal_moves)

        rand_val = random.random()

        print("rand val: ", rand_val)
        print("epsilon val: ", self.epsilon)

        # Initialize best_move outside the if-else block
        best_move = None
        if rand_val < self.epsilon:
            # Explore: Choose a random legal move
            best_move = random.choice(legal_moves)
            print("using random vals")
            # Calculate Q-value for the random move, even though it's not used in decision making
            state_tensor = self.board_to_tensor(board).unsqueeze(0)  # Convert the current board state to tensor
            current_q_values = self.model(state_tensor).squeeze(0)  # Get the Q-values for the current state

            # Find the index of the random move in the legal_moves list
            movedex = legal_moves.index(best_move)

            # Get the Q-value for the random move
            random_move_q_value = current_q_values[movedex].item()
            print(f"Q-value of random move {best_move}: {random_move_q_value}")

            # Since it's a random move, we can assign a default Q-value (e.g., 0) for it
            best_move_q_value = random_move_q_value

            self.total_reward += self.reward_of_move(best_move, board)
        else:
            print("using q-vals")

            # Exploit: Choose the move with the highest Q-value from the model's output
            with torch.no_grad():
                # Loop through each legal move and calculate the Q-value
                for move in legal_moves:
                    # Step 1: Convert the current board state to tensor
                    state_tensor = self.board_to_tensor(board).unsqueeze(0)  # Add batch dimension

                    # Step 2: Get the Q-values for the current state
                    current_q_values = self.model(state_tensor).squeeze(0)

                    # Step 3: Simulate the move by copying the board and making the move on the clone
                    cloned_board = copy.deepcopy(board)  # Clone the board
                    cloned_board.push(move)  # Make the move on the cloned board

                    # Step 4: Convert the new state after the move to tensor
                    next_state_tensor = self.board_to_tensor(cloned_board).unsqueeze(0)  # Get the next state tensor after the move

                    # Step 5: Get the Q-values for the next state
                    next_q_values = self.model(next_state_tensor).squeeze(0)

                    # Step 6: Calculate the maximum Q-value for the next state
                    max_next_q_value = torch.max(next_q_values).item()

                    # Step 7: Find the index of the move in the model's Q-value predictions
                    move_index = legal_moves.index(move)

                    # Step 8: Get the current Q-value for the move
                    current_q_value = current_q_values[move_index].item()

                    # Step 9: Calculate the new Q-value using the Q-learning equation
                    updated_q_value = current_q_value + alpha * (self.reward_of_move(move, board) + gamma * max_next_q_value - current_q_value)

                    # Append the updated Q-value to the list
                    q_values.append(updated_q_value)

                # Convert q_values list to tensor
                q_values_tensor = torch.tensor(q_values)

                # Step 10: Calculate the loss (using MSE) and update Q-value
                # We should update the Q-value for the move that was selected
                updated_q_value_tensor = torch.tensor([updated_q_value])  # Convert updated_q_value to tensor

                # Ensure q_values_tensor is a tensor of shape (N,)
                loss = self.loss_fn(q_values_tensor, updated_q_value_tensor)

                # Update the total loss
                self.total_loss += loss.item()

                # Step 11: Find the move with the highest Q-value (from q_values)
                best_move_index = np.argmax(q_values)
                best_move = legal_moves[best_move_index]

                best_move_q_value = q_values[best_move_index]

                self.turn_reward = self.reward_of_move(best_move, board)

                self.total_reward += self.reward_of_move(best_move, board)

                self.sum_reward += self.reward_of_move(best_move, board)
                
        self.turn_q_values.append(best_move_q_value)
        

        # Decay epsilon after each move
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        return best_move