import chess

blackScore = 0
whiteScore = 0

# Define piece values (positive for white, negative for black)
piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 4,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # King is never captured in normal play
}


# Initialize the chess board
board = chess.Board()

# Function to display the board in a textual format
def display_board(board):
    print(board)

# Function to calculate the total captured piece values
def calculate_captured_score(board, previous_board):
    white_score = 0
    black_score = 0
    # Loop over all moves in the previous board
    for move in board.move_stack:
        if move in previous_board.move_stack:
            continue  # Skip moves that are already in the previous board
        piece = board.piece_at(move.to_square)
        if piece:
            # Add the value of captured pieces to the appropriate player
            if piece.color == chess.WHITE:
                black_score += piece_values.get(piece.piece_type, 0)
            elif piece.color == chess.BLACK:
                white_score += piece_values.get(piece.piece_type, 0)
    
    return white_score, black_score


# Function to update and print the rewards based on captured pieces
def get_move_reward(board, previous_board):
    white_score, black_score = calculate_captured_score(board, previous_board)  # Fix: pass previous_board

    whiteScore += white_score
    black_score += black_score
    
    # Calculate the change in score (capture by either side)
    if white_score > black_score:
        reward = f"White captured pieces worth {white_score} points!"
    elif black_score > white_score:
        reward = f"Black captured pieces worth {black_score} points!"
    else:
        reward = "No pieces captured this move."
    
    return reward

# Function to play the game
def play_game():
    while not board.is_game_over():
        display_board(board)

        legal_moves = board.legal_moves

        # Convert each legal move to UCI notation and print
        legal_moves_uci = [move.uci() for move in legal_moves]
        print("legal moves are:", legal_moves_uci)

        # Get the move from the player (example: "e2e4")
        move = input("Enter your move (e.g. 'e2e4'): ")

        # Try to make the move, if invalid, prompt again
        try:
            # Apply the move using UCI notation
            previous_board = board.copy()  # Save a copy of the board before the move
            board.push_uci(move)

            # Check if a capture occurred and update reward
            reward = get_move_reward(board, previous_board)
            print(reward)

        except ValueError:
            print("Invalid move! Please try again.")

    display_board(board)
    print("Game Over!")
    print("Blacks Total Reward is: " + blackScore)
    print("Blacks Total Reward is: " + whiteScore)
    print("Result: " + board.result())

# Start the game
if __name__ == "__main__":
    print("Starting a new chess game!")
    play_game()
