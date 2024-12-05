import chess
import chess.engine

def get_move_reward(board: chess.Board, move: chess.Move, stockfish_path: str):
    """
    Evaluates the given move on the current chess board using Stockfish.
    
    :param board: The current chess board state (chess.Board object).
    :param move: The move to evaluate (chess.Move object).
    :param stockfish_path: Path to the Stockfish engine executable.
    :return: Reward score for the move (float). Positive values favor white, negative values favor black.
    """
    
    # Make the move on the board
    board.push(move)
    
    # Start Stockfish engine
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        # Get the evaluation of the current position
        info = engine.analyse(board, chess.engine.Limit(time=3.0))  # Analyze for a small time
        evaluation = info['score']
        
        # Undo the move to return to the original state
        board.pop()
    
    # Convert evaluation to a numeric reward (if necessary)
    if evaluation.is_mate():
        # Handle checkmate as a special case
        if evaluation.mate() > 0:
            return -10000  # White wins
        elif evaluation.mate() < 0:
            return 1000  # Black wins
    else:
        return -evaluation.relative.score() / 100.0  # Convert centipawns to standard scale
    
def get_best_move(board: chess.Board, stockfish_path: str, time_limit: float = 3.0):
    """
    Determines the best move from the current board state using Stockfish.
    
    :param board: The current chess board state (chess.Board object).
    :param stockfish_path: Path to the Stockfish engine executable.
    :param time_limit: Time in seconds for Stockfish to think (default is 1.0 second).
    :return: The best move (chess.Move object).
    """
    
    # Start Stockfish engine
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        # Get the best move by analyzing the current board state
        result = engine.play(board, chess.engine.Limit(time=time_limit))
        
        # The best move chosen by Stockfish
        best_move = result.move
    
    return best_move

# Example usage
if __name__ == "__main__":
    board = chess.Board()  # Start from the initial position
    stockfish_path = r"C:\Users\Grace\Documents\GitHub\chessAI\Chess\Pre-train\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe" 
    
    best_move = get_best_move(board, stockfish_path)
    
    reward = get_move_reward(board, best_move, stockfish_path)
    print(f"Reward for the move {best_move}: {reward}")
