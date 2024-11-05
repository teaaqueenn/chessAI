import chess
import chess.svg

# Initialize the chess board
board = chess.Board()

# Function to display the board in a textual format
def display_board(loc):
    print(loc)

# Function to play the game
def play_game():
    while not board.is_game_over():
        display_board(board)

        # Get the move from the player (example: "e2e4")
        move = input("Enter your move (e.g. 'e2e4'): ")

        legal_moves = list(board.legal_moves)
        print("Legal moves:", [move.uci() for move in legal_moves])


        # Try to make the move, if invalid, prompt again
        board.push(board.push_uci(move))

    display_board(board)
    print("Game Over!")
    print("Result: " + board.result())

# Start the game
if __name__ == "__main__":
    print("Starting a new chess game!")
    play_game()
