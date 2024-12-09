import chess
import tkinter as tk
from ChessAI import ChessRLAI
from ChessDQN import DQN

class ChessGame:
    """
    Class for a game of chess, with a tkinter GUI and the ability to play against a PvRLA AI.
    """

    def __init__(self, canvas, root):
        """
        Initialize a ChessGame object.

        @param canvas: A tkinter Canvas widget to draw the board on.
        @param root: A tkinter root window to display the board in.
        """
        self.canvas = canvas
        self.root = root
        self.board = chess.Board()
        self.firstClick = True
        self.uci_click = ""
        self.highlighted_square = None
        self.highlighted_legal_moves = []
        self.turn = tk.IntVar(value=0)  # 0 = White's turn, 1 = Black's turn
        self.rla_agent = ChessRLAI(model=DQN())
        self.white_pieces = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}
        self.black_pieces = {chess.PAWN: "p", chess.KNIGHT: "n", chess.BISHOP: "b", chess.ROOK: "r", chess.QUEEN: "q", chess.KING: "k"}
        self.turn_number = 0
        self.total_rewards = []
        self.game_total_rewards = []
        self.stalemates = 0
        self.white_wins = 0
        self.black_wins = 0
        self.game_numbers = [1]
        self.average_game_q_values = []
        self.average_q_value = 0.0

    # Function to handle click events
    def on_click(self, event):
        """
        Handle a click event on the board.

        @param event: A tkinter event object for the click.
        """
        col = event.x // 80
        row = event.y // 80

        # Convert to chess notation (e.g., "a1", "h8") based on the reversed orientation
        chess_col = chr(col + 97)
        chess_row = 8 - row
        clicked_square = chess_col + str(chess_row)   
        square = chess.square(col, 7 - row)

        print(f"Clicked on square: {clicked_square}")
        print(f"Square index: {square}")

        piece = self.board.piece_at(chess.square(col, 7 - row))

        # If it's the first click and the user selects a piece
        if self.firstClick:
            if piece:
                self.highlighted_legal_moves = self.get_legal_moves(self.board, clicked_square)
                self.highlighted_square = (row, col)
                self.uci_click = clicked_square
                print("uci clicked = ", self.uci_click)
                self.firstClick = False
            else:
                print("No piece at clicked square. Please select a piece.")
        else:
            # If it's the second click, check if the square is a legal move
            if square in self.highlighted_legal_moves:
                print("Legal move found!")
                self.uci_click += clicked_square
                self.makePlayerMove(self.uci_click)
                self.firstClick = True   
                self.uci_click = ""
                self.highlighted_square = None
                self.highlighted_legal_moves = []
                self.turn.set(1)
            elif piece:
                # Handle castling logic
                self.handle_castling(clicked_square)
            else:
                print("Clicked on an invalid square or no piece to move.")
                self.highlighted_square = None
                self.highlighted_legal_moves = []
                self.firstClick = True  # Go back to selecting a new piece

        # Redraw the board with the updated highlighted square and legal moves
        self.canvas.delete("all")
        self.display_board()

    def handle_castling(self, clicked_square):
        """
        Handle a click on a square that is a potential castling destination.

        @param clicked_square: The chess square that was clicked.
        """
        if self.board.has_castling_rights(chess.WHITE):
            if (clicked_square == "h1" or clicked_square == "g1") and self.uci_click == "e1":
                self.makePlayerMove("e1g1")
                self.firstClick = True
                self.highlighted_square = None
                self.highlighted_legal_moves = []
                self.turn.set(1)
            elif (clicked_square == "a1" or clicked_square == "c1")  and self.uci_click == "e1":
                self.makePlayerMove("e1c1")
                self.firstClick = True
                self.highlighted_square = None
                self.highlighted_legal_moves = []
                self.turn.set(1)
            else:
                self.firstClick = True  # Go back to selecting a new piece

    def get_legal_moves(self, board, clicked_square):
        """
        Get the legal moves for a piece on a given square.

        @param board: The current state of the board.
        @param clicked_square: The square that the piece is on.
        @return: A list of legal moves for the piece.
        """
        legal_moves = []
        moves = list(board.legal_moves)
        
        for move in moves:
            uci_move = move.uci()
            if uci_move[:2] == clicked_square:
                if uci_move in ["e1g1", "e1c1", "e8g8", "e8c8"]:
                    legal_moves.append(uci_move[2:4])  # Add the destination square for castling
                    moves.remove(move)
                else:
                    file_char = uci_move[2]
                    rank_char = uci_move[3]
                    file_index = ord(file_char) - 97
                    rank_index = int(rank_char) - 1
                    destination_square = chess.square(file_index, rank_index)
                    legal_moves.append(destination_square)
        return legal_moves

    def display_board(self):
        """
        Draw the board on the canvas.
        """
        square_size = 80
        for row in range(8):
            for col in range(8):
                color = "green" if (row + col) % 2 == 0 else "beige"
                self.canvas.create_rectangle(col * square_size, row * square_size,
                                             (col + 1) * square_size, (row + 1) * square_size,
                                             fill=color)

                if self.highlighted_square == (row, col):
                    self.canvas.create_rectangle(col * square_size, row * square_size,
                                                 (col + 1) * square_size, (row + 1) * square_size,
                                                 outline="red", width=3)

                if chess.square(col, 7 - row) in self.highlighted_legal_moves:
                    self.canvas.create_rectangle(col * square_size, row * square_size,
                                                 (col + 1) * square_size, (row + 1) * square_size,
                                                 outline="blue", width=2)

                piece = self.board.piece_at(chess.square(col, 7 - row))
                if piece:
                    piece_char = self.white_pieces.get(piece.piece_type, "") if piece.color == chess.WHITE else self.black_pieces.get(piece.piece_type, "")
                    self.canvas.create_text(col * square_size + square_size / 2,
                                           row * square_size + square_size / 2,
                                           text=piece_char, font=("Times New Roman", 36))

        # Draw row and column labels
        for row in range(8):
            self.canvas.create_text(8 * square_size + 10, row * square_size + square_size / 2,
                                    text=str(8 - row), font=("Times New Roman", 14))

        for col in range(8):
            self.canvas.create_text(col * square_size + square_size / 2,
                                    8 * square_size + 10, text=chr(ord('a') + col), font=("Times New Roman", 14))

    def makePlayerMove(self, move_uci):
        """
        Make a player move on the board.

        @param move_uci: The move in UCI format.
        """
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                print(f"Move made: {move_uci}")
                self.display_board()
            else:
                print("Invalid move!")
        except ValueError:
            print("Invalid UCI format or move.")

    def reset_game(self):
        """
        Reset the game to its initial state.
        """
        self.highlighted_square = None
        self.highlighted_legal_moves = []
        self.firstClick = True
        self.uci_click = ""
        self.turn.set(0)  # White's turn
        self.board = chess.Board()
        self.canvas.delete("all")
        self.display_board()

    def start_game(self):
        """
        Start a new game of chess.
        """
        self.reset_game()
        self.play_pvrla()

    def play_pvrla(self):
        """
        Play a game of chess against a PvRLA AI.
        """
        print("Game started")
        while not self.board.is_game_over():
            self.display_board()
            self.root.update_idletasks()

            if self.turn.get() == 0:
                print("White's Turn")
                self.canvas.bind("<Button-1>", self.on_click)
                self.root.wait_variable(self.turn)
                self.canvas.unbind("<Button-1>")

                if self.turn.get() == 0:
                    print("Turn didn't change after White's move")
                self.turn.set(1)

