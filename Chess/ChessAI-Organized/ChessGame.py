import chess
import tkinter as tk
from tkinter import simpledialog
from ChessAI import ChessRLAI

class ChessGame:
    """
    Class for a game of chess, with a tkinter GUI and the ability to play against a PvRLA AI.
    """
    def __init__(self, canvas, root):
        self.canvas = canvas
        self.root = root        
        self.board = chess.Board()
        self.turn = tk.IntVar(value=0)
        self.firstClick = True
        self.highlighted_square = None
        self.highlighted_legal_moves = []
        self.uci_click = ""

        self.rla_agent = ChessRLAI()
        
        # For debugging purposes, track statistics
        self.stalemates = 0
        self.white_wins = 0
        self.black_wins = 0
        self.game_numbers = [0]
        self.total_rewards = []
        self.turn_numbers = []
        self.turn_number = 0
        self.average_game_q_values = []
        self.turn_q_values = []
        self.average_q_value = 0.0
        
        self.canvas.bind("<Button-1>", self.on_click)
        
    def reset_game(self):
        self.highlighted_square = None
        self.highlighted_legal_moves = []
        self.firstClick = True
        self.uci_click = ""
        self.turn.set(0)  # White's turn
        self.board = chess.Board()
        self.canvas.delete("all")
        self.display_board()
        print("Game has been reset. It is now White's turn.")

    def on_click(self, event):
        col = event.x // 80
        row = event.y // 80
        chess_col = chr(col + 97)
        chess_row = 8 - row
        clicked_square = chess_col + str(chess_row)

        file_index = ord(chess_col) - 97
        rank_index = int(chess_row) - 1
        square = chess.square(file_index, rank_index)

        print(f"Clicked on square: {clicked_square}")
        print(f"Square index: {square}")

        piece = self.board.piece_at(chess.square(col, 7 - row))

        if self.firstClick:
            if piece:
                self.highlighted_legal_moves = self.get_legal_moves(self.board, clicked_square)
                self.highlighted_square = (row, col)
                self.uci_click = clicked_square
                print(f"Highlighted legal moves: {self.highlighted_legal_moves}")
                print("uci clicked = ", self.uci_click)
                self.firstClick = False
            else:
                print("No piece at clicked square. Please select a piece.")
        else:
            if square in self.highlighted_legal_moves:
                self.uci_click += clicked_square
                self.make_player_move(self.uci_click)
                self.firstClick = True
                self.uci_click = ""
                self.highlighted_square = None
                self.highlighted_legal_moves = []
                self.turn.set(1)
            elif piece:
                self.handle_piece_click(clicked_square, piece)
            else:
                self.highlighted_square = None
                self.highlighted_legal_moves = []
                self.firstClick = True

        self.canvas.delete("all")
        self.display_board()

    def handle_piece_click(self, clicked_square, piece):
        print(f"Clicked on: {piece} at {clicked_square}")
        if self.board.has_castling_rights(chess.WHITE):
            if (clicked_square == "h1" or clicked_square == "g1") and self.uci_click == "e1":
                self.make_player_move("e1g1")
                self.firstClick = True
                self.highlighted_square = None
                self.highlighted_legal_moves = []
                self.turn.set(1)
                return
            elif (clicked_square == "a1" or clicked_square == "c1") and self.uci_click == "e1":
                self.make_player_move("e1c1")
                self.firstClick = True
                self.highlighted_square = None
                self.highlighted_legal_moves = []
                self.turn.set(1)
                return
        self.highlighted_square = None
        self.highlighted_legal_moves = []
        self.firstClick = True

    def get_legal_moves(self, board, clicked_square):
        legal_moves = []
        moves = list(board.legal_moves)

        for move in moves:
            uci_move = move.uci()

            if uci_move[:2] == clicked_square:
                if uci_move == "e1g1" or uci_move == "e1c1" or uci_move == "e8g8" or uci_move == "e8c8":
                    destination_square_uci = uci_move[2:4]
                    if destination_square_uci == "g1":
                        destination_square_uci = "f1"
                    elif destination_square_uci == "c1":
                        destination_square_uci = "d1"
                    elif destination_square_uci == "g8":
                        destination_square_uci = "f8"
                    elif destination_square_uci == "c8":
                        destination_square_uci = "d8"
                    legal_moves.append(destination_square_uci)
                    moves.remove(move)
                else:
                    destination_square_uci = uci_move[2:4]
                    file_char = destination_square_uci[0]
                    rank_char = destination_square_uci[1]

                    file_index = ord(file_char) - 97
                    rank_index = int(rank_char) - 1

                    destination_square = chess.square(file_index, rank_index)
                    legal_moves.append(destination_square)
        return legal_moves

    def make_player_move(self, move_uci):
        try:
            move = chess.Move.from_uci(move_uci)
            piece = self.board.piece_at(move.from_square)

            if piece and piece.piece_type == chess.PAWN and (
                (move.to_square < 65 and move.to_square > 55) or (move.to_square < 8 and move.to_square > -1)
            ):
                print("promoting")
                root = tk.Tk()
                root.withdraw()
                promotion_choice = simpledialog.askstring(
                    "Pawn Promotion",
                    "Promote to (Q [Queen], R [Rook], B [Bishop], N [Knight]):",
                    parent=root,
                )

                piece_map = {'Q': chess.QUEEN, 'R': chess.ROOK, 'B': chess.BISHOP, 'N': chess.KNIGHT}
                if promotion_choice in piece_map:
                    promotion_piece = piece_map[promotion_choice]
                    move = chess.Move(move.from_square, move.to_square, promotion=promotion_piece)
                    print(f"Pawn promoting to {promotion_choice}")

            if move in self.board.legal_moves:
                self.board.push(move)
                print(f"Move made: {move_uci}")
                self.display_board()
            else:
                print("Invalid move!")
        except ValueError:
            print("Invalid UCI format or move.")

    def display_board(self):
        square_size = 80
        for row in range(8):
            for col in range(8):
                color = f'#395631' if (row + col) % 2 == 0 else f'#d1b281'
                self.canvas.create_rectangle(
                    col * square_size, row * square_size,
                    (col + 1) * square_size, (row + 1) * square_size,
                    fill=color
                )

                if self.highlighted_square == (row, col):
                    self.canvas.create_rectangle(
                        col * square_size, row * square_size,
                        (col + 1) * square_size, (row + 1) * square_size,
                        outline="red", width=3
                    )

                if chess.square(col, 7 - row) in self.highlighted_legal_moves:
                    self.canvas.create_rectangle(
                        col * square_size, row * square_size,
                        (col + 1) * square_size, (row + 1) * square_size,
                        outline="blue", width=2
                    )

                piece = self.board.piece_at(chess.square(col, 7 - row))
                if piece:
                    piece_char = ""
                    if piece.color == chess.WHITE:
                        piece_char = "♙♘♗♖♕♔"[piece.piece_type - 1]
                    else:
                        piece_char = "♟♞♝♜♛♚"[piece.piece_type - 1]
                    self.canvas.create_text(
                        col * square_size + square_size / 2,
                        row * square_size + square_size / 2,
                        text=piece_char, font=("Times New Roman", 36)
                    )

        for row in range(8):
            self.canvas.create_text(
                8 * square_size + 10, row * square_size + square_size / 2,
                text=str(8 - row), font=("Times New Roman", 14)
            )

        for col in range(8):
            self.canvas.create_text(
                col * square_size + square_size / 2,
                8 * square_size + 10,
                text=chr(ord('a') + col), font=("Times New Roman", 14)
            )

    def play_pvrla(self):
        while not self.board.is_game_over():
            self.display_board()
            self.root.update_idletasks()
            print("Legal moves: ", self.board.legal_moves)

            if self.turn.get() == 0:
                self.canvas.bind("<Button-1>", self.on_click)
                self.root.wait_variable(self.turn)
                self.canvas.unbind("<Button-1>")
                self.turn.set(1)

            elif self.turn.get() == 1:
                action = self.rla_agent.find_best_move_with_q_values(self.board)
                print(f"Black AI plays: {action}")
                self.make_player_move(action.uci())
                self.turn.set(0)
                self.root.update_idletasks()

            self.root.after(20)
        self.game_over()

    def game_over(self):
        print("Game Over!")
        print(f"Result: {self.board.result()}")
        self.reset_game()