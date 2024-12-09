from ChessGame import ChessGame
from ChessTrainer import ModelTrainer
import tkinter as tk
class ChessGameRunner:
    """
    Class to run a game of chess between a human player and a reinforcement learning AI.

    This class is the entry point for the game. It creates an instance of the ChessGame class and calls the
    `play_pvrla` method to start the game. It can optionally train and save the model before starting the game.
    """

    def __init__(self, canvas, root):
        """
        Initialize the ChessGameRunner.

        Args:
            canvas (tk.Canvas): The Tkinter canvas to draw the game on.
            root (tk.Tk): The Tkinter root window for the application.
        """
        self.canvas = canvas  # Canvas for drawing the chess board
        self.root = root      # Root window of the application
        self.game = ChessGame(canvas, root)  # Instance of the ChessGame class
        self.trainer = ModelTrainer()        # Instance of the ModelTrainer class

    def start_game(self):
        """
        Method to start the game.

        This method is called when the game is started. It can optionally train and save the model before starting the game.
        """
        # Optional: Train and save the model before starting the game
        # Uncomment the following line if you want to pretrain the model before starting the game
        # self.trainer.pretrain_and_save_model()

        # Print a message to the console to indicate that the game has started
        print("Starting Player vs RLAI mode...")
        # Call the play_pvrla method of the ChessGame class to start the game
        self.game.play_pvrla()

    def run(self):
        """
        Main method to initiate the game.

        This method is the entry point for the game. It resets the game state and starts the game by calling the
        `start_game` method.
        """
        # Reset the game state
        self.game.reset_game()
        # Start the game
        self.start_game()        

root = tk.Tk()
canvas = tk.Canvas(root, width=640, height=640)
canvas.pack()

# Create a ChessGameRunner instance and start the game
game_runner = ChessGameRunner(canvas, root)
game_runner.run()

root.mainloop()