import ChessGame
from ChessTrainer import ModelTrainer
import tkinter as tk
import tkinter.messagebox
class ChessGameRunner:
    def __init__(self, canvas, root):
        self.canvas = canvas
        self.root = root
        self.game = ChessGame(canvas, root)  # ChessGame should be the previously defined game logic class
        self.trainer = ModelTrainer()  # Assuming ModelTrainer class exists for training purposes
        
    def start_game(self):
        """Method to start the game."""
        # Optional: Train and save the model before starting the game
        # Uncomment the following line if you want to pretrain the model before starting the game
        # self.trainer.pretrain_and_save_model()

        print("Starting Player vs RLAI mode...")
        self.game.play_pvrla()

    def run(self):
        """Main method to initiate the game."""
        self.game.reset_game()
        self.start_game()
        

# Assuming ChessGame and ModelTrainer classes are defined elsewhere as needed.
# The ChessGame class is expected to be implemented with the methods like `play_pvrla()` and `reset_game()`

# Example Usage:
# Create a Tkinter canvas and root window
import tkinter as tk

root = tk.Tk()
canvas = tk.Canvas(root, width=640, height=640)
canvas.pack()

# Create a ChessGameRunner instance and start the game
game_runner = ChessGameRunner(canvas, root)
game_runner.run()

root.mainloop()