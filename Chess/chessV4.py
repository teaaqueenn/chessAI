import gym
import gym_chess
import random
import time
import tkinter as tk  # Import tkinter for creating the GUI window
from tkinter import Label

# Create the chess environment
env = gym.make('Chess-v0')

# Reset the environment (start a new game)
env.reset()

# Set up the tkinter window
window = tk.Tk()
window.title("Chess Game")  # Title of the window

# Set the window size (width x height in pixels)
window.geometry("400x400")  # You can adjust these values

# A label to display the chessboard (we'll update this label each time)
# Increase font size and label dimensions
chessboard_label = Label(window, font=('Courier', 24), padx=10, pady=10, width=15, height=10)
chessboard_label.pack()

done = False

def update_chessboard():
    """Update the chessboard display in the tkinter window."""
    chessboard_label.config(text=env.render(mode='unicode'))  # Set the chessboard text
    window.update_idletasks()  # Update the window tasks
    window.update()  # Refresh the window

# Run a loop until the game is over
while not done:
    # Choose a random legal move
    action = random.choice(env.legal_moves)  # Use random.choice() to pick a move
    
    # Apply the action to the environment
    next_state, reward, done, info = env.step(action)
    
    # Update the chessboard in the tkinter window
    update_chessboard()
    
    # Introduce a 1-second delay between moves
    time.sleep(1)

# Close the environment and tkinter window when done
env.close()
window.destroy()
