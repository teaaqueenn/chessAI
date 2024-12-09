import chess
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

class ModelTrainer:
    def __init__(self, model=DQN(), loss_fn=nn.MSELoss(), lr=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = loss_fn

    def load_data(self):
        if os.path.exists('chess_games.pt'):
            games = torch.load('chess_games.pt')

            # Debug: Print structure of the first game to verify
            print("Structure of first game:", games[0])  # Should be a tuple with board tensor and move
            
            inputs = []
            targets = []
            move_to_index = {}  # To map each unique move to an index
            index = 0

            # Process the games
            for game in games:
                if isinstance(game, tuple) and len(game) == 2:
                    board_tensor, move = game  # Unpack the tuple correctly
                    inputs.append(board_tensor)

                    # Get the UCI string of the move and map it to an index
                    move_uci = move.uci()
                    if move_uci not in move_to_index:
                        move_to_index[move_uci] = index
                        index += 1
                    
                    targets.append(move_to_index[move_uci])  # Use the index of the move
                else:
                    print("Unexpected format for game:", game)

            # Convert lists to tensors
            inputs = torch.stack(inputs)
            targets = torch.tensor(targets)  # Targets are now move indices
            return inputs, targets, len(move_to_index)
        else:
            raise FileNotFoundError("chess_games.pt not found!")

    def compute_accuracy(self, outputs, targets):
        _, predicted = torch.max(outputs, 1)  # Get the index of the highest probability
        correct = (predicted == targets).sum().item()  # Count correct predictions
        accuracy = correct / len(targets)  # Compute accuracy
        return accuracy

    def compute_metrics(self, outputs, targets):
        _, predicted = torch.max(outputs, 1)
        precision = precision_score(targets.cpu(), predicted.cpu(), average='weighted', zero_division=1)
        recall = recall_score(targets.cpu(), predicted.cpu(), average='weighted', zero_division=1)
        f1 = f1_score(targets.cpu(), predicted.cpu(), average='weighted', zero_division=1)
        return precision, recall, f1

    def plot_loss_curve(self, epoch_losses):
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
        input("Press 'done' to continue to the next graph...")

    def plot_gradients(self):
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.abs().mean().item())
        plt.plot(gradients)
        plt.title('Average Gradient Magnitudes')
        plt.xlabel('Parameter Index')
        plt.ylabel('Average Gradient Magnitude')
        plt.show()
        input("Press 'done' to continue to the next graph...")

    def plot_output_distribution(self, outputs):
        probs = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        probs = probs.cpu().detach().numpy()  # Convert to numpy for plotting
        plt.hist(probs.flatten(), bins=50)  # Plot a histogram of probabilities
        plt.title('Output Probability Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.show()
        input("Press 'done' to continue to the next graph...")

    def plot_weight_histograms(self):
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                plt.clf()
                plt.hist(param.detach().cpu().numpy().flatten(), bins=50)
                plt.title(f'Weight Histogram - {name}')
                plt.xlabel('Weight Value')
                plt.ylabel('Frequency')
                plt.show()
                input("Press 'done' to continue to the next graph...")

    def plot_confusion_matrix(self, outputs, targets):
        _, predicted = torch.max(outputs, 1)
        cm = confusion_matrix(targets.cpu(), predicted.cpu())
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(cm.shape[0]), yticklabels=range(cm.shape[0]))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        input("Press 'done' to continue to the next graph...")

    def plot_metric_curve(self, metric_name, metric_values):
        plt.plot(range(1, len(metric_values) + 1), metric_values, marker='o')
        plt.title(f'{metric_name} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.show()
        input("Press 'done' to continue to the next graph...")

    def train_model(self, epochs=400, batch_size=400, lr=0.001):
        # Load data
        inputs, targets, indexMoveLength = self.load_data()

        # Create a DataLoader for batching
        dataset = data.TensorDataset(inputs, targets)
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Change to CrossEntropyLoss for classification
        self.loss_fn = nn.CrossEntropyLoss()  

        # Store metrics
        epoch_losses = []
        epoch_accuracies = []
        epoch_precisions = []
        epoch_recalls = []
        epoch_f1_scores = []

        # Training loop with tqdm for progress bars
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            epoch_accuracy = 0.0
            epoch_precision = 0.0
            epoch_recall = 0.0
            epoch_f1 = 0.0

            for i, (inputs_batch, targets_batch) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}", ncols=100, unit="batch"):
                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass through the model
                outputs = self.model(inputs_batch)

                # Ensure targets are 1D class indices (not one-hot encoded)
                targets_batch = targets_batch.view(-1)  # Flatten targets to shape (batch_size,)

                # Compute the loss
                loss = self.loss_fn(outputs, targets_batch)

                # Compute accuracy, precision, recall, and F1-score
                accuracy = self.compute_accuracy(outputs, targets_batch)
                precision, recall, f1 = self.compute_metrics(outputs, targets_batch)

                # Backpropagation
                loss.backward()

                # Optimize the model
                self.optimizer.step()

                running_loss += loss.item()
                epoch_accuracy += accuracy
                epoch_precision += precision
                epoch_recall += recall
                epoch_f1 += f1

            # Average loss and metrics for the epoch
            avg_loss = running_loss / len(dataloader)
            avg_accuracy = epoch_accuracy / len(dataloader)
            avg_precision = epoch_precision / len(dataloader)
            avg_recall = epoch_recall / len(dataloader)
            avg_f1 = epoch_f1 / len(dataloader)

            epoch_losses.append(avg_loss)
            epoch_accuracies.append(avg_accuracy)
            epoch_precisions.append(avg_precision)
            epoch_recalls.append(avg_recall)
            epoch_f1_scores.append(avg_f1)

            print(f"Epoch {epoch+1}/{epochs}, Avg. Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")

        # Optionally, plot the loss and other metrics
        plt.clf()
        self.plot_loss_curve(epoch_losses)
        plt.clf()
        self.plot_metric_curve('Accuracy', epoch_accuracies)
        plt.clf()
        self.plot_metric_curve('Precision', epoch_precisions)
        plt.clf()
        self.plot_metric_curve('Recall', epoch_recalls)
        plt.clf()
        self.plot_metric_curve('F1 Score', epoch_f1_scores)
        plt.clf()

        # Additional plots
        self.plot_gradients()
        plt.clf()
        self.plot_output_distribution(outputs)  # After training ends, use the final outputs
        plt.clf()
        self.plot_weight_histograms()
        plt.clf()

    def pretrain_and_save_model(self):
        # Pre-train the model
        self.train_model(epochs=400, batch_size=100, lr=0.001)

        # Save the trained model
        torch.save(self.model.state_dict(), 'chess_ai_model_v1.pth')
        print("Model saved as 'chess_ai_model_v1.pth'")