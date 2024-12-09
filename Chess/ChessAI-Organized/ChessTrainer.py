import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from ChessDQN import DQN

class ModelTrainer:
    """
    Trains a chess AI model using the provided data.

    This class is responsible for loading the data, creating a DataLoader for batching, training the model, computing metrics, and optionally plotting the loss and other metrics.
    """

    def __init__(self, model=DQN(), loss_fn=nn.MSELoss(), lr=0.001):
        """
        Initializes the ModelTrainer.

        Args:
            model (DQN): The neural network model used to estimate Q-values.
            loss_fn (nn.MSELoss): The loss function used to compute the difference between predicted and target Q-values.
            lr (float): The learning rate for the optimizer.
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = loss_fn

    def load_data(self):
        """
        Loads the chess game data from a file.

        @return inputs (torch.Tensor): A tensor of shape (batch_size, 768) containing the input data.
        @return targets (torch.Tensor): A tensor of shape (batch_size,) containing the target data.
        @return indexMoveLength (int): The number of unique moves in the dataset.
        """
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
        """
        Computes the accuracy of the model.

        @param outputs (torch.Tensor): A tensor of shape (batch_size, 768) containing the output predictions.
        @param targets (torch.Tensor): A tensor of shape (batch_size,) containing the target data.

        @return accuracy (float): The accuracy of the model.
        """
        _, predicted = torch.max(outputs, 1)  # Get the index of the highest probability
        correct = (predicted == targets).sum().item()  # Count correct predictions
        accuracy = correct / len(targets)  # Compute accuracy
        return accuracy


    def compute_metrics(self, outputs, targets):
        """
        Computes the precision, recall, and F1-score of the model.

        @param outputs (torch.Tensor): A tensor of shape (batch_size, 768) containing the output predictions.
        @param targets (torch.Tensor): A tensor of shape (batch_size,) containing the target data.

        @return precision (float): The precision of the model.
        @return recall (float): The recall of the model.
        @return f1 (float): The F1-score of the model.
        """
        _, predicted = torch.max(outputs, 1)
        precision = precision_score(targets.cpu(), predicted.cpu(), average='weighted', zero_division=1)
        recall = recall_score(targets.cpu(), predicted.cpu(), average='weighted', zero_division=1)
        f1 = f1_score(targets.cpu(), predicted.cpu(), average='weighted', zero_division=1)
        return precision, recall, f1


    def plot_loss_curve(self, epoch_losses):
        """
        Plots the loss curve of the model.

        @param epoch_losses (list): A list of floats containing the loss values for each epoch.
        """
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
        input("Press 'done' to continue to the next graph...")


    def plot_gradients(self):
        """
        Plots the average gradient magnitudes of the model.
        """
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
        """
        Plots the output probability distribution of the model.

        @param outputs (torch.Tensor): A tensor of shape (batch_size, 768) containing the output predictions.
        """
        probs = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        probs = probs.cpu().detach().numpy()  # Convert to numpy for plotting
        plt.hist(probs.flatten(), bins=50)  # Plot a histogram of probabilities
        plt.title('Output Probability Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.show()
        input("Press 'done' to continue to the next graph...")


    def plot_weight_histograms(self):
        """
        Plots the weight histograms of the model.
        """
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
        """
        Plots the confusion matrix of the model.

        @param outputs (torch.Tensor): A tensor of shape (batch_size, 768) containing the output predictions.
        @param targets (torch.Tensor): A tensor of shape (batch_size,) containing the target data.
        """
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
        """
        Plots the metric curve of the model.

        @param metric_name (str): The name of the metric (e.g. accuracy, precision, recall, F1-score).
        @param metric_values (list): A list of floats containing the metric values for each epoch.
        """
        plt.plot(range(1, len(metric_values) + 1), metric_values, marker='o')
        plt.title(f'{metric_name} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.show()
        input("Press 'done' to continue to the next graph...")


    def train_model(self, epochs=400, batch_size=400, lr=0.001):
        """
        Trains the model using the provided data.

        @param epochs (int): The number of epochs to train the model for.
        @param batch_size (int): The batch size to use for training.
        @param lr (float): The learning rate for the optimizer.
        """
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
        """
        Pre-train the model and save it to a file.

        The model is trained for 400 epochs with a batch size of 100 and a learning rate of 0.001.
        The trained model is then saved to a file named 'chess_ai_model_v1.pth'.
        """
        # Pre-train the model
        self.train_model(epochs=400, batch_size=100, lr=0.001)

        # Save the trained model
        torch.save(self.model.state_dict(), 'chess_ai_model_v1.pth')
        print("Model saved as 'chess_ai_model_v1.pth'")
        