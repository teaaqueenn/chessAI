import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    A Deep Q-Network (DQN) implemented with PyTorch.

    This class represents a DQN model with a very deep neural network architecture (20 layers) to estimate
    the Q-values of a given state in a chess game. The Q-values are computed by passing the input through all
    the layers with ReLU activation.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer with 2048 units.
        -->
        fc20 (nn.Linear): The output layer with 4672 units (for possible moves).

    Methods:
        forward(self, x): Computes the Q-values of a given state by passing the input through all the layers with ReLU activation.
    """

    def __init__(self):
        """
        Initializes the DQN model.

        Initializes the DQN model with a very deep neural network architecture (20 layers) to estimate
        the Q-values of a given state in a chess game.
        """
        super(DQN, self).__init__()
        
        # Input layer: 12 * 8 * 8 = 768 (flattened input)
        self.fc1 = nn.Linear(12 * 8 * 8, 2048)  # First hidden layer with 2048 units
        self.fc2 = nn.Linear(2048, 2048)        # Second hidden layer with 2048 units
        self.fc3 = nn.Linear(2048, 1024)        # Third hidden layer with 1024 units
        self.fc4 = nn.Linear(1024, 1024)        # Fourth hidden layer with 1024 units
        self.fc5 = nn.Linear(1024, 512)         # Fifth hidden layer with 512 units
        self.fc6 = nn.Linear(512, 512)          # Sixth hidden layer with 512 units
        self.fc7 = nn.Linear(512, 256)          # Seventh hidden layer with 256 units
        self.fc8 = nn.Linear(256, 256)          # Eighth hidden layer with 256 units
        self.fc9 = nn.Linear(256, 128)          # Ninth hidden layer with 128 units
        self.fc10 = nn.Linear(128, 128)         # Tenth hidden layer with 128 units
        self.fc11 = nn.Linear(128, 64)          # Eleventh hidden layer with 64 units
        self.fc12 = nn.Linear(64, 64)           # Twelfth hidden layer with 64 units
        self.fc13 = nn.Linear(64, 32)           # Thirteenth hidden layer with 32 units
        self.fc14 = nn.Linear(32, 32)           # Fourteenth hidden layer with 32 units
        self.fc15 = nn.Linear(32, 16)           # Fifteenth hidden layer with 16 units
        self.fc16 = nn.Linear(16, 16)           # Sixteenth hidden layer with 16 units
        self.fc17 = nn.Linear(16, 8)            # Seventeenth hidden layer with 8 units
        self.fc18 = nn.Linear(8, 8)             # Eighteenth hidden layer with 8 units
        self.fc19 = nn.Linear(8, 4)             # Nineteenth hidden layer with 4 units
        self.fc20 = nn.Linear(4, 4672)          # Output layer with 4672 units (for possible moves)

        # Dropout layer to prevent overfitting (applied after each hidden layer)
        self.dropout = nn.Dropout(0.3)  # Dropout with 30% probability of zeroing out inputs

    def forward(self, x):
        """
        Computes the Q-values of a given state by passing the input through all the layers with ReLU activation.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 768) containing the flattened chess board state.

        Returns:
            x (torch.Tensor): The output tensor of shape (batch_size, 4672) containing the estimated Q-values.
        """
        # Passing the input through all the layers with ReLU activation
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        x = torch.relu(self.fc3(x))
        x = self.dropout(x)

        x = torch.relu(self.fc4(x))
        x = self.dropout(x)

        x = torch.relu(self.fc5(x))
        x = self.dropout(x)

        x = torch.relu(self.fc6(x))
        x = self.dropout(x)

        x = torch.relu(self.fc7(x))
        x = self.dropout(x)

        x = torch.relu(self.fc8(x))
        x = self.dropout(x)

        x = torch.relu(self.fc9(x))
        x = self.dropout(x)

        x = torch.relu(self.fc10(x))
        x = self.dropout(x)

        x = torch.relu(self.fc11(x))
        x = self.dropout(x)

        x = torch.relu(self.fc12(x))
        x = self.dropout(x)

        x = torch.relu(self.fc13(x))
        x = self.dropout(x)

        x = torch.relu(self.fc14(x))
        x = self.dropout(x)

        x = torch.relu(self.fc15(x))
        x = self.dropout(x)

        x = torch.relu(self.fc16(x))
        x = self.dropout(x)

        x = torch.relu(self.fc17(x))
        x = self.dropout(x)

        x = torch.relu(self.fc18(x))
        x = self.dropout(x)

        x = self.fc19(x)  # No activation for the last hidden layer
        x = self.fc20(x)  # Output layer with dynamic action space size
        return x

