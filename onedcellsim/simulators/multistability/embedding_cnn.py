import torch
import torch.nn as nn

class EmbeddingCNN(nn.Module):
    def __init__(self, L=151):
        super(EmbeddingCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256*L, 128),
            nn.ReLU(inplace=True)
        )

    def forward(self, trajectory):
        
        # Input shape: (batch_size, L, 3)
        batch_size, seq_len = trajectory.size()
        seq_len = int(seq_len/3)
        trajectory = trajectory.reshape(batch_size, seq_len, 3)
        trajectory = trajectory.permute(0, 2, 1)
        # Reshape to (batch_size, 3, L)
        # import matplotlib.pyplot as plt
        # plt.plot(trajectory[0, :, 0])
        # plt.plot(trajectory[0, :, 1])
        # plt.plot(trajectory[0, :, 2])
        # plt.show()
        # import sys
        # sys.exit()
        #trajectory = trajectory.permute(0, 2, 1)  # Reshape to (batch_size, 3, L)

        # Apply convolutional layers
        conv_out = self.conv_layers(trajectory)

        # Flatten the output and apply fully connected layers
        flatten = conv_out.view(batch_size, -1)
        output = self.fc_layers(flatten)
        return output