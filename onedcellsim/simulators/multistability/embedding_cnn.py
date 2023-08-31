import torch
import torch.nn as nn

class EmbeddingCNN(nn.Module):
    def __init__(self, L=151, nchan=3):
        super(EmbeddingCNN, self).__init__()
        self.L = L
        self.nchan = nchan

        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.nchan, 64, kernel_size=3, stride=1, padding=1),
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
        seq_len = int(seq_len/self.nchan)
        trajectory = trajectory.reshape(batch_size, seq_len, self.nchan)
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

class EmbeddingCNN_2D(nn.Module):
    def __init__(self, L=151, W=3):
        super(EmbeddingCNN_2D, self).__init__()
        self.L = L
        self.W = W
        self.kernel_length=3

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(self.W, self.kernel_length), stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(self.W, self.kernel_length), stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(self.W, self.kernel_length), stride=1, padding='same'),
            nn.ReLU(inplace=True),
        )

        self.fc_layers = nn.Sequential(
            #npoints = self.L -int((self.W-self.kernel_length)*2)
            nn.Linear(256*self.W*self.L, 128),
            #nn.Linear(256*(L-6)*W, 128),
            nn.ReLU(inplace=True)
        )

    def forward(self, trajectory):
        
        # Input shape: (batch_size, L, 3)
        batch_size, seq_len = trajectory.size()
        seq_len = int(seq_len/self.W)
        trajectory = trajectory.reshape(batch_size, 1, seq_len, self.W)
        #trajectory = trajectory.permute(0, 1, 3, 2)
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
        #print(conv_out.shape, (self.L-6), self.W, 256*(self.L-6))
        # Flatten the output and apply fully connected layers
        flatten = conv_out.view(batch_size, -1)
        output = self.fc_layers(flatten)
        return output