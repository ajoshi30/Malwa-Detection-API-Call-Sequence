import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_classes):

        super(CNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.conv = nn.Conv1d(
            embed_dim,
            128,
            kernel_size=5
        )

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.embedding(x)

        x = x.permute(0, 2, 1)

        x = F.relu(self.conv(x))

        x = self.pool(x).squeeze(-1)

        out = self.fc(x)

        return out
