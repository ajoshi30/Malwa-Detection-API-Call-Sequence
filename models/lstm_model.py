import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)

        out = self.fc(hidden[-1])

        return out
