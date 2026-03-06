import torch
import torch.nn as nn


class GRUClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        x = self.embedding(x)
        _, hidden = self.gru(x)

        out = self.fc(hidden[-1])

        return out
