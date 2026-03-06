import torch
import torch.nn as nn


class GRUClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):

        super(GRUClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        x = self.embedding(x)

        output, hidden = self.gru(x)

        hidden = hidden[-1]

        out = self.fc(hidden)

        return out
