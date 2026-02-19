import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=1,
        n_layers=2,
        dropout=0.3
    ):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        # hidden shape: (n_layers * 2, batch_size, hidden_dim)
        
        # Take the last output from the sequence
        last_output = lstm_out[:, -1, :]
        # last_output shape: (batch_size, hidden_dim * 2)
        
        dropped = self.dropout(last_output)
        logits = self.fc(dropped)
        # logits shape: (batch_size, output_dim)
        
        return logits