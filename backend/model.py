import torch
import torch.nn as nn

class RNNEmotionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=8, dropout=0.3):
        super(RNNEmotionClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bi-directional LSTM
        # Input shape: (Batch, Time, Features)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # bidirectional means hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (Batch, Time, Features)
        
        # Initialize hidden and cell states (optional, defaults to 0)
        # h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :] 
        
        out = self.dropout(out)
        out = self.fc(out)
        return out
