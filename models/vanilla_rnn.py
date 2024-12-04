import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, price_output_size, trend_output_size, learning_rate):
        super(VanillaRNN, self).__init__()
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.price_predictor = nn.Linear(hidden_size, price_output_size)
        self.trend_predictor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, trend_output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        rnn_out, _ = self.rnn(x, h0)
        last_hidden = rnn_out[:, -1, :]
        
        price_pred = self.price_predictor(last_hidden)
        trend_pred = self.trend_predictor(last_hidden)
        
        return price_pred, trend_pred