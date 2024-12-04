import torch
import torch.nn as nn

class StockLSTM(nn.Module):
   def __init__(self, input_size, hidden_size, num_layers, price_output_size, trend_output_size, learning_rate):
       super(StockLSTM, self).__init__()
       self.learning_rate = learning_rate
       self.hidden_size = hidden_size
       self.num_layers = num_layers
       
       self.lstm1 = nn.LSTM(input_size, hidden_size*2, num_layers, batch_first=True, dropout=0.2)
       self.lstm2 = nn.LSTM(hidden_size*2, hidden_size, num_layers, batch_first=True, dropout=0.2)
       
       self.attention = nn.MultiheadAttention(hidden_size, 4)
       
       self.price_predictor = nn.Sequential(
           nn.Linear(hidden_size, hidden_size),
           nn.ReLU(),
           nn.Dropout(0.2),
           nn.Linear(hidden_size, hidden_size//2),
           nn.ReLU(),
           nn.Dropout(0.2),
           nn.Linear(hidden_size//2, price_output_size),
           nn.ReLU()
       )
       
       self.trend_predictor = nn.Sequential(
           nn.Linear(hidden_size, hidden_size),
           nn.ReLU(),
           nn.Dropout(0.2),
           nn.Linear(hidden_size, hidden_size//2),
           nn.ReLU(),
           nn.Dropout(0.2),
           nn.Linear(hidden_size//2, trend_output_size),
           nn.Sigmoid()
       )
       
       self.price_bn = nn.BatchNorm1d(hidden_size)
       self.trend_bn = nn.BatchNorm1d(hidden_size)
       
   def forward(self, x):
       batch_size = x.size(0)
       
       h0_1 = torch.zeros(self.num_layers, batch_size, self.hidden_size*2).to(x.device)
       c0_1 = torch.zeros(self.num_layers, batch_size, self.hidden_size*2).to(x.device)
       
       h0_2 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
       c0_2 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
       
       lstm_out1, _ = self.lstm1(x, (h0_1, c0_1))
       lstm_out2, _ = self.lstm2(lstm_out1, (h0_2, c0_2))
       
       lstm_out2 = lstm_out2.permute(1, 0, 2)
       attn_out, _ = self.attention(lstm_out2, lstm_out2, lstm_out2)
       attn_out = attn_out.permute(1, 0, 2)
       
       last_out = attn_out[:, -1, :]
       
       if self.training:
           price_features = self.price_bn(last_out)
           trend_features = self.trend_bn(last_out)
       else:
           price_features = last_out
           trend_features = last_out
       
       price_pred = self.price_predictor(price_features)
       trend_pred = self.trend_predictor(trend_features)
       
       return price_pred, trend_pred