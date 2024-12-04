import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, price_output_size, trend_output_size, learning_rate):
        super(Generator, self).__init__()
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define the generator network without the initial Linear layer
        self.main = None

        # Define outputs
        self.price_predictor = nn.Linear(hidden_size * 2, price_output_size)
        self.trend_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(32, trend_output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1)
        flattened_input_size = sequence_length * self.input_size

        # Dynamically define the main network if it hasn't been defined yet
        if self.main is None or self.main[0].in_features != flattened_input_size:
            self.main = nn.Sequential(
                nn.Linear(flattened_input_size, self.hidden_size),
                nn.LeakyReLU(0.2),
                nn.Linear(self.hidden_size, self.hidden_size * 2),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ).to(x.device)

        # Flatten the input to match the expected shape for Linear layers
        x = x.view(batch_size, -1)

        # Pass through the main network
        features = self.main(x)

        # Get the price and trend predictions
        price_pred = self.price_predictor(features)
        trend_pred = self.trend_predictor(features)

        return price_pred, trend_pred


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, learning_rate):
        super(Discriminator, self).__init__()
        self.learning_rate = learning_rate
        self.input_size = input_size

        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1) 
        return self.main(x)