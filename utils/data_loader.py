import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class StockDataset(Dataset):
    """Custom Dataset for loading stock market data"""
    def __init__(self, X, y_price, y_trend):
        self.X = torch.FloatTensor(X)
        self.y_price = torch.FloatTensor(y_price)
        self.y_trend = torch.FloatTensor(y_trend).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_price[idx], self.y_trend[idx]

def prepare_data(df, seq_length):
    """Prepare sequences and labels from dataframe"""
    try:
        print(f"Preparing data with sequence length: {seq_length}")
        print(f"Input dataframe shape: {df.shape}")

        # Scale the features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        print("Data scaled successfully")
        
        X, y_price, y_trend = [], [], []
        
        for i in range(len(df) - seq_length - 5):  # -5 for weekly trend
            # Prepare sequence data
            seq = scaled_data[i:i+seq_length]
            # Prepare next day price
            next_day = scaled_data[i+seq_length]
            # Prepare weekly trend (1 if price goes up, 0 if down)
            current_close = df.iloc[i+seq_length]['Close']
            future_close = df.iloc[i+seq_length+5]['Close']  # 5 days later
            trend = 1.0 if future_close > current_close else 0.0
            
            X.append(seq)
            y_price.append(next_day)
            y_trend.append(trend)
        
        X = np.array(X)
        y_price = np.array(y_price)
        y_trend = np.array(y_trend)
        
        print(f"Prepared data shapes:")
        print(f"X shape: {X.shape}")
        print(f"y_price shape: {y_price.shape}")
        print(f"y_trend shape: {y_trend.shape}")
        
        return X, y_price, y_trend
        
    except Exception as e:
        print(f"Error in prepare_data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def load_data():
    """Load and prepare the stock data"""
    try:
        print("Starting to load data...")
        # Load data with thousands separator
        train_df = pd.read_csv('data/Google_Stock_Price_Train.csv', thousands=',')
        print("Loaded training data successfully")
        test_df = pd.read_csv('data/Google_Stock_Price_Test.csv', thousands=',')
        print("Loaded testing data successfully")
        
        # Convert date to datetime
        train_df['Date'] = pd.to_datetime(train_df['Date'])
        test_df['Date'] = pd.to_datetime(test_df['Date'])
        print("Converted dates successfully")
        
        # Sort by date
        train_df = train_df.sort_values('Date')
        test_df = test_df.sort_values('Date')
        
        # Extract features and ensure numeric
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        train_data = train_df[features].apply(pd.to_numeric, errors='coerce')
        test_data = test_df[features].apply(pd.to_numeric, errors='coerce')
        print("Extracted features successfully")
        
        # Handle missing values
        train_data = train_data.ffill()
        test_data = test_data.ffill()
        
        print("Data loaded and processed successfully")
        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Create and fit scaler
        scaler = MinMaxScaler()
        scaler.fit(train_data)
        
        return train_data, test_data, scaler
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()  
        return None, None, None

def prepare_dataloaders(train_data, test_data, seq_length, batch_size, device):
    """Prepare train, validation, and test dataloaders"""
    print("\nPreparing dataloaders...")
    
    if train_data is None or test_data is None:
        print("Error: Input data is None")
        return None, None, None
        
    try:
        test_seq_length = min(seq_length, len(test_data) // 2)

        # Prepare sequences and labels
        print("Preparing sequences and labels...")
        X, y_price, y_trend = prepare_data(train_data, seq_length)
        print(f"Prepared data shapes - X: {X.shape}, y_price: {y_price.shape}, y_trend: {y_trend.shape}")
        
        # Split into train and validation sets
        indices = np.random.permutation(len(X))
        train_size = int(0.8 * len(indices))
        print(f"Training size: {train_size}, Validation size: {len(indices) - train_size}")
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create datasets
        print("Creating datasets...")
        train_dataset = StockDataset(
            X[train_indices], 
            y_price[train_indices], 
            y_trend[train_indices]
        )
        
        val_dataset = StockDataset(
            X[val_indices], 
            y_price[val_indices], 
            y_trend[val_indices]
        )
        
        # Create test dataset
        print("Preparing test data...")
        X_test, y_price_test, y_trend_test = prepare_data(test_data, test_seq_length)
        test_dataset = StockDataset(X_test, y_price_test, y_trend_test)
        
        print("Creating dataloaders...")
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=(device == 'cuda')
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(device == 'cuda')
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(device == 'cuda')
        )
        
        print("Dataloaders created successfully")
        return train_loader, val_loader, test_loader
    
    except Exception as e:
        print(f"Error in prepare_dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def augment_data(sequence, alpha=0.1):
    """Add random noise to sequence data"""
    noise = np.random.normal(0, alpha, sequence.shape)
    return sequence + noise



