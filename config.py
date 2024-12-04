import torch
import os

# GPU settings
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")  
    BATCH_SIZE = 128  # Larger batch size for GPU
    PIN_MEMORY = True
else:
    DEVICE = torch.device("cpu")
    BATCH_SIZE = 32   # Smaller batch size for CPU
    PIN_MEMORY = False

# Common model parameters
COMMON_MODEL_PARAMS = {
    'input_size': 5,  # Open, High, Low, Close, Volume
    'hidden_size': 128,
    'num_layers': 2,
    'output_size': 5
}

CONFIG = {
    'data_config': {
        'batch_size': BATCH_SIZE,
        'num_workers': 4,
        'pin_memory': PIN_MEMORY,
        'seq_length': 30,        # Length of input sequence
        'pred_day': 1,          # For next day price prediction
        'pred_week': 5,         # For weekly trend prediction
        'train_val_split': 0.8
    },
    
    # Training configuration
    'train_config': {
        'epochs': 100,
        'weight_decay': 1e-4,
        'device': DEVICE,
        'gradient_clip': 0.5,
        # Learning rate scheduler
        'scheduler': {
            'type': 'ReduceLROnPlateau',
            'factor': 0.1,
            'patience': 10,
            'verbose': True
        },
        # Early stopping
        'early_stopping': {
            'patience': 20,
            'min_delta': 0.001
        }
    },
    
    # Model configurations
    'model_config': {
        # Vanilla RNN
        'rnn': {
            'input_size': 5,
            'hidden_size': 128,
            'num_layers': 2,
            'price_output_size': 5,
            'trend_output_size': 1,
            'learning_rate': 0.0001
        },
        # LSTM
        'lstm': {
            'input_size': 5,
            'hidden_size': 64,
            'num_layers': 3,
            'price_output_size': 5,
            'trend_output_size': 1,
            'learning_rate': 0.001
        },
        # GAN
        'gan': {
            'generator': {
                'input_size': 5,  
                'hidden_size': 128,
                'sequence_length': 30,
                'price_output_size': 5,
                'trend_output_size': 1,
                'learning_rate': 0.001
            },
            'discriminator': {
                'input_size': 5,
                'hidden_size': 64,
                'learning_rate': 0.0002
            }
        }
    },
    
    # Paths configuration
    'paths': {
        'data_dir': './data',
        'save_dir': './checkpoints',
        'log_dir': './logs',
        'results_dir': './results',
        'model_names': {
            'rnn': 'rnn_model.pth',
            'lstm': 'lstm_model.pth',
            'gan': 'gan_model.pth'
        }
    },
    
    # Logging configuration
    'logging_config': {
        'log_interval': 10,  # Log every N batches
        'save_interval': 5,  # Save model every N epochs
        'eval_interval': 1,  # Evaluate on validation set every N epochs
    }
}

# Create necessary directories
for dir_path in CONFIG['paths'].values():
    if isinstance(dir_path, str) and not os.path.exists(dir_path):
        os.makedirs(dir_path)