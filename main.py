import os
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

from utils.data_loader import load_data, prepare_dataloaders
from utils.evaluation import ModelEvaluator

from models.vanilla_rnn import VanillaRNN
from models.lstm_model import StockLSTM
from models.gan_model import Generator, Discriminator

from config import CONFIG

def train_model(model, train_loader, val_loader, optimizer, num_epochs, model_name, device):
    # Define loss functions
    price_criterion = nn.MSELoss()
    trend_criterion = nn.BCELoss()
    
    # Loss weights
    price_weight = 0.7
    trend_weight = 0.3
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):

        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        # Training phase
        model.train()
        epoch_train_loss = 0
        num_train_batches = 0
        
        batch_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        for X_batch, y_price, y_trend in batch_bar:
            X_batch = X_batch.to(device)
            y_price = y_price.to(device)
            y_trend = y_trend.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            price_pred, trend_pred = model(X_batch)
            
            # Calculate losses
            price_loss = price_criterion(price_pred, y_price)
            trend_loss = trend_criterion(trend_pred, y_trend)
            
            # Combined loss
            loss = price_weight * price_loss + trend_weight * trend_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_train_batches += 1
            batch_bar.set_postfix(
                loss=f'{loss.item():.4f}',
                price_loss=f'{price_loss.item():.4f}',
                trend_loss=f'{trend_loss.item():.4f}'
            )
        
        # Calculate average training loss
        avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_price, y_trend in val_loader:
                X_batch = X_batch.to(device)
                y_price = y_price.to(device)
                y_trend = y_trend.to(device)
                
                price_pred, trend_pred = model(X_batch)
                
                price_loss = price_criterion(price_pred, y_price)
                trend_loss = trend_criterion(trend_pred, y_trend)
                loss = price_weight * price_loss + trend_weight * trend_loss
                
                epoch_val_loss += loss.item()
                num_val_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        

        print(f'Average Train Loss: {avg_train_loss:.4f}')
        print(f'Average Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'models/best_{model_name.lower()}_model.pth')
            print(f'Saved best model with validation loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

def train_gan(generator, discriminator, train_loader, num_epochs, device):
   g_price_criterion = nn.MSELoss()
   g_trend_criterion = nn.BCELoss()
   d_criterion = nn.BCELoss()

   g_losses = []
   d_losses = []
   
   g_optimizer = torch.optim.Adam(generator.parameters(), lr=CONFIG['model_config']['gan']['generator']['learning_rate'])
   d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=CONFIG['model_config']['gan']['discriminator']['learning_rate'])

   for epoch in range(num_epochs):
       epoch_g_loss = 0
       epoch_d_loss = 0
       batch_count = 0

       print(f'Epoch [{epoch+1}/{num_epochs}]')

       for X_batch, y_price, y_trend in train_loader:
           batch_size = X_batch.size(0)
           X_batch = X_batch.to(device)
           y_price = y_price.to(device)
           y_trend = y_trend.to(device).view(batch_size, -1)
           
           # Train Discriminator
           d_optimizer.zero_grad()
           
           real_labels = torch.ones(batch_size, 1).to(device)
           fake_labels = torch.zeros(batch_size, 1).to(device)
           
           # Generate fake data
           price_pred, trend_pred = generator(X_batch)
           trend_pred = trend_pred.view(batch_size, -1)
           fake_data = torch.cat([price_pred, trend_pred], dim=1)
           
           # Train with real and fake data
           y_trend = y_trend.squeeze().unsqueeze(1)
           real_data = torch.cat([y_price, y_trend], dim=1)
           d_real_loss = d_criterion(discriminator(real_data), real_labels)
           d_fake_loss = d_criterion(discriminator(fake_data.detach()), fake_labels)
           d_loss = (d_real_loss + d_fake_loss) / 2
           d_loss.backward()
           d_optimizer.step()
           
           # Train Generator 
           g_optimizer.zero_grad()
           
           # Generate new predictions
           price_pred, trend_pred = generator(X_batch)
           trend_pred = trend_pred.view(batch_size, -1)
           fake_data = torch.cat([price_pred, trend_pred], dim=1)
           
           # Calculate generator losses
           g_price_loss = g_price_criterion(price_pred, y_price)
           g_trend_loss = g_trend_criterion(trend_pred, y_trend)
           g_adv_loss = d_criterion(discriminator(fake_data), real_labels)
           
           g_loss = 0.4 * g_price_loss + 0.3 * g_trend_loss + 0.3 * g_adv_loss
           g_loss.backward()
           g_optimizer.step()

           epoch_g_loss += g_loss.item()
           epoch_d_loss += d_loss.item()
           batch_count += 1

       # Record average losses for this epoch
       avg_g_loss = epoch_g_loss / batch_count
       avg_d_loss = epoch_d_loss / batch_count
       g_losses.append(avg_g_loss)
       d_losses.append(avg_d_loss)
       

       print(f'D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}')
       print(f'Price Loss: {g_price_loss.item():.4f}, Trend Loss: {g_trend_loss.item():.4f}')

   return g_losses, d_losses

def get_lr_scheduler(optimizer):
    """Get learning rate scheduler"""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

def evaluate_model(model_name, model, test_loader, device, scaler):
    """Evaluate a trained model"""
    evaluator = ModelEvaluator(model, test_loader, device, scaler)

    for X_batch, _, _ in test_loader:

        if isinstance(model, Generator):
            if X_batch.size(1) < 30:
                padding_length = 30 - X_batch.size(1)
                pad = torch.zeros((X_batch.size(0), padding_length, X_batch.size(2))).to(X_batch.device)
                X_batch = torch.cat([X_batch, pad], dim=1)
        
        results = evaluator.evaluate_predictions()

    print(f"\nEvaluation Results for {model_name}:")
    print("\nPrice Prediction Metrics:")
    for metric, value in results['price_metrics'].items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("\nTrend Prediction Metrics:")
    for metric, value in results['trend_metrics'].items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("\nTrading Simulation Results:")
    for metric, value in results['trading_metrics'].items():
        if isinstance(value, float):
            print(f"{metric}: {value:.2f}")
        else:
            print(f"{metric}: {value}")
    
    return results




def plot_results(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 6))

    if model_name == 'GAN':
        plt.plot(train_losses, label='Generator Loss')
        plt.plot(val_losses, label='Discriminator Loss')
    else:
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training Results')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{model_name}_results.png')
    print(f"{model_name} training results saved as 'plots/{model_name}_results.png'")
    plt.close()

def plot_models_comparison(models_results):
   plt.figure(figsize=(12, 6))
   
   max_epochs = 0
   for results in models_results.values():
       if 'train_losses' in results:
           max_epochs = max(max_epochs, len(results['train_losses']))
       elif 'g_losses' in results:
           max_epochs = max(max_epochs, len(results['g_losses']))
   
   x_axis = range(max_epochs)
   csv_data = []
   
   for model_name, results in models_results.items():
       if 'train_losses' in results and 'val_losses' in results:
           plt.plot(x_axis[:len(results['train_losses'])], 
                   results['train_losses'], 
                   label=f'{model_name} Train')
           plt.plot(x_axis[:len(results['val_losses'])], 
                   results['val_losses'], 
                   label=f'{model_name} Val')
           
           for epoch in range(len(results['train_losses'])):
               row = {
                   'model_name': model_name,
                   'epoch': epoch + 1,
                   'train_loss': results['train_losses'][epoch],
                   'val_loss': results['val_losses'][epoch],
                   'g_loss': '',
                   'd_loss': ''
               }
               csv_data.append(row)
               
       elif 'g_losses' in results and 'd_losses' in results:
           plt.plot(x_axis[:len(results['g_losses'])], 
                   results['g_losses'], 
                   label=f'{model_name} Generator Loss')
           plt.plot(x_axis[:len(results['d_losses'])], 
                   results['d_losses'], 
                   label=f'{model_name} Discriminator Loss')
           
           for epoch in range(len(results['g_losses'])):
               row = {
                   'model_name': model_name,
                   'epoch': epoch + 1,
                   'train_loss': '',
                   'val_loss': '',
                   'g_loss': results['g_losses'][epoch],
                   'd_loss': results['d_losses'][epoch]
               }
               csv_data.append(row)
   
   plt.title('Models Comparison')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()
   plt.grid(True)
   plt.savefig('plots/models_comparison.png')
   print("Models comparison plot saved as 'plots/models_comparison.png'")
   plt.close()
   csv_file_path = 'plots/models_comparison_data.csv'
   with open(csv_file_path, mode='w', newline='') as csv_file:
       fieldnames = ['model_name', 'epoch', 'train_loss', 'val_loss', 'g_loss', 'd_loss']
       writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
       writer.writeheader()
       for row in csv_data:
           writer.writerow(row)
   
   print(f"Models comparison data saved as '{csv_file_path}'")

def main():
    device = CONFIG['train_config']['device']
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    # Data loading
    train_data, test_data, scaler = load_data()
    if train_data is None:
        print("Failed to load data")
        return
        
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_data=train_data,
        test_data=test_data,
        seq_length=CONFIG['data_config']['seq_length'],
        batch_size=CONFIG['data_config']['batch_size'],
        device=CONFIG['train_config']['device']
    )
    
    results = {}
    
    # 1. Train RNN
    print("Training RNN model...")
    rnn_model = VanillaRNN(**CONFIG['model_config']['rnn']).to(device)
    optimizer = torch.optim.Adam(
        rnn_model.parameters(),
        lr=CONFIG['model_config']['rnn']['learning_rate'],
        weight_decay=CONFIG['train_config']['weight_decay']
    )
    rnn_train_losses, rnn_val_losses = train_model(
        model=rnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=CONFIG['train_config']['epochs'],
        model_name='rnn',
        device=device
    )
    results['rnn'] = {'train_losses': rnn_train_losses, 'val_losses': rnn_val_losses}

    print("\nEvaluating RNN model...")
    rnn_results = evaluate_model(
        model_name='RNN',
        model=rnn_model,
        test_loader=test_loader,
        device=device,
        scaler=scaler
    )
    results['rnn'].update(rnn_results)
    
    # 2. Train LSTM
    print("\nTraining LSTM model...")
    lstm_model = StockLSTM(**CONFIG['model_config']['lstm']).to(device)
    optimizer = torch.optim.Adam(
        lstm_model.parameters(),
        lr=CONFIG['model_config']['lstm']['learning_rate'], 
        weight_decay=CONFIG['train_config']['weight_decay']
    )
    lstm_train_losses, lstm_val_losses = train_model(
        model=lstm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=CONFIG['train_config']['epochs'],
        model_name='lstm',
        device=device
    )
    results['lstm'] = {'train_losses': lstm_train_losses, 'val_losses': lstm_val_losses}

    print("\nEvaluating LSTM model...")
    lstm_results = evaluate_model(
        model_name='LSTM',
        model=lstm_model,
        test_loader=test_loader,
        device=device,
        scaler=scaler
    )
    results['lstm'].update(lstm_results)
    
    # 3. Train GAN
    print("\nTraining GAN model...")
    discriminator_params = CONFIG['model_config']['gan']['discriminator']
    discriminator = Discriminator(
        input_size=6,  
        hidden_size=discriminator_params['hidden_size'],
        learning_rate=discriminator_params['learning_rate']
    ).to(device)

    generator_params = CONFIG['model_config']['gan']['generator']
    generator = Generator(
        input_size=generator_params['input_size'],
        hidden_size=generator_params['hidden_size'],
        sequence_length=CONFIG['data_config']['seq_length'],  
        price_output_size=5,  
        trend_output_size=1,  
         learning_rate=generator_params['learning_rate']
    ).to(device)

    g_losses, d_losses = train_gan(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        num_epochs=CONFIG['train_config']['epochs'],
        device=device
    )
    results['gan'] = {'g_losses': g_losses, 'd_losses': d_losses}

    print("\nEvaluating GAN model...")
    gan_results = evaluate_model(
        model_name='GAN',
        model=generator,  
        test_loader=test_loader,
        device=device,
        scaler=scaler
    )
    
    results['gan'].update(gan_results)

    
    # Plot and compare results
    if not results:
        print("No training results available. The 'results' dictionary is empty.")
    else:
        plot_models_comparison(results)

    for model_name, model_results in results.items():
        print(f"\nEvaluation Results for {model_name.upper()}:")
        if 'price_metrics' in model_results:
            print("\nPrice Prediction Metrics:")
            for metric, value in model_results['price_metrics'].items():
                print(f"{metric.upper()}: {value:.4f}")
        if 'trend_metrics' in model_results:
            print("\nTrend Prediction Metrics:")
            for metric, value in model_results['trend_metrics'].items():
                print(f"{metric.upper()}: {value:.4f}")
        if 'trading_metrics' in model_results:
            print("\nTrading Simulation Results:")
            for metric, value in model_results['trading_metrics'].items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.2f}")
                else:
                    print(f"{metric}: {value}")

    # Save evaluation results to CSV
    import csv
    csv_file_path = os.path.join(CONFIG['paths']['results_dir'], 'evaluation_results.csv')
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = ['model_name', 'metric', 'value']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for model_name, model_results in results.items():
            for key, metrics in model_results.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        writer.writerow({
                            'model_name': model_name,
                            'metric': metric,
                            'value': value
                        })

    print(f"\nEvaluation results have been saved to '{csv_file_path}'.")


if __name__ == "__main__":
    main()