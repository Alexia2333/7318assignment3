import numpy as np
import torch
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score, 
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt

class ModelEvaluator:
   def __init__(self, model, test_loader, device, scaler):
       self.model = model
       self.test_loader = test_loader
       self.device = device
       self.scaler = scaler

   def evaluate_predictions(self):

       self.model.eval()
       price_predictions = []
       price_targets = []
       trend_predictions = []
       trend_targets = []
           
       with torch.no_grad():
           for X_batch, y_price, y_trend in self.test_loader:
               X_batch = X_batch.to(self.device)
               price_pred, trend_pred = self.model(X_batch)
               
               # Only take first 5 features per sample
               price_pred = price_pred[:, :5] if len(price_pred.shape) > 2 else price_pred
               y_price = y_price[:, :5] if len(y_price.shape) > 2 else y_price
               
               price_predictions.append(price_pred.cpu().detach().numpy())
               price_targets.append(y_price.numpy())
               trend_predictions.append(trend_pred.cpu().detach().numpy())
               trend_targets.append(y_trend.numpy())

       price_predictions = np.vstack(price_predictions)
       price_targets = np.vstack(price_targets)
       trend_predictions = np.concatenate(trend_predictions).ravel()
       trend_predictions = (trend_predictions > 0.5).astype(int) 
       trend_targets = np.concatenate(trend_targets).ravel()
       trend_targets = (trend_targets > 0.5).astype(int)

       # Calculate metrics
       results = {
           'price_metrics': self.calculate_price_metrics(price_predictions, price_targets),
           'trend_metrics': self.calculate_trend_metrics(trend_predictions, trend_targets),
           'trading_metrics': self.simulate_trading(price_predictions, trend_predictions, price_targets)
       }

       # Visualize only closing prices
       self.visualize_predictions(
           self.scaler.inverse_transform(price_predictions)[:, 3], 
           self.scaler.inverse_transform(price_targets)[:, 3],
           f'plots/predictions_{type(self.model).__name__}.png'
       )
       
       return results

   def calculate_price_metrics(self, predictions, targets):
       # Get close prices after inverse transform
       predictions_transformed = self.scaler.inverse_transform(predictions)
       targets_transformed = self.scaler.inverse_transform(targets)
       
       close_predictions = predictions_transformed[:, 3]
       close_targets = targets_transformed[:, 3]

       # Calculate directional accuracy
       direction_accuracy = np.mean(
           (close_targets[1:] - close_targets[:-1]) * 
           (close_predictions[1:] - close_predictions[:-1]) > 0
       )

       return {
           'mse': mean_squared_error(close_targets, close_predictions),
           'mae': mean_absolute_error(close_targets, close_predictions),
           'rmse': np.sqrt(mean_squared_error(close_targets, close_predictions)),
           'mape': np.mean(np.abs((close_targets - close_predictions) / close_targets)) * 100,
           'direction_accuracy': direction_accuracy,
           'r2': r2_score(close_targets, close_predictions)
       }

   def calculate_trend_metrics(self, predictions, targets):
       
       predictions_binary = (predictions > 0.5).astype(int)
       targets_binary = (targets > 0.5).astype(int)
       return {
           'accuracy': accuracy_score(targets_binary, predictions_binary),
           'f1': f1_score(targets_binary, predictions_binary),
           'precision': precision_score(targets_binary, predictions_binary),
           'recall': recall_score(targets_binary, predictions_binary)
       }

   def simulate_trading(self, price_preds, trend_preds, true_prices,
                       initial_capital=10000, transaction_cost=0.001):
       price_preds = self.scaler.inverse_transform(price_preds)
       true_prices = self.scaler.inverse_transform(true_prices)
       
       capital = initial_capital
       position = 0  
       trades = []
       daily_returns = []
       portfolio_values = [initial_capital]

       for i in range(1, len(price_preds)):
           current_price = true_prices[i, 3]
           prev_price = true_prices[i-1, 3]
           
           if position:
               daily_return = (current_price - prev_price) / prev_price
               daily_returns.append(daily_return)
           else:
               daily_returns.append(0)

           signal = trend_preds[i] > 0.5 and price_preds[i, 3] > price_preds[i-1, 3]

           if position == 0 and signal:
               position = 1
               entry_price = current_price * (1 + transaction_cost)
               trades.append(('BUY', entry_price))
               capital -= transaction_cost * capital

           elif position == 1 and not signal:
               position = 0
               exit_price = current_price * (1 - transaction_cost)
               trades.append(('SELL', exit_price))
               capital *= (exit_price / trades[-2][1])
               capital -= transaction_cost * capital

           portfolio_values.append(capital if position == 0 else 
                                capital * (current_price / trades[-1][1]))

       daily_returns = np.array(daily_returns)
       
       if len(daily_returns) > 0 and np.std(daily_returns) > 0:
           sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
           max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - 
                               portfolio_values) / np.max(portfolio_values)
           
           return {
               'final_capital': capital,
               'total_return': ((capital - initial_capital) / initial_capital) * 100,
               'sharpe_ratio': sharpe_ratio,
               'max_drawdown': max_drawdown * 100,
               'num_trades': len(trades) // 2,
               'win_rate': self.calculate_win_rate(trades)
           }

       return {}

   def calculate_win_rate(self, trades):
       if len(trades) < 2:
           return 0
           
       profits = []
       for i in range(0, len(trades)-1, 2):
           entry = trades[i][1]
           exit = trades[i+1][1]
           profits.append(exit > entry)
       
       return np.mean(profits) * 100 if profits else 0

   def visualize_predictions(self, price_predictions, price_targets, save_path):
       plt.figure(figsize=(15, 10))
       
       plt.subplot(2, 1, 1)
       plt.plot(price_targets, 'b-', label='Actual Price', linewidth=2)
       plt.plot(price_predictions, 'r--', label='Predicted Price', linewidth=2)
       plt.title('Stock Price Prediction')
       plt.xlabel('Time Steps') 
       plt.ylabel('Price ($)')
       plt.legend()
       plt.grid(True)
       
       plt.subplot(2, 1, 2)
       errors = price_targets - price_predictions
       plt.plot(errors, 'g-', linewidth=2)
       plt.axhline(y=0, color='r', linestyle='--')
       plt.title('Prediction Error Over Time')
       plt.xlabel('Time Steps')
       plt.ylabel('Error ($)')
       plt.grid(True)
       
       plt.tight_layout()
       plt.savefig(save_path)
       plt.close()