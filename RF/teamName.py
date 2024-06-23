import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

nInst = 50
currentPos = np.zeros(nInst)

class OptimizedTradingModel:
    def __init__(self, n_instruments):
        self.n_instruments = n_instruments
        self.models = [RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=8, min_samples_split=14, random_state=42) 
                       for _ in range(n_instruments)]
        self.scalers = [StandardScaler() for _ in range(n_instruments)]
        self.is_trained = False

    def preprocess_data(self, prices):
        features = []
        for i in range(self.n_instruments):
            df = prices[i]
            returns = np.diff(np.log(df))
            
            # Technical indicators
            sma_5 = np.convolve(df, np.ones(5), 'valid') / 5
            sma_10 = np.convolve(df, np.ones(10), 'valid') / 10
            sma_20 = np.convolve(df, np.ones(20), 'valid') / 20
            rsi = self.compute_rsi(df)
            
            # Ensure all features have the same length
            feature_length = min(len(returns), len(sma_20), len(rsi[20:]))
            
            # Combine features
            inst_features = np.column_stack([
                returns[-feature_length:],
                (df[-feature_length:] - sma_5[-feature_length:]) / sma_5[-feature_length:],
                (df[-feature_length:] - sma_10[-feature_length:]) / sma_10[-feature_length:],
                (df[-feature_length:] - sma_20[-feature_length:]) / sma_20[-feature_length:],
                rsi[-feature_length:],
                np.log(df[-feature_length:] / df[-feature_length-1:-1]),
                np.log(df[-feature_length:] / df[-feature_length-5:-5]),
                np.log(df[-feature_length:] / df[-feature_length-20:-20]),
            ])
            features.append(inst_features)
        return features

    @staticmethod
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)
        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            up = (up*(period - 1) + upval)/period
            down = (down*(period - 1) + downval)/period
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)
        return rsi

    def train(self, prices):
        features = self.preprocess_data(prices)
        for i in range(self.n_instruments):
            X = features[i][:-1]  # All but last day
            y = features[i][1:, 0]  # Next day's return
            
            X_train, y_train = X, y
            
            X_train_scaled = self.scalers[i].fit_transform(X_train)
            
            self.models[i].fit(X_train_scaled, y_train)
        
        self.is_trained = True

    def predict(self, prices):
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        features = self.preprocess_data(prices)
        predictions = []
        for i in range(self.n_instruments):
            X = features[i][-1].reshape(1, -1)
            X_scaled = self.scalers[i].transform(X)
            pred = self.models[i].predict(X_scaled)[0]
            predictions.append(pred)
        return np.array(predictions)

model = OptimizedTradingModel(nInst)

def getMyPosition(prcSoFar):
    global currentPos, model
    (nins, nt) = prcSoFar.shape
    
    if nt < 50:
        return np.zeros(nins)
    
    if not model.is_trained and nt >= 100:
        model.train(prcSoFar)
    
    if model.is_trained:
        predictions = model.predict(prcSoFar)
        
        # Calculate previous day's returns
        prev_day_returns = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
        
        # Create a mask for momentum trading
        momentum_mask = (predictions > 0) & (prev_day_returns > 0) | (predictions < 0) & (prev_day_returns < 0)
        
        # Convert predictions to desired positions
        desired_positions = predictions * 5000  # Scale predictions to position sizes
        
        # Apply momentum mask
        desired_positions = desired_positions * momentum_mask
        
        # Apply position limits ($10k per instrument)
        max_positions = 10000
        desired_positions = np.clip(desired_positions, -max_positions, max_positions)
        
        # Risk management: reduce exposure in high volatility regime
        volatility = np.std(np.log(prcSoFar[:, -20:] / prcSoFar[:, -21:-1]), axis=1)
        volatility_factor = 1 / (1 + np.exp(10 * (volatility - np.mean(volatility))))  # Sigmoid function
        desired_positions *= volatility_factor
        
        # Convert to integer positions
        desired_positions = np.round(desired_positions).astype(int)
        
        # Calculate the change in positions
        position_changes = desired_positions - currentPos
        
        # Apply a maximum change of 20% of the maximum allowed position
        max_change = 0.2 * max_positions
        position_changes = np.clip(position_changes, -max_change, max_change)
        
        # Update current positions
        currentPos += position_changes.astype(int)
    
    return currentPos