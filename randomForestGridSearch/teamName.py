import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst)

# Load the price data
prices_df = pd.read_csv('prices.txt', sep='\s+', header=None)
prices = prices_df.values.T

def calculate_features(prcSoFar):
    features = []
    for i in range(nInst):
        company_prices = prcSoFar[i]
        
        # Moving averages
        ma5 = np.convolve(company_prices, np.ones(5), 'valid') / 5
        ma10 = np.convolve(company_prices, np.ones(10), 'valid') / 10
        ma20 = np.convolve(company_prices, np.ones(20), 'valid') / 20
        
        # Returns
        returns_1d = np.diff(company_prices) / company_prices[:-1]
        returns_2d = (company_prices[2:] - company_prices[:-2]) / company_prices[:-2]
        returns_5d = (company_prices[5:] - company_prices[:-5]) / company_prices[:-5]
        
        # Relative Strength Index (RSI)
        delta = np.diff(company_prices)
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        avg_gain = np.convolve(gain, np.ones(14), 'valid') / 14
        avg_loss = np.convolve(loss, np.ones(14), 'valid') / 14
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Determine the shortest length
        min_length = min(len(ma20), len(returns_5d), len(rsi))
        
        # Combine features, trimming to the shortest length
        company_features = np.column_stack((
            ma5[-min_length:],
            ma10[-min_length:],
            ma20[-min_length:],
            returns_1d[-min_length:],
            returns_2d[-min_length:],
            returns_5d[-min_length:],
            rsi[-min_length:]
        ))
        features.append(company_features)
    
    return np.array(features)

def train_model(X, y):
    param_grid = {
        'n_estimators': [300, 400, 500],
        'max_depth': [3, 4],
        'min_samples_split': [14, 16],
        'min_samples_leaf': [6, 8]
    }
    
    rf = RandomForestRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=2)
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_

models = [None] * nInst

def getMyPosition(prcSoFar):
    global currentPos, models
    
    (nins, nt) = prcSoFar.shape
    if nt < 35:  # We need at least 35 days of data to calculate all features
        return np.zeros(nins)
    
    features = calculate_features(prcSoFar)
    
    for i in range(nInst):
        if models[i] is None:
            X = features[i][:-1]
            y = features[i][1:, 3]  # Use next day's return as target
            models[i] = train_model(X, y)
            print(f"for column {i} the best model is {models[i]}")
    
    predictions = []
    for i in range(nInst):
        X = features[i][-1].reshape(1, -1)
        pred = models[i].predict(X)[0]
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Normalize predictions to get position sizes
    position_sizes = predictions / np.sum(np.abs(predictions)) * 10000
    
    # Clip positions to respect the $10k limit
    currentPos = np.clip(position_sizes, -10000 / prcSoFar[:, -1], 10000 / prcSoFar[:, -1])
    
    return currentPos.astype(int)