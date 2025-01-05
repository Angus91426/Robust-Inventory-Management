from matplotlib import lines
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import math
import os
import warnings
warnings.filterwarnings( "ignore" )

def Lag_feature( df, lag ):
    if type( lag ) == list:
        for l in lag:
            df[f'lag_{l}'] = df['demand'].shift( l ).fillna( df['demand'].tolist()[0] )
            df[f'lag_{l}_diff'] = df[f'lag_{l}'] - df['demand']
    else:
        df[f'lag_{lag}'] = df['demand'].shift( lag ).fillna( df['demand'].tolist()[0] )
        df[f'lag_{lag}_diff'] = df[f'lag_{lag}'] - df['demand']
    
    return df

def Rolling_mean( df, window ):
    if type( window ) == list:
        for w in window:
            df[f'rolling_mean_{w}'] = df['demand'].rolling( window = w ).mean().fillna(method='bfill')
    else:
        df[f'rolling_mean_{window}'] = df['demand'].rolling( window = window ).mean().fillna(method='bfill')
    
    return df

def Feature_Engineering( df, lag = None, window = None ):
    if lag is not None:
        df = Lag_feature( df, lag )
    
    if window is not None:
        df = Rolling_mean( df, window )
    return df

def Train_Test_Split( df, omega ):
    Num_of_Stage = len( df )
    test_T = math.ceil( Num_of_Stage * omega )
    train_T = Num_of_Stage - test_T
    train_data = df.iloc[:train_T, :]
    test_data = df.iloc[train_T:, :]
    return train_data, test_data

def Fit_and_Predict( train_data, test_data, params, cv, random_seed, model = 'xgb' ):
    n_estimators = params['n_estimators']
    max_depth = params['max_depth']
    
    if model == 'rf':
        # Random Forest Regressor
        pred_model = RandomForestRegressor( n_estimators = n_estimators, max_depth = max_depth, random_state = random_seed )
    elif model == 'xgb':
        # XGBoost Regressor
        pred_model = XGBRegressor( n_estimators = n_estimators, max_depth = max_depth, random_state = random_seed )
    
    pred_model.fit( train_data.drop( 'demand', axis = 1 ), train_data['demand'] )
    predictions = pred_model.predict( test_data.drop( 'demand', axis = 1 ) )
    prediction_MSE = mean_squared_error( test_data['demand'], predictions )
    
    return predictions, prediction_MSE

def Construct_Hybrid_Uncertainty( demand, n_estimators, max_depth, lag = [1, 2, 3], window = [1, 2, 3], omega = 0.7, random_seed = 320426, model = 'xgb' ):
    # Feature Engineering
    df = pd.DataFrame( data = { 'demand': demand } )
    df = Feature_Engineering( df, lag, window )
    
    # Train and Test Split
    train_data, test_data = Train_Test_Split( df, omega )
    
    # Fit and Predict
    model_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
    cv = 5
    model_predictions, model_MSE = Fit_and_Predict( train_data, test_data, model_params, cv, random_seed, model = model )
    
    # Construct Hybrid Uncertainty
    hybrid_data = np.array( train_data['demand'].tolist() + model_predictions.tolist() )
    
    return hybrid_data, model_predictions, model_MSE