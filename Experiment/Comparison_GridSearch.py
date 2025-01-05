import numpy as np, pandas as pd

import warnings
warnings.filterwarnings( "ignore" )

Historical_N_space = [25, 50, 100] # 
omega_space = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # 
simulation_time = 30

df = pd.read_csv( 'Result/Comparison/Comparison_Result.csv' )
epsilon_N_space = df['epsilon_N'].unique()

grid_search_data = {
    'Historical_N': [],
    'omega': [],
    'epsilon_N': [],
    'simulation_id': [],
    'n_estimators': [],
    'max_depth': [],
    'Pred_MSE': [],
    'Train_Cost': [],
    'Test_Cost': [],
    'Service_Level': [],
    'Time': []
}
for Historical_N in Historical_N_space:
    for omega in omega_space:
        for epsilon in epsilon_N_space:
            for simulation_id in range( simulation_time ):
                # Filter the dataframe
                df_temp = df.loc[ ( df['Historical_N'] == Historical_N ) & ( df['omega'] == omega ) & ( df['epsilon_N'] == epsilon ) & ( df['simulation_id'] == simulation_id ) ]
                
                # Find the minimum testing cost
                min_test_cost = df_temp['Test_Cost'].min()
                
                # Filter the dataframe again
                df_temp = df_temp.loc[ df_temp['Test_Cost'] == min_test_cost ]
                
                if len( df_temp ) > 1:
                    fast_Time = df_temp['Time'].min()
                    df_temp = df_temp.loc[ df_temp['Time'] == fast_Time ]
                
                # Append to the grid search data
                grid_search_data['Historical_N'].append( Historical_N )
                grid_search_data['omega'].append( omega )
                grid_search_data['epsilon_N'].append( epsilon )
                grid_search_data['simulation_id'].append( simulation_id )
                grid_search_data['n_estimators'].append( df_temp['n_estimators'].iloc[0] )
                grid_search_data['max_depth'].append( df_temp['max_depth'].iloc[0] )
                grid_search_data['Pred_MSE'].append( df_temp['Pred_MSE'].iloc[0] )
                grid_search_data['Train_Cost'].append( df_temp['Train_Cost'].iloc[0] )
                grid_search_data['Test_Cost'].append( df_temp['Test_Cost'].iloc[0] )
                grid_search_data['Service_Level'].append( df_temp['Service_Level'].iloc[0] )
                grid_search_data['Time'].append( df_temp['Time'].iloc[0] )

GridSearch_df = pd.DataFrame( grid_search_data )
GridSearch_df.to_csv( 'Result/Comparison/GridSearch_Result.csv', index = False )

df = pd.read_csv( 'Result/Comparison/GridSearch_Result.csv' )
Historical_N_space = [25, 50, 100] # 
omega_space = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # 
epsilon_N_space = df['epsilon_N'].unique()

average_data = {
    'Historical_N': [],
    'omega': [],
    'epsilon_N': [],
    'Train_Cost': [],
    'Test_Cost': [],
    'Service_Level': [],
    'Time': []
}

for Historical_N in Historical_N_space:
    for omega in omega_space:
        for epsilon_N in epsilon_N_space:
            df_temp = df.loc[ ( df['Historical_N'] == Historical_N ) & ( df['omega'] == omega ) & ( df['epsilon_N'] == epsilon_N ) ]
            
            average_data['Historical_N'].append( Historical_N )
            average_data['omega'].append( omega )
            average_data['epsilon_N'].append( epsilon_N )
            average_data['Train_Cost'].append( df_temp['Train_Cost'].mean() )
            average_data['Test_Cost'].append( df_temp['Test_Cost'].mean() )
            average_data['Service_Level'].append( df_temp['Service_Level'].mean() )
            average_data['Time'].append( df_temp['Time'].mean() )

average_df = pd.DataFrame( data = average_data )
average_df.to_csv( 'Result/Comparison/GridSearch_Average_Result.csv', index = False )