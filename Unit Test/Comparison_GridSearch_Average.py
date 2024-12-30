import pandas as pd

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