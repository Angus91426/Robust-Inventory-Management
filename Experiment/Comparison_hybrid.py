import pandas as pd, numpy as np

df = pd.read_csv( 'Result/Comparison/GridSearch_Average_Result.csv' )
N_space = [25, 50, 100]
epsilon_space = df['epsilon_N'].unique()

Hybrid_Original_data = {
    'Historical_N': [],
    'epsilon_N': [],
    'Original_Train': [],
    'Original_Test': [],
    'Hybrid_Train': [],
    'Hybrid_Test': [],
    'Original_Service_level': [],
    'Hybrid_Service_level': [],
    'Original_Time': [],
    'Hybrid_Time': []
}

for N in N_space:
    for epsilon_N in epsilon_space:
        Hybrid_Original_data['Historical_N'].append( N )
        Hybrid_Original_data['epsilon_N'].append( epsilon_N )
        
        temp_df = df.loc[ ( df['Historical_N'] == N ) & ( df['epsilon_N'] == epsilon_N ) & ( df['omega'] == 0 ) ]
        Hybrid_Original_data['Original_Train'].append( temp_df['Train_Cost'].tolist()[0] )
        Hybrid_Original_data['Original_Test'].append( temp_df['Test_Cost'].tolist()[0] )
        Hybrid_Original_data['Original_Service_level'].append( temp_df['Service_Level'].tolist()[0] )
        Hybrid_Original_data['Original_Time'].append( temp_df['Time'].tolist()[0] )
        
        temp_df = df.loc[ ( df['Historical_N'] == N ) & ( df['epsilon_N'] == epsilon_N ) & ( df['omega'] != 0 ) ]
        # Find the optimal omega with the minimum testing cost
        min_test_cost = temp_df['Test_Cost'].min()
        temp_df = temp_df.loc[ temp_df['Test_Cost'] == min_test_cost ]
        if len( temp_df ) > 1:
            fast_Time = temp_df['Time'].min()
            temp_df = temp_df.loc[ temp_df['Time'] == fast_Time ]
        
        Hybrid_Original_data['Hybrid_Train'].append( temp_df['Train_Cost'].tolist()[0] )
        Hybrid_Original_data['Hybrid_Test'].append( temp_df['Test_Cost'].tolist()[0] )
        Hybrid_Original_data['Hybrid_Service_level'].append( temp_df['Service_Level'].tolist()[0] )
        Hybrid_Original_data['Hybrid_Time'].append( temp_df['Time'].tolist()[0] )

df = pd.DataFrame( Hybrid_Original_data )
df.to_csv( 'Result/Comparison/Hybrid_Original_Result.csv', index = False )