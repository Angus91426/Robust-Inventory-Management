import matplotlib.pyplot as plt
import pandas as pd, numpy as np

# Set the global font size
plt.rcParams.update({'font.size': 20}) 

df = pd.read_csv( 'Result/Simulation_Result_Multistage_30.csv' )
rec_df = pd.read_csv( 'Result/Simulation_Result_Multistage_Rec_30.csv' )
N_space = [25, 50, 100] # 
epsilon_space = np.logspace( start = np.log10( 10 ** (-3) ), stop = np.log10( 10 ** (1) ), num = 41, endpoint = True )

for N in N_space:
    temp_df = df[df['Historical_N'] == N].reset_index( drop = True )
    
    percentage_20, percentage_80 = np.zeros( len( epsilon_space ) ), np.zeros( len( epsilon_space ) )
    for epsilon_id, epsilon_N in enumerate(epsilon_space):
        temp_rec = rec_df[f'{N}_{epsilon_N}_Test'].to_numpy()
        sorted_rec = np.sort( temp_rec )
        percentage_20[epsilon_id] = sorted_rec[5]
        percentage_80[epsilon_id] = sorted_rec[23]
    
    benchmark = [206 for _ in range( len( epsilon_space ) )]
    x_ticks = [0, 10, 20, 30, 40]
    x_labels = temp_df['epsilon_N'].values[[0, 10, 20, 30, 40]]
    plt.figure( figsize = ( 10, 10 ) )
    x = np.arange( len( temp_df ) )
    plt.plot( x, temp_df['Train_Average_Cost'].values, color = "red", linestyle = '--', label = "Train" )
    plt.plot( x, temp_df['Test_Average_Cost'].values, color = "blue", label = "Test" )
    # plt.plot( x, benchmark, color = "green", label = "Benchmark" )
    plt.fill_between( x, percentage_20, percentage_80, alpha = 0.3 )
    plt.xticks( x_ticks, x_labels )
    plt.xlabel( 'epsilon_N' )
    plt.ylim( 203, 215 )
    plt.ylabel( 'Average_Cost' )
    plt.legend( loc = 'best' )
    plt.grid()
    plt.title( f'N = {N}' )
    plt.savefig( f'Figure/Simulation_Result/Multistage_30/N_{N}.jpg' )
    # plt.show()
    plt.close()
