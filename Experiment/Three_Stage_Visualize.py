import matplotlib.pyplot as plt
import pandas as pd, numpy as np

plt.rcParams.update({'font.size': 20})

df = pd.read_csv( 'Result/Simulation_Result_ThreeStage.csv' )
rec_df = pd.read_csv( 'Result/Simulation_Result_ThreeStage_Rec.csv' )
N_space = [50, 200, 800] # 
epsilon_space = np.logspace( start = np.log10( 10 ** (-3) ), stop = np.log10( 10 ** (0) ), num = 61, endpoint = True )


for N in N_space:
    temp_df = df[df['Historical_N'] == N].reset_index( drop = True )
    
    percentage_20, percentage_80 = np.zeros( len( epsilon_space ) ), np.zeros( len( epsilon_space ) )
    for epsilon_id, epsilon_N in enumerate(epsilon_space):
        temp_rec = rec_df[f'{N}_{epsilon_N}_Test'].to_numpy()
        sorted_rec = np.sort( temp_rec )
        percentage_20[epsilon_id] = sorted_rec[9]
        percentage_80[epsilon_id] = sorted_rec[39]
    
    benchmark = [19.7 for _ in range( len( epsilon_space ) )]
    x_ticks = [i for i in range( 0, 61, 10)]
    x_labels = [r'$10^{-3}$', r'$10^{-0.5}$', r'$10^{-2}$', r'$10^{-1.5}$', r'$10^{-1}$', r'$10^{-0.5}$', r'$10^{0}$']
    plt.figure( figsize = ( 8, 8 ) )
    x = np.arange( len( temp_df ) )
    plt.plot( x, temp_df['Train_Average_Cost'].values, color = "red", linestyle = '--', label = "Train" )
    plt.plot( x, temp_df['Test_Average_Cost'].values, color = "blue", label = "Test" )
    plt.plot( x, benchmark, color = "green", label = "Benchmark" )
    plt.fill_between( x, percentage_20, percentage_80, alpha = 0.3 )
    plt.xticks( x_ticks, x_labels )
    plt.xlabel( 'epsilon_N' )
    plt.ylabel( 'Average_Cost' )
    plt.ylim( 19, 21 )
    plt.yticks( [19, 20, 21, 22, 23, 24] )
    plt.legend( loc = 'best' )
    plt.grid()
    plt.title( f'N = {N}' )
    plt.savefig( f'Figure/Simulation_Result/Three_Stage/N_{N}.jpg' )
    # plt.show()
    plt.close()
