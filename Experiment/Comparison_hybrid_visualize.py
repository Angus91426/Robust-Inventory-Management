import pandas as pd, numpy as np, matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 20})

if not os.path.exists( 'Figure/Hybrid_Original/Vertical' ):
    os.makedirs( 'Figure/Hybrid_Original/Vertical', exist_ok = True )
if not os.path.exists( 'Figure/Hybrid_Original/Horizon' ):
    os.makedirs( 'Figure/Hybrid_Original/Horizon', exist_ok = True )

df = pd.read_csv( 'Result/Comparison/Hybrid_Original_Result.csv' )
N_space = [25, 50, 100]
epsilon_space = df['epsilon_N'].unique()
x = np.arange( len( epsilon_space ) )
x_ticks = [1, 11, 21, 31, 41]
x_labels = [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$']

for N in N_space:
    temp_df = df[df['Historical_N'] == N].reset_index( drop = True )
    # Find the optimal epsilon_N for hybrid
    min_hybrid_df = temp_df.loc[ temp_df['Hybrid_Test'] == temp_df['Hybrid_Test'].min() ]
    if len( min_hybrid_df ) > 1:
        fast_Time = min_hybrid_df['Hybrid_Time'].min()
        min_hybrid_df = min_hybrid_df.loc[ min_hybrid_df['Hybrid_Time'] == fast_Time ]
    
    min_hybrid_epsilon_N = min_hybrid_df['epsilon_N'].iloc[0]
    min_idx = temp_df['epsilon_N'].tolist().index( min_hybrid_epsilon_N )
    
    #! Horizon
    # Cost
    plt.figure( figsize = ( 14, 5 ) )
    # plt.suptitle( f'N = {N}' )
    plt.subplot( 1, 2, 1 )
    plt.plot( x, temp_df['Original_Train'].values, color = "red", linestyle = '--', label = "Original Train" )
    plt.plot( x, temp_df['Original_Test'].values, color = "red", label = "Original Test" )
    plt.plot( x, temp_df['Hybrid_Train'].values, color = "blue", linestyle = '--', label = "Hybrid Train" )
    plt.plot( x, temp_df['Hybrid_Test'].values, color = "blue", label = "Hybrid Test" )
    plt.vlines( min_idx, 204, 213, color = 'black', linestyle = 'dotted' )
    plt.xlim( 0, 42 )
    plt.xticks( x_ticks, x_labels )
    plt.xlabel( r'$\epsilon_N$' )
    plt.ylim( 204, 213 )
    plt.ylabel( 'Average Cost ($)' )
    plt.grid()
    plt.legend( loc = 'best', fontsize = 14 )
    
    # Service level
    plt.subplot( 1, 2, 2 )
    plt.plot( x, temp_df['Original_Service_level'].values, color = "red", label = "Original" )
    plt.plot( x, temp_df['Hybrid_Service_level'].values, color = "blue", label = "Hybrid" )
    plt.vlines( min_idx, 80, 100, color = 'black', linestyle = 'dotted' )
    plt.xlim( 0, 42 )
    plt.xticks( x_ticks, x_labels )
    plt.xlabel( r'$\epsilon_N$' )
    plt.ylim( 80, 100 )
    plt.yticks( [80, 85, 90, 95, 100] )
    plt.ylabel( 'Service Level (%)' )
    plt.grid()
    plt.legend( loc = 'best', fontsize = 14 )
    
    plt.tight_layout()
    plt.savefig( f'Figure/Hybrid_Original/Horizon/N_{N}.jpg', dpi = 330 )
    # plt.show()
    
    # #! Vertical
    # # Cost
    # plt.figure( figsize = ( 8, 10 ) )
    # plt.suptitle( f'N = {N}\nOriginal: Without #40 Composite materials\nHybrid: With #40 Composite materials' )
    # plt.subplot( 2, 1, 1 )
    # plt.plot( x, temp_df['Original_Train'].values, color = "red", linestyle = '--', label = "Train" )
    # plt.plot( x, temp_df['Original_Test'].values, color = "green", label = "Test" )
    # plt.plot( x, temp_df['Hybrid_Train'].values, color = "blue", linestyle = '--', label = "Hybrid Train" )
    # plt.plot( x, temp_df['Hybrid_Test'].values, color = "orange", label = "Hybrid Test" )
    # plt.vlines( min_idx, 204, 213, color = 'black', linestyle = 'dotted' )
    # plt.xlim( 0, 42 )
    # plt.xticks( x_ticks, x_labels )
    # plt.xlabel( r'$\epsilon_N$' )
    # plt.ylim( 204, 213 )
    # plt.ylabel( 'Average Cost ($)' )
    # plt.grid()
    # plt.legend( loc = 'best', fontsize = 15 )
    
    # # Service level
    # plt.subplot( 2, 1, 2 )
    # plt.plot( x, temp_df['Original_Service_level'].values, color = "darkviolet", label = "Original" )
    # plt.plot( x, temp_df['Hybrid_Service_level'].values, color = "lightpink", label = "Hybrid" )
    # plt.vlines( min_idx, 80, 100, color = 'black', linestyle = 'dotted' )
    # plt.xlim( 0, 42 )
    # plt.xticks( x_ticks, x_labels )
    # plt.xlabel( r'$\epsilon_N$' )
    # plt.ylim( 80, 100 )
    # plt.ylabel( 'Service Level (%)' )
    # plt.grid()
    # plt.legend( loc = 'best', fontsize = 15 )
    
    # plt.tight_layout( pad = 3 )
    # plt.savefig( f'Figure/Hybrid_Original/Vertical/N_{N}.jpg', dpi = 330 )