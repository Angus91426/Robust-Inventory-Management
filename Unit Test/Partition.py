import Python.Experiment_Data as Experiment_Data
import Python.Three_stage_Stochastic as Three_stage_Stochastic
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

if not os.path.exists( 'Figure/Partition' ):
    os.makedirs( 'Figure/Partition' )

plt.rcParams.update({'font.size': 16})

Historical_N = 6
Num_Retailers = 2
simulation_time = 1
random_seed = 910320
epsilon_N = 0.5
rng = np.random.default_rng( random_seed )
simulation_seeds = rng.integers( low = 0, high = 2**32, size = simulation_time )

for seed_id, seed in enumerate( simulation_seeds ):
    historical = Experiment_Data.Historical_Data( N = Historical_N, R = Num_Retailers, random_seed = seed )
    
    max_val = max( [max( historical[:, 0, 0] ), max( historical[:, 1, 0] )] )
    
    Partition = Three_stage_Stochastic.Partition( N = Historical_N, R = Num_Retailers, Historical = historical )
    
    plt.figure( figsize = ( 6, 6 ), dpi = 330 )
    plt.plot( historical[:, 0, 0], historical[:, 1, 0], 'ro' )
    plt.xlim( 0, max_val + 0.1 )
    plt.ylim( 0, max_val + 0.1 )
    plt.xticks( [] )
    plt.yticks( [] )
    plt.xlabel( 'Retailer 1' )
    plt.ylabel( 'Retailer 2' )
    plt.tight_layout()
    # plt.title( 'Historical data' )
    plt.savefig( f'Figure/Historical_Example.png' )
    # Draw the vertical and horizontal lines for the thresholds
    for i in range( len( Partition ) ):
        if Partition[i][0][0] == 0:
            plt.vlines( x = Partition[i][0][1], ymin = 0, ymax = max_val + 0.1, color = 'r', linestyles = 'dashed' )
        else:
            plt.vlines( x = Partition[i][0], ymin = 0, ymax = max_val + 0.1, color = 'r', linestyles = 'dashed' )
    
    plt.savefig( f'Figure/Partition1_Example.png' )
    
    for i in range( len( Partition ) ):
        if Partition[i][0][1] == np.inf:
            plt.hlines( y = Partition[i][1][1], xmin = Partition[i][0][0], xmax = max_val + 0.1, color = 'b', linestyles = 'dashed' )
        else:
            plt.hlines( y = Partition[i][1][1], xmin = Partition[i][0][0], xmax = Partition[i][0][1], color = 'b', linestyles = 'dashed' )
    
    # plt.title( 'Partition result' )
    plt.savefig( f'Figure/Partition2_Example.png' )
    
    # Draw the uncertainty region
    for historical_idx in range( len( historical ) ):
        stg1_demand = historical[historical_idx, :, 0]
        ub = stg1_demand + epsilon_N
        lb = np.maximum( stg1_demand - epsilon_N, 0 )
        plt.fill_between( x = [ lb[0], ub[0] ], y1 = lb[1], y2 = ub[1], color = 'green', alpha = 0.3 )
    
    # plt.title( 'Uncertainty region' )
    plt.savefig( f'Figure/Uncertainty_Example.png' )
    # plt.close()
    # input( 'Press any key to continue...' )