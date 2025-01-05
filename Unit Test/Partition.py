import Experiment_Data as Experiment_Data
import Multistage_Stochastic as Multistage_Stochastic
import matplotlib.pyplot as plt
import numpy as np
import os

if not os.path.exists( 'Figure/Partition' ):
    os.makedirs( 'Figure/Partition' )

Historical_N = 6
Num_Retailers = 2
simulation_time = 1
random_seed = 426
rng = np.random.default_rng( random_seed )
simulation_seeds = rng.integers( low = 0, high = 2**32, size = simulation_time )

for seed_id, seed in enumerate( simulation_seeds ):
    historical = Experiment_Data.Historical_Data( N = Historical_N, R = Num_Retailers, random_seed = seed )
    for i in range( len( historical ) ):
        for r in range( Num_Retailers ):
            print( f'Historical {i + 1} retailer {r + 1}: \n{historical[i, r, :]}' )
        print( '-.' * 50 )
    print( '*' * 100 )
    # evaluation = Experiment_Data.Evaluation_Data( N = Historical_N, R = Num_Retailers, random_seed = seed )
    
    Partition = Multistage_Stochastic.Partition( N = Historical_N, R = Num_Retailers, Historical = historical )
    for i in range( len( Partition ) ):
        print( f'Partition {i + 1}: \n{Partition[i]}' )
        print( '-.' * 50 )
    
    # plt.figure( figsize = ( 20, 20 ) )
    # plt.plot( historical[:, 0, 0], historical[:, 1, 0], 'ro' )
    # # Draw the vertical and horizontal lines for the thresholds
    # for i in range( len( Partition ) ):
    #     plt.vlines( x = Partition[i][0], ymin = 0, ymax = max( historical[:, 1, 0] ), color = 'r', linestyles = 'dashed' )
    #     if Partition[i][0][1] == np.inf:
    #         plt.hlines( y = Partition[i][1], xmin = Partition[i][0][0], xmax = max( historical[:, 0, 0] ) + 1, color = 'b', linestyles = 'dashed' )
    #     else:
    #         plt.hlines( y = Partition[i][1], xmin = Partition[i][0][0], xmax = Partition[i][0][1], color = 'b', linestyles = 'dashed' )
    # plt.xlim( 0, max( historical[:, 0, 0] ) + 1 )
    # plt.ylim( 0, max( historical[:, 1, 0] ) + 1 )
    # plt.xlabel( 'Retailer 1' )
    # plt.ylabel( 'Retailer 2' )
    # plt.show()
    # plt.savefig( f'Figure/Partition/Experiment_{seed_id + 1}.png' )
    # plt.close()
    input( 'Press any key to continue...' )