import Experiment_Data as Experiment_Data
import Multistage_Stochastic as Multistage_Stochastic
import matplotlib.pyplot as plt
import numpy as np
import os

Historical_N = 6
Num_Retailers = 2
simulation_time = 1
random_seed = 320
rng = np.random.default_rng( random_seed )
simulation_seeds = rng.integers( low = 0, high = 2**32, size = simulation_time )

for seed_id, seed in enumerate( simulation_seeds ):
    historical = Experiment_Data.Historical_Data( N = Historical_N, R = Num_Retailers, random_seed = seed )
    for i in range( len( historical ) ):
        print( f'Historical {i + 1}: \n{historical[i, :, :]}' )
        print( '-.' * 50 )
    
    evaluation = Experiment_Data.Evaluation_Data( N = Historical_N, R = Num_Retailers, random_seed = seed )
    
    Partition_Path = Multistage_Stochastic.Partition( N = Historical_N, historical = historical, R = Num_Retailers )
    # print( '*' * 100 )
    # for i in range( len( Partition_Path ) ):
    #     print( f'Partition {i + 1}: \n{Partition_Path[i]}' )
    
    epsilon_N = 10 ** ( -0.5 )
    print( '*' * 100 )
    print( f'Epsilon_N: {epsilon_N}' )
    Kj = Multistage_Stochastic.Construct_Kj( historical, Num_Retailers, Partition_Path )
    # print( '*' * 100 )
    # print( f'Kj: {Kj}' )
    
    uncertainty_max, uncertainty_min = Multistage_Stochastic.Uncertainty_Bounds( historical = historical, epsilon = epsilon_N, Partition_Path = Partition_Path, R = Num_Retailers )
    for data_idx in range( len( uncertainty_max ) ):
        print( f'Maximum - historical: \n{uncertainty_max[data_idx, :, :] - historical[data_idx, :, :]}' )
        print( f'Historical - minimum: \n{historical[data_idx, :, :] - uncertainty_min[data_idx, :, :] }' )
        print( '-.' * 50 )
        max_distance = np.linalg.norm( uncertainty_max[data_idx, :, :] - historical[data_idx, :, :], ord = np.inf )
        min_distance = np.linalg.norm( historical[data_idx, :, :] - uncertainty_min[data_idx, :, :], ord = np.inf )
        print( f'Historical {data_idx + 1} maximum distance: {round( max_distance, 2 )}, minimum distance: {round( min_distance, 2 )}' )
    
    # print( '*' * 100 )
    # for i in range( len( uncertainty_max ) ):
    #     for j in range( len( uncertainty_max[i] ) ):
    #         print( f'Historical {i + 1} stage {j + 1} max: {uncertainty_max[i, :, j]}' )
    #         print( f'Historical {i + 1} stage {j + 1} min: {uncertainty_min[i, :, j]}' )
    #         print( '-.' * 50 )
    input( 'Press any key to continue...' )