import Experiment_Data as Experiment_Data
import Multistage_Stochastic as Multistage_Stochastic
import matplotlib.pyplot as plt
import numpy as np
import os

Historical_N = 6
Num_Retailers = 2
simulation_time = 1
random_seed = 426
rng = np.random.default_rng( random_seed )
simulation_seeds = rng.integers( low = 0, high = 2**32, size = simulation_time )

for seed_id, seed in enumerate( simulation_seeds ):
    historical = Experiment_Data.Historical_Data( N = Historical_N, R = Num_Retailers, random_seed = seed )
    # for i in range( len( historical ) ):
    #     print( f'Historical {i + 1}: \n{historical[i, :, :]}' )
    #     print( '-.' * 50 )  
    
    evaluation = Experiment_Data.Evaluation_Data( N = Historical_N, R = Num_Retailers, random_seed = seed )
    
    Partition_Path = Multistage_Stochastic.Partition( N = Historical_N, historical = historical, R = Num_Retailers )
    
    # for i in range( len( Partition_Path ) ):
    #     print( f'Partition {i}: \n{Partition_Path[i]}' )
    
    epsilon_N = 10 ** ( -3 )
    # print( '*' * 100 )
    # print( f'Epsilon_N: {epsilon_N}' )
    
    Kj = Multistage_Stochastic.Construct_Kj( historical, Num_Retailers, Partition_Path, epsilon_N )
    print( Kj )
    input( 'Press any key to continue...' )