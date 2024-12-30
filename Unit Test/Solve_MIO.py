import Experiment_Data as Experiment_Data
import Multistage_Stochastic as Multistage_Stochastic
import matplotlib.pyplot as plt
import numpy as np
import os

Historical_N = 6
Num_Retailers = 2
c = 0.25
h = 0.05
b = 0.5
f = 0.1
simulation_time = 1
random_seed = 320
rng = np.random.default_rng( random_seed )
simulation_seeds = rng.integers( low = 0, high = 2**32, size = simulation_time )

for seed_id, seed in enumerate( simulation_seeds ):
    true_data = Experiment_Data.Historical_Data( N = 1, R = Num_Retailers, random_seed = seed )
    
    historical = Experiment_Data.Historical_Data( N = Historical_N, R = Num_Retailers, random_seed = seed )
    # for i in range( len( historical ) ):
    #     for j in range( 2 ):
    #         print( f'Historical {i + 1} stage {j + 1}: \n{historical[i, :, j]}' )
    #     print( '-.' * 50 )
    
    evaluation = Experiment_Data.Evaluation_Data( N = 10, R = Num_Retailers, random_seed = seed )
    
    Partition_Path = Multistage_Stochastic.Partition( N = Historical_N, historical = historical, R = Num_Retailers )
    
    epsilon_N = 10 ** ( -0.5 )
    
    Kj = Multistage_Stochastic.Construct_Kj( historical, Num_Retailers, Partition_Path )
    
    Partition_Path = Multistage_Stochastic.Partition_Reindex( Partition_Path, Kj )
    
    uncertainty_max, uncertainty_min = Multistage_Stochastic.Uncertainty_Bounds( historical = historical, epsilon = epsilon_N, R = Num_Retailers )
    # print( '*' * 100 )
    # for i in range( len( uncertainty_max ) ):
    #     for j in range( len( uncertainty_max[i] ) ):
    #         print( f'Historical {i + 1} stage {j + 1} max: {uncertainty_max[i, :, j]}' )
    #         print( f'Historical {i + 1} stage {j + 1} min: {uncertainty_min[i, :, j]}' )
    #         print( '-.' * 50 )
    # input( 'Press any key to continue...' )
    
    Q1_opt, Q2_opt, z_opt = Multistage_Stochastic.Solve_MIO( N = Historical_N, R = Num_Retailers, Historical = historical, Partition_Path = Partition_Path, \
                                                    uncertainty_max = uncertainty_max, uncertainty_min = uncertainty_min, f = f, h = h, c = c, b = b )
    
    Q2_sol, z_sol = Multistage_Stochastic.Generate_Q2_Sol( Q2_pool = Q2_opt, z_pool = z_opt, T1_demand = evaluation[:, :, 0], partition_path = Partition_Path )