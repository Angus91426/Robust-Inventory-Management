from itertools import product
import numpy as np, pandas as pd
import Multistage_Stochastic as Multistage_Stochastic
import Experiment_Data as Experiment_Data
import Time_Series_Forecasting as Time_Series_Forecasting
import matplotlib.pyplot as plt
import os, time
import warnings
warnings.filterwarnings( "ignore" )

plt.rcParams.update({'font.size': 20})

if __name__ == '__main__':
    run_again = True
    
    if not os.path.exists( 'Result/Comparison' ):
        os.makedirs( 'Result/Comparison', exist_ok = True )
    
    average_result_filename = 'Comparison_Result.csv'
    
    # The size of the uncertainty set
    epsilon_space = np.logspace( start = np.log10( 10 ** (-3) ), stop = np.log10( 10 ** (1) ), num = 41, endpoint = True ).tolist()
    epsilon_space = [0] + epsilon_space
    
    # Parameters for data generation
    alpha = 0.25
    mu = 200
    bound = 20
    T = 10
    Historical_N_space = [25, 50, 100] # 
    Evaluation_N = 10000
    
    # Parameters for model construction
    capacity = 260
    x_bar = [capacity for _ in range( T )]      # Capacity
    c = [0.1 for _ in range( T )]               # Production cost
    h = [0.02 for _ in range( T )]              # Holding cost
    b = [0.2 for _ in range( T - 1 )] + [2]     # Backorder cost
    
    # Parameters for the proportion of predicted data in the uncertainty set
    omega_space = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # 
    
    # Parameters for grid search
    n_estimators_space = [10, 20, 30, 40, 50, 100, 150] # 
    max_depth_space = [5, 10, 15] # 
    
    simulation_time = 30
    rng = np.random.default_rng( seed = 320426 )
    # Random seeds for each simulation
    seed_pool = rng.integers( low = 0, high = 2**32, size = simulation_time )
    
    # Common testing data set
    evaluation = Experiment_Data.Multistage_Inventory_Data( N = Evaluation_N, T = T, alpha = alpha, mu = mu, bound = bound, \
                                                                random_seed = rng.integers( low = 0, high = 2**32 ) )
    
    # Load in the history data if not run_again
    if os.path.exists( f'Result/Comparison/{average_result_filename}' ) and not run_again:
        average_df = pd.read_csv( f'Result/Comparison/{average_result_filename}' )
        average_data = average_df.to_dict( orient = 'list' )
    else:
        average_data = {
            'Historical_N': [],
            'omega': [],
            'epsilon_N': [],
            'simulation_id': [],
            'n_estimators': [],
            'max_depth': [],
            'Pred_MSE': [],
            'Train_Cost': [],
            'Test_Cost': [],
            'Service_Level': [],
            'Time': []
        }
    
    for Historical_N in Historical_N_space:
        for omega in omega_space:
            for simulation_id in range( simulation_time ):
                seed = seed_pool[ simulation_id ]
                
                # Generate historical training data
                historical = Experiment_Data.Multistage_Inventory_Data( N = Historical_N, T = T, alpha = alpha, mu = mu, bound = bound,  random_seed = seed )
                
                # x = np.arange( T ) + 1
                # plt.figure( figsize = ( 10, 7 ) )
                # plt.plot( x, historical[0], label = 'Historical' )
                # plt.xticks( x, [( i + 1 ) for i in range( T )] )
                # plt.xlabel( 'Stage' )
                # plt.ylabel( 'Demand' )
                # plt.grid()
                # plt.title( 'Example of Generated Demand' )
                # plt.savefig( 'Figure/Demand_Example.jpg' )
                # angus
                
                for n_estimators, max_depth in product( n_estimators_space, max_depth_space ):
                    # Construct hybrid data
                    hybrid_start_time = time.time()
                    hybrid_data = np.zeros( ( Historical_N, T ) )
                    predict_MSE = np.zeros( Historical_N )
                    for j in range( Historical_N ):
                        if omega == 0:
                            hybrid_data[j] = historical[j]
                        else:
                            hybrid_data[j], _, predict_MSE[j] = Time_Series_Forecasting.Construct_Hybrid_Uncertainty( demand = historical[j], omega = omega, random_seed = seed, \
                                                                                                            n_estimators = n_estimators, max_depth = max_depth )
                    average_hybrid_MSE = np.mean( predict_MSE )
                    hybrid_end_time = time.time()
                    hybrid_time = hybrid_end_time - hybrid_start_time
                    
                    # Run each parameter combination for each value of epsilon_N
                    print( '=.' * 50 )
                    for epsilon_N in epsilon_space:
                        robust_time_start = time.time()
                        # Solve the optimization problem
                        x_opt, X_opt, train_Average_Cost, _ = Multistage_Stochastic.Solve_Model( Historical_N, T, hybrid_data, epsilon_N, c, h, x_bar, b )
                        
                        # Generate the decision rule for evaluation data
                        test_decision = Multistage_Stochastic.Generate_Decision( Evaluation_N, T, evaluation, x_opt, X_opt, capacity )
                        
                        # Compute the cost for evaluation data
                        test_Average_Cost, Average_Service_Level = Multistage_Stochastic.Compute_Cost( Evaluation_N, T, evaluation, test_decision, c, h, b )
                        robust_time_end = time.time()
                        robust_time = robust_time_end - robust_time_start
                        
                        print( '-.' * 50 )
                        print( f'Historical_N: {Historical_N}, omega: {omega}, epsilon_N: {epsilon_N:.4f}, simulation_id: {simulation_id}, n_estimators: {n_estimators}, max_depth: {max_depth}' )
                        print( f'Pred_MSE: {average_hybrid_MSE:.4f}, Train_Cost: {train_Average_Cost:.4f}, Test_Cost: {test_Average_Cost:.4f}, Service_Level: {Average_Service_Level:.2f}%, Time: {(hybrid_time + robust_time):.2f}s' )
                        print( '-.' * 50 )
                        
                        average_data['Historical_N'].append( Historical_N )
                        average_data['omega'].append( omega )
                        average_data['epsilon_N'].append( epsilon_N )
                        average_data['simulation_id'].append( simulation_id )
                        average_data['n_estimators'].append( n_estimators )
                        average_data['max_depth'].append( max_depth )
                        average_data['Pred_MSE'].append( average_hybrid_MSE )
                        average_data['Train_Cost'].append( train_Average_Cost )
                        average_data['Test_Cost'].append( test_Average_Cost )
                        average_data['Service_Level'].append( Average_Service_Level )
                        average_data['Time'].append( hybrid_time + robust_time )
                    print( '=.' * 50 )
                    print( '\n' )
                    
                    average_df = pd.DataFrame( data = average_data )
                    average_df.to_csv( f'Result/Comparison/{average_result_filename}', index = False )