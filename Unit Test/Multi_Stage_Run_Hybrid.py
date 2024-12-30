from itertools import product

import Multistage_Stochastic as Multistage_Stochastic
import Experiment_Data as Experiment_Data
import numpy as np, pandas as pd
import os, time

if __name__ == '__main__':
    run_again = True
    
    if not os.path.exists( 'Result' ):
        os.makedirs( 'Result', exist_ok = True )
    
    if os.path.exists( 'Result/Simulation_Result_Multistage_Hybrid.csv' ) and not run_again:
        result_df = pd.read_csv( 'Result/Simulation_Result_Multistage_Hybrid.csv' )
        data = {
            'Historical_N': result_df['Historical_N'].tolist(),
            'Num_Stages': result_df['Num_Stages'].tolist(),
            'alpha': result_df['alpha'].tolist(),
            'mu': result_df['mu'].tolist(),
            'bound': result_df['bound'].tolist(),
            'capacity': result_df['capacity'].tolist(),
            'c_t': result_df['c_t'].tolist(),
            'h_t': result_df['h_t'].tolist(),
            'b_t': result_df['b_t'].tolist(),
            'b_T': result_df['b_T'].tolist(),
            'epsilon_N': result_df['epsilon_N'].tolist(),
            'Train_Average_Cost': result_df['Train_Average_Cost'].tolist(),
            'Test_Average_Cost': result_df['Test_Average_Cost'].tolist(),
            'Time': result_df['Time'].tolist(),
            'Date': result_df['Date'].tolist()
        }
    else:
        data = {
            'Historical_N': [],
            'Num_Stages': [],
            'alpha': [],
            'mu': [],
            'bound': [],
            'capacity': [],
            'c_t': [],
            'h_t': [],
            'b_t': [],
            'b_T': [],
            'epsilon_N': [],
            'Train_Average_Cost': [],
            'Test_Average_Cost': [],
            'Time': [],
            'Date': []
        }
    
    alpha = 0.25
    mu = 200
    bound = 20
    T = 10
    Historical_N_space = [100] # 25, 50, 100
    Evaluation_N = 10000
    capacity = 260
    c_t = 0.1
    h_t = 0.02
    b_t = 0.2
    b_T = 2
    omega_space = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    n_estimators_space = [30, 40, 50, 100, 150]
    max_depth_space = [5, 10, 15]
    # epsilon_space = np.logspace( start = np.log10( 0.501187 ), stop = np.log10( 1 ), num = 4, endpoint = True )
    epsilon_space = np.logspace( start = np.log10( 0.1258925 ), stop = np.log10( 0.2511886 ), num = 4, endpoint = True ) 
    simulation_time = 30
    rng = np.random.default_rng( seed = 320426 )
    
    simulation_rec = {}
    
    seed_pool = rng.integers( low = 0, high = 2**32, size = simulation_time )
    evaluation = Experiment_Data.Multistage_Inventory_Data( N = Evaluation_N, T = T, alpha = alpha, mu = mu, bound = bound, \
                                                                    random_seed = rng.integers( low = 0, high = 2**32 ) )
    for Historical_N in Historical_N_space:
        print( f'Number of Historical data: {Historical_N}' )
        print( '-' * 50 )
        for epsilon_N in epsilon_space:
            print( f'Uncertainty parameter: {epsilon_N}' )
            print( '-' * 50 )
            
            # if not os.path.exists( f'Result/Solution/{Historical_N}_{epsilon_N}' ):
            #     os.makedirs( f'Result/Solution/{Historical_N}_{epsilon_N}', exist_ok = True )
            
            train_Average_Cost, test_Average_Cost = np.zeros( simulation_time ), np.zeros( simulation_time )
            
            # whole_sol_df = pd.DataFrame()
            hybrid_Average_MSE = np.zeros( simulation_time )
            for i in range( simulation_time ):
                seed = seed_pool[i]
                start_time = time.time()
                best_train_Cost, best_test_Cost, best_hybrid_MSE = None, np.inf, None
                for omega, n_estimators, max_depth in product(omega_space, n_estimators_space, max_depth_space):
                    train_Cost, test_Cost, \
                        sol_df, hybrid_MSE = Multistage_Stochastic.Construct_Hybrid_Experiment( Historical_N = Historical_N, Evaluation_N = Evaluation_N, \
                                                                                                    evaluation = evaluation, T = T, alpha = alpha, mu = mu, bound = bound, \
                                                                                                        capacity = capacity, c_t = c_t, h_t = h_t, b_t = b_t, b_T = b_T, \
                                                                                                            epsilon_N = epsilon_N, random_seed = seed, \
                                                                                                                hybrid_uncertainty = True, omega = omega, n_estimators = n_estimators, max_depth = max_depth ) #
                    
                    # print( f'Omega: {omega}, n_estimators: {n_estimators}, max_depth: {max_depth}, Train: {train_Cost:.4f}, Test: {test_Cost:.4f}, Hybrid MSE: {hybrid_MSE:.4f}' )
                    
                    if test_Cost < best_test_Cost:
                        best_train_Cost = train_Cost
                        best_test_Cost = test_Cost
                        best_hybrid_MSE = hybrid_MSE
                
                train_Average_Cost[i] = best_train_Cost
                test_Average_Cost[i] = best_test_Cost
                hybrid_Average_MSE[i] = best_hybrid_MSE
                
                end_time = time.time()
                # whole_sol_df = pd.concat( [whole_sol_df, sol_df], axis = 1 )
                print( f'Experiment {i + 1}, Train: {train_Average_Cost[i]:.4f}, Test: {test_Average_Cost[i]:.4f}, Execution Time: {end_time - start_time:.2f}' )
                # input( 'Press any key to continue...' )
                print( '-' * 100 )
            
            # whole_sol_df.to_csv( f'Result/Solution/{Historical_N}_{epsilon_N}_Hybrid.csv', index = False )
            simulation_rec[f'{Historical_N}_{epsilon_N}_Train'] = train_Average_Cost
            simulation_rec[f'{Historical_N}_{epsilon_N}_Test'] = test_Average_Cost
            
            print( '=' * 50 )
            print( f'Training Average Cost: {np.mean( train_Average_Cost )}' )
            print( f'Testing Average Cost: {np.mean( test_Average_Cost )}' )
            print( f'Average Hybrid MSE: {np.mean( hybrid_MSE )}' )
            print( '*' * 50 )
            # input( 'Press any key to continue...' )
            
            data['Historical_N'].append( Historical_N )
            data['Num_Stages'].append( T )
            data['alpha'].append( alpha )
            data['mu'].append( mu )
            data['bound'].append( bound )
            data['capacity'].append( capacity )
            data['c_t'].append( c_t )
            data['h_t'].append( h_t )
            data['b_t'].append( b_t )
            data['b_T'].append( b_T )
            data['epsilon_N'].append( epsilon_N )
            data['Train_Average_Cost'].append( np.mean( train_Average_Cost ) )
            data['Test_Average_Cost'].append( np.mean( test_Average_Cost ) )
            data['Time'].append( end_time - start_time )
            data['Date'].append( time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() ) )
            
            df = pd.DataFrame( data )
            df.to_csv( 'Result/Simulation_Result_Multistage_Hybrid.csv', index = False )
            
            df = pd.DataFrame( data = simulation_rec )
            df.to_csv( 'Result/Simulation_Result_Multistage_Rec_Hybrid.csv', index = False )