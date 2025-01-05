import Three_stage_Stochastic as Three_stage_Stochastic
import Experiment_Data as Experiment_Data
import numpy as np
import pandas as pd
import os, time
import warnings
warnings.filterwarnings( "ignore" )

if __name__ == '__main__':
    if not os.path.exists( 'Result' ):
        os.makedirs( 'Result', exist_ok = True )
    
    if os.path.exists( 'Result/Simulation_Result.csv' ):
        result_df = pd.read_csv( 'Result/Simulation_Result.csv' )
        data = {
            'Historical_N': result_df['Historical_N'].tolist(),
            'Evaluation_N': result_df['Evaluation_N'].tolist(),
            'Num_Retailers': result_df['Num_Retailers'].tolist(),
            'f': result_df['f'].tolist(),
            'h': result_df['h'].tolist(),
            'c': result_df['c'].tolist(),
            'b': result_df['b'].tolist(),
            'epsilon_N': result_df['epsilon_N'].tolist(),
            'Train_Average_Cost': result_df['Train_Average_Cost'].tolist(),
            'Test_Average_Cost': result_df['Test_Average_Cost'].tolist(),
            'Date': result_df['Date'].tolist()
        }
    else:
        data = {
            'Historical_N': [],
            'Evaluation_N': [],
            'Num_Retailers': [],
            'f': [],
            'h': [],
            'c': [],
            'b': [],
            'epsilon_N': [],
            'Train_Average_Cost': [],
            'Test_Average_Cost': [],
            'Date': []
        }
    
    Num_Retailers = 3
    Historical_N_space = [50, 200, 800]
    Evaluation_N = 10000
    epsilon_space = np.logspace( start = np.log10( 10 ** (-3) ), stop = np.log10( 10 ** (0) ), num = 61, endpoint = True )
    # epsilon_space = [10 ** (-3), 10 ** (-2.5), 10 ** (-2), 10 ** (-1.5), 10 ** (-1), 10 ** (-0.5), 10 ** (0)]
    # epsilon_space = [10 ** (-0.5)]
    c = 0.25
    h = 0.05
    b = 0.5
    f = 0
    simulation_time = 50
    
    simulation_rec = {}
    
    rng = np.random.default_rng( seed = 320426 )
    seed_pool = rng.integers( low = 0, high = 2**32, size = simulation_time )
    evaluation = Experiment_Data.Evaluation_Data( N = Evaluation_N, R = Num_Retailers, random_seed = rng.integers( low = 0, high = 2**32 ) )
    
    for Historical_N in Historical_N_space:
        print( f'Number of Historical data: {Historical_N}' )
        print( '-' * 50 )
        for epsilon_N in epsilon_space:
            print( f'Uncertainty parameter: {epsilon_N}' )
            print( '-' * 50 )
            
            train_Average_Cost, test_Average_Cost = np.zeros( simulation_time ), np.zeros( simulation_time )
            
            for i in range( simulation_time ):
                seed = seed_pool[i]
                train_Average_Cost[i], \
                    test_Average_Cost[i] = Three_stage_Stochastic.Construct_Experiment( Historical_N = Historical_N, Num_Retailers = Num_Retailers, \
                                                                                        Evaluation_N = Evaluation_N, evaluation = evaluation, \
                                                                                            f = f, h = h, c = c, b = b, \
                                                                                                epsilon_N = epsilon_N, random_seed = seed, \
                                                                                                    save_result = False, save_path = 'Result/Test.csv' ) # 
                print( f'Experiment {i + 1}, Train: {train_Average_Cost[i]}, Test: {test_Average_Cost[i]}' )
            
            # input( 'Press any key to continue...' )
            
            print( '=' * 50 )
            print( f'Training Average Cost: {np.mean( train_Average_Cost )}' )
            print( f'Testing Average Cost: {np.mean( test_Average_Cost )}' )
            print( '*' * 50 )
            
            simulation_rec[f'{Historical_N}_{epsilon_N}_Train'] = train_Average_Cost
            simulation_rec[f'{Historical_N}_{epsilon_N}_Test'] = test_Average_Cost
            
            data['Historical_N'].append( Historical_N )
            data['Evaluation_N'].append( Evaluation_N )
            data['Num_Retailers'].append( Num_Retailers )
            data['f'].append( f )
            data['h'].append( h )
            data['c'].append( c )
            data['b'].append( b )
            data['epsilon_N'].append( epsilon_N )
            data['Train_Average_Cost'].append( np.mean( train_Average_Cost ) )
            data['Test_Average_Cost'].append( np.mean( test_Average_Cost ) )
            data['Date'].append( time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() ) )
            
            df = pd.DataFrame( data )
            df.to_csv( 'Result/Simulation_Result.csv', index = False )
            
            df = pd.DataFrame( data = simulation_rec )
            df.to_csv( 'Result/Simulation_Result_Rec.csv', index = False )