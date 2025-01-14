import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists( 'Figure/Comparison/Partial/Cost/Zoom_in' ):
    os.makedirs( 'Figure/Comparison/Partial/Cost/Zoom_in', exist_ok = True )

if not os.path.exists( 'Figure/Comparison/Partial/Service_level' ):
    os.makedirs( 'Figure/Comparison/Partial/Service_level', exist_ok = True )

if not os.path.exists( 'Figure/Comparison/Partial/Time' ):
    os.makedirs( 'Figure/Comparison/Partial/Time', exist_ok = True )

if not os.path.exists( 'Figure/Comparison/Partial/Combined' ):
    os.makedirs( 'Figure/Comparison/Partial/Combined', exist_ok = True )

# Set the global font size
plt.rcParams.update({'font.size': 20})

df = pd.read_csv( 'Result/Comparison/GridSearch_Average_Result.csv' )
epsilon_N_space = df['epsilon_N'].unique()
x = np.arange( len( epsilon_N_space ) )
x_ticks = [1, 11, 21, 31, 41]
x_labels = [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$']
Historical_N_space = [25, 50, 100] # 
omega_space1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] # 
omega_space2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # 
zoom_in = True
omega1 = True
omega2 = False

if omega1: 
    omega_space = omega_space1
else:
    omega_space = omega_space2

color_list = ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple']

for Historical_N in Historical_N_space:
    test_cost = np.zeros( ( len( omega_space ), len( epsilon_N_space ) ) )
    service_level = np.zeros( ( len( omega_space ), len( epsilon_N_space ) ) )
    time = np.zeros( ( len( omega_space ), len( epsilon_N_space ) ) )
    for omega_id, omega in enumerate( omega_space ):
        df_temp = df.loc[ ( df['Historical_N'] == Historical_N ) & ( df['omega'] == omega ) ]
        
        if omega == 0:
            train_cost = df_temp['Train_Cost'].to_numpy()
        
        test_cost[omega_id, :] = df_temp['Test_Cost'].to_numpy()
        service_level[omega_id, :] = df_temp['Service_Level'].to_numpy()
        time[omega_id, :] = df_temp['Time'].to_numpy()
    
    # # Cost + Service level
    # plt.figure( figsize = ( 8, 4 ) )
    # plt.xticks( x_ticks, x_labels )
    # for omega_id, omega in enumerate( omega_space ):
    #     if omega == 0:
    #         plt.plot( x, test_cost[omega_id, :], color = 'black', linestyle = '--', label = 'Test' )
    #     else:
    #         plt.plot( x, test_cost[omega_id, :], color = color_list[omega_id - 1], label = f'omega = {omega}' )
    # plt.plot( x, train_cost, color = 'violet', linestyle = '--', label = 'Train' )
    # plt.xlabel( 'epsilon_N' )
    # plt.xlim( 0, 42 )
    # plt.ylabel( 'Average Cost ($)' )
    # # plt.ylim( 203, 215 )
    # plt.ylim( 205, 215 )
    # plt.yticks( [205, 210, 215] )
    # plt.grid()
    # ax2 = plt.twinx()
    # ax2.set_ylim( 0, 100 )
    # ax2.set_yticks( [0, 50, 100] )
    # ax2.set_ylabel( 'Service Level (%)' )
    # for omega_id, omega in enumerate( omega_space ):
    #     if omega == 0:
    #         plt.plot( x, service_level[omega_id, :], linestyle = 'dotted', color = 'black', label = 'Prof.Bertsimas' )
    #     else:
    #         plt.plot( x, service_level[omega_id, :], linestyle = 'dotted', color = color_list[omega_id - 1], label = f'omega = {omega}' )
    # plt.title( f'N = {Historical_N}' )
    # plt.tight_layout()
    # plt.savefig( f'Figure/Comparison/Partial/Combined/N_{Historical_N}.jpg', dpi = 330 )
    
    # Cost
    plt.figure( figsize = ( 8, 4 ) )
    for omega_id, omega in enumerate( omega_space ):
        if omega == 0:
            plt.plot( x, test_cost[omega_id, :], color = 'black', linestyle = '--', label = 'Test' )
        else:
            plt.plot( x, test_cost[omega_id, :], color = color_list[omega_id - 1], label = f'omega = {omega}' )
    plt.plot( x, train_cost, color = 'violet', linestyle = '--', label = 'Train' )
    plt.xticks( x_ticks, x_labels )
    plt.xlim( 0, 42 )
    plt.ylim( 205, 213 )
    plt.xlabel( 'epsilon_N' )
    plt.ylabel( 'Average_Cost ($)' )
    plt.grid()
    plt.title( f'N = {Historical_N}' )
    plt.tight_layout()
    plt.savefig( f'Figure/Comparison/Partial/Cost/N_{Historical_N}.jpg', dpi = 330 )
    
    # # Zoom in cost
    # if zoom_in:
    #     plt.figure( figsize = ( 12, 12 ) )
    #     for omega_id, omega in enumerate( omega_space ):
    #         if omega == 0:
    #             plt.plot( x, test_cost[omega_id, :], color = 'black', linestyle = '--', label = 'Test' )
    #         else:
    #             plt.plot( x, test_cost[omega_id, :], color = color_list[omega_id - 1], label = f'omega = {omega}' )
    #     plt.plot( x, train_cost, color = 'violet', linestyle = '--', label = 'Train' )
    #     plt.xticks( x_ticks, x_labels )
    #     plt.ylim( 207, 210 )
    #     plt.xlabel( 'epsilon_N' )
    #     plt.ylabel( 'Average_Cost ($)' )
    #     plt.grid()
    #     plt.title( f'N = {Historical_N}' )
    #     plt.savefig( f'Figure/Comparison/Partial/Cost/Zoom_in/N_{Historical_N}.jpg' )
    
    # Service level
    plt.figure( figsize = ( 8, 4 ) )
    for omega_id, omega in enumerate( omega_space ):
        if omega == 0:
            plt.plot( x, service_level[omega_id, :], linestyle = 'dotted', color = 'black', label = 'Prof.Bertsimas' )
        else:
            plt.plot( x, service_level[omega_id, :], linestyle = 'dotted', color = color_list[omega_id - 1], label = f'omega = {omega}' )
    plt.xticks( x_ticks, x_labels )
    plt.xlim( 0, 42 )
    plt.ylim( 80, 100 )
    plt.xlabel( 'epsilon_N' )
    plt.ylabel( 'Service_Level (%)' )
    plt.grid()
    plt.title( f'N = {Historical_N}' )
    plt.tight_layout()
    plt.savefig( f'Figure/Comparison/Partial/Service_level/N_{Historical_N}.jpg', dpi = 330 )
    
    # plt.figure( figsize = ( 12, 12 ) )
    # for omega_id, omega in enumerate( omega_space ):
    #     if omega == 0:
    #         plt.plot( x, time[omega_id, :], linestyle = 'dotted', color = 'black', label = 'Prof.Bertsimas' )
    #     else:
    #         plt.plot( x, time[omega_id, :], linestyle = 'dotted', color = color_list[omega_id - 1], label = f'omega = {omega}' )
    # plt.xticks( x_ticks, x_labels )
    # plt.xlabel( 'epsilon_N' )
    # plt.ylabel( 'Computational Time(s)' )
    # plt.grid()
    # plt.title( f'N = {Historical_N}' )
    # plt.savefig( f'Figure/Comparison/Partial/Time/N_{Historical_N}.jpg' )