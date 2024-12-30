import matplotlib.pyplot as plt
import pandas as pd, numpy as np

# Set the global font size
plt.rcParams.update({'font.size': 20}) 

# df = pd.read_csv( 'Result/Simulation_Result_Multistage_30.csv' )
# df = df[df['Historical_N'] == 25]
# df['epsilon_N'] = df['epsilon_N'].round( 6 )
# df_epsilon = df['epsilon_N'].to_numpy()
# rec_df = pd.read_csv( 'Result/Simulation_Result_Multistage_Rec_30.csv' )

# hybrid_df = pd.read_csv( 'Result/Simulation_Result_Multistage_Hybrid_N25.csv' )
# hybrid_rec_df = pd.read_csv( 'Result/Simulation_Result_Multistage_Rec_Hybrid_N25.csv' )
# N = 25
# epsilon_N = hybrid_df['epsilon_N'].round( 6 ).tolist()
# idx = []
# for epsilon in epsilon_N:
#     idx.append( np.where( df_epsilon == epsilon )[0].tolist()[0] )

# partial_df = df.iloc[idx]
# original_test = partial_df['Test_Average_Cost'].to_numpy()
# hybrid_test = hybrid_df['Test_Average_Cost'].to_numpy()

# plt.figure( figsize = ( 10, 10 ) )
# x = np.arange( len( epsilon_N ) )
# plt.plot( x, original_test, color = "red", marker = 'o', label = "Original" )
# plt.plot( x, hybrid_test, color = "blue", marker = 'o', label = "Hybrid" )
# plt.xticks( x, epsilon_N )
# plt.xlabel( 'epsilon_N' )
# plt.ylabel( 'Average_Cost' )
# plt.legend( loc = 'best' )
# plt.grid()
# plt.title( f'N = {N}' )
# plt.savefig( f'Figure/Simulation_Result/Multistage_Compare/N_{N}.jpg' )
# plt.show()
# plt.close()

N = 100
df = pd.read_csv( 'Result/Simulation_Result_Multistage_30.csv' )
df = df[df['Historical_N'] == N]
df['epsilon_N'] = df['epsilon_N'].round( 6 )
df_epsilon = df['epsilon_N'].to_numpy()
rec_df = pd.read_csv( 'Result/Simulation_Result_Multistage_Rec_30.csv' )

hybrid_df = pd.read_csv( 'Result/Simulation_Result_Multistage_Hybrid_N{}.csv'.format( N ) )
hybrid_rec_df = pd.read_csv( 'Result/Simulation_Result_Multistage_Rec_Hybrid_N{}.csv'.format( N ) )
epsilon_N = hybrid_df['epsilon_N'].round( 6 ).tolist()
idx = np.where( df_epsilon == epsilon_N[0] )[0].tolist()[0]

partial_df = df.iloc[idx:idx + (len( epsilon_N ))]
original_test = partial_df['Test_Average_Cost'].to_numpy()
hybrid_test = hybrid_df['Test_Average_Cost'].to_numpy()

plt.figure( figsize = ( 10, 10 ) )
x = np.arange( len( epsilon_N ) )
plt.plot( x, original_test, color = "red", marker = 'o', label = "Original" )
plt.plot( x, hybrid_test, color = "blue", marker = 'o', label = "Hybrid" )
plt.xticks( x, epsilon_N )
plt.xlabel( 'epsilon_N' )
plt.ylabel( 'Average_Cost' )
plt.legend( loc = 'best' )
plt.grid()
plt.title( f'N = {N}' )
# plt.savefig( f'Figure/Simulation_Result/Multistage_Compare/N_{N}.jpg' )
plt.show()
plt.close()