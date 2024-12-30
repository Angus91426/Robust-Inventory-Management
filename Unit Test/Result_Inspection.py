import pandas as pd

c = 0.25
h = 0.05
b = 0.5
f = 0

result = pd.read_csv( 'Result/Test.csv' )
for retailer_idx in range( 1, 4 ):
    result[f'T1_Ending_{retailer_idx}'] = [max( [0, (result[f'Q1_{retailer_idx}'].tolist()[i] - result[f'T1_Demand_{retailer_idx}'].tolist()[i]) ] ) for i in range( len( result ) )]
    result[f'T1_Backorder_{retailer_idx}'] = [max( [0, (result[f'T1_Demand_{retailer_idx}'].tolist()[i] - result[f'Q1_{retailer_idx}'].tolist()[i]) ] ) for i in range( len( result ) )]
for retailer_idx in range( 1, 4 ):
    result[f'Ending_{retailer_idx}'] = [result[f'T1_Ending_{retailer_idx}'].tolist()[i] + result[f'Q2_{retailer_idx}'].tolist()[i] - result[f'T2_Demand_{retailer_idx}'].tolist()[i] for i in range( len( result ) )]
    result[f'Ending_Backorder_{retailer_idx}'] = [max( [0, -result[f'Ending_{retailer_idx}'].tolist()[i]] ) for i in range( len( result ) )]
    result[f'Ending_Inventory_{retailer_idx}'] = [max( [0, result[f'Ending_{retailer_idx}'].tolist()[i]] ) for i in range( len( result ) )]

result['Total_Q1'] = result['Q1_1'] + result['Q1_2'] + result['Q1_3']
result['Total_Q2'] = result['Q2_1'] + result['Q2_2'] + result['Q2_3']
result['Total_Backorder_T1'] = result['T1_Backorder_1'] + result['T1_Backorder_2'] + result['T1_Backorder_3']
result['Total_Ship'] = result['z_1'] + result['z_2'] + result['z_3']
result['Total_Backorder'] = result['Ending_Backorder_1'] + result['Ending_Backorder_2'] + result['Ending_Backorder_3']
result['Total_Inventory'] = result['Ending_Inventory_1'] + result['Ending_Inventory_2'] + result['Ending_Inventory_3']
result['Production_T1'] = result['Warehouse'] + result['Total_Q1']
result['Holding_Warehouse'] = result['Warehouse'] - result['Total_Q2']

result['Production_Cost_T1'] = c * result['Production_T1']
result['Holding_Cost_Warehouse'] = h * result['Holding_Warehouse']
result['Backorder_Cost_T1'] = b * result['Total_Backorder_T1']
result['Shipping_Cost'] = f * result['Total_Ship']
result['Backorder_Cost'] = b * result['Total_Backorder']
result['Holding_Cost'] = h * result['Total_Inventory']
result['Total_Cost'] = result['Production_Cost_T1'] + result['Holding_Cost_Warehouse'] + result['Backorder_Cost_T1'] + result['Shipping_Cost'] + result['Backorder_Cost'] + result['Holding_Cost']
result.to_csv( 'Result/Test_new.csv', index = False )