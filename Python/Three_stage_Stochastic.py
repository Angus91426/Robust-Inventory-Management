import numpy as np, pandas as pd, gurobipy as gp
import Python.Experiment_Data as Experiment_Data

import os, time, math
import warnings
warnings.filterwarnings(  "ignore"  )

def Partition( N, R, Historical ):
    """ Construct the partition based on the historical data efficiently.
    
    Parameters
    ----------
    N : int
        The number of historical data.
    historical : array
        The historical data for constructing the uncertainty set.
    R : int
        The number of retailers.
    
    Returns
    ----------
    Partition_Path : array
        The partition path for each retailer.
    
    Reference
    ----------
    Dimitris Bertsimas, Shimrit Shtern, Bradley Sturt ( 2023 ) A Data-Driven Approach to Multistage Stochastic Linear
    Optimization. Management Science 69( 1 ):51-74. https://doi.org/10.1287/mnsc.2022.4352
    Appendix J.2
    """
    
    # Initialization
    M = math.ceil( N ** ( 1 / R ) )
    Partition = [( 0, np.inf )]
    Retailer_Partition = [] # Partition (lb and ub) for each retailer
    Partition_Path = []
    
    for retailer_idx in range( R ):
        add_Partition = [] # List of the new partitions to be added
        remove_Partition = [] # List of the partitions to be removed
        
        for Partition_idx, P in enumerate( Partition ):
            if len( Retailer_Partition ) == 0: # First retailer
                J = list( range( N ) )
            elif len( Retailer_Partition ) == 1: # Second retailer
                J = np.where( ( Historical[:, retailer_idx - 1, 0] >= P[0] ) & ( Historical[:, retailer_idx - 1, 0] <= P[1] ) )[0]
            else: # If retailer > 2, find the historical data that falls into the current partition
                current_path = Partition_Path[Partition_idx]
                J = []
                for path_idx in range( len( current_path ) ):
                    if J == []:
                        J = [j_id for j_id in range( N ) if ( Historical[j_id, path_idx, 0] >= current_path[path_idx][0] ) & ( Historical[j_id, path_idx, 0] <= current_path[path_idx][1] ) ]
                    else:
                        J = [j_id for j_id in J if ( Historical[j_id, path_idx, 0] >= current_path[path_idx][0] ) & ( Historical[j_id, path_idx, 0] <= current_path[path_idx][1] ) ]
            
            Partial_historical = Historical[J]
            arg_sort_j = np.argsort( Partial_historical[:, retailer_idx, 0] )
            remove_Partition.append( P )
            
            # Setting parameters
            K = min( M, len( arg_sort_j ) )
            k = [0]
            threshold = 0
            for l in range( 1, K + 1 ):
                k.append( max( math.ceil( ( len( arg_sort_j ) * l ) / K ), ( k[l - 1] + 1 ) ) )
                k[l] = math.floor( k[l] )
                if k[l] < len( arg_sort_j ):
                    # Threshold is the average of the two demands
                    new_threshold = ( Partial_historical[arg_sort_j[k[l]], retailer_idx, 0] + Partial_historical[arg_sort_j[k[l] - 1], retailer_idx, 0] ) / 2
                    if threshold == 0: # The first partition
                        add_Partition.append( ( 0, new_threshold ) )
                        threshold = new_threshold
                    else:
                        add_Partition.append( ( threshold, new_threshold ) )
                        threshold = new_threshold
                else: # The last partition
                    add_Partition.append( ( threshold, np.inf ) )
        
        # Update the partition for current retailer
        Partition = list( set( Partition ) - set( remove_Partition ) ) + add_Partition
        
        Retailer_Partition.append( Partition )
        
        if len( Retailer_Partition ) >= 1:
            Partition_Path = Partition_Transformation( Retailer_Partition )
    
    return Partition_Path

def Partition_Transformation( Retailer_Partition ):
    Num_Retailers = len( Retailer_Partition )
    temp = []
    Transform_Partition = []
    for retailer_idx in range( Num_Retailers - 1 ):
        reshape = []
        Transform_temp = [Retailer_Partition[retailer_idx]]
        for j in range( len( Retailer_Partition[retailer_idx + 1] ) ):
            temp.append( Retailer_Partition[retailer_idx + 1][j] )
            if Retailer_Partition[retailer_idx + 1][j][1] == np.inf:
                reshape.append( temp )
                temp = []
        Transform_temp.append( reshape )
        
        split = []
        for i in range( len( Transform_temp[0] ) ):
            for j in range( len( Transform_temp[1][i] ) ):
                split.append( [Transform_temp[0][i], Transform_temp[1][i][j]] )
        
        if Transform_Partition == []:
            Transform_Partition = split
        else:
            new_concat = []
            for concat_idx in range( len( Transform_Partition ) ):
                for split_idx in range( len( split ) ):
                    if Transform_Partition[concat_idx][-1] == split[split_idx][0]:
                        new_concat.append( Transform_Partition[concat_idx] + [split[split_idx][1]] )
            Transform_Partition = new_concat
    
    return Transform_Partition

def Construct_Kj( R, Historical, Partition_Path, epsilon_N ):
    """ Receive the indices of the partitions that each historical data belongs to.
    
    Parameters
    ----------
    historical : array
        The historical data for constructing the uncertainty set.
    R : int
        The number of retailers.
    Partition_Path : array
        The partition path for each retailer.
    
    Returns
    ----------
    Kj : array
        The indices of the partitions that each historical data belongs to.
    
    """
    N = len( Historical )
    Kj = []
    for data_idx in range( N ):
        temp = []
        current_data = Historical[data_idx, :, 0]
        uncertainty_ub = current_data + epsilon_N
        uncertainty_lb = np.maximum( current_data - epsilon_N, 0 )
        for path_idx in range( len( Partition_Path ) ):
            intercept = True
            for retailer_idx in range( R ):
                if uncertainty_ub[retailer_idx] < Partition_Path[path_idx][retailer_idx][0] or\
                    uncertainty_lb[retailer_idx] > Partition_Path[path_idx][retailer_idx][1]:
                    intercept = False
            
            if intercept:
                temp.append( path_idx )
        
        Kj.append( temp )
    return Kj

def Uncertainty_Bounds( Historical, epsilon_N, Partition_Path, idx_j, idx_k, idx_r ):
    """ Receive the maximum and minimum values of each interception of the uncertainty set and partition.
    
    Parameters
    ----------
    historical : array
        The historical data for constructing the uncertainty set.
    epsilon : float
        The epsilon value for constructing the uncertainty set.
    Partition_Path : array
        The partition path for each retailer.
    idx_j : int
        The index of the historical data.
    idx_k : int
        The index of the partition.
    idx_r : int
        The index of the retailer.
    
    Returns
    ----------
    uncertainty_max : array
        The maximum values of each stage.
    uncertainty_min : array
        The minimum values of each stage.
    """
    
    historical_j = Historical[idx_j, idx_r, :]
    partition_k = Partition_Path[idx_k]
    
    uncertainty_max, uncertainty_min = np.zeros( 2 ), np.zeros( 2 )
    # Stage 1 consider the bound from the partition
    uncertainty_max[0] = min( historical_j[0] + epsilon_N, partition_k[0][1] )
    uncertainty_min[0] = max( historical_j[0] - epsilon_N, partition_k[0][0] )
    # Stage 2 no limit
    uncertainty_max[1] = historical_j[1] + epsilon_N
    uncertainty_min[1] = max( 0, historical_j[1] - epsilon_N ) # Avoid negative values
    return uncertainty_max, uncertainty_min

def Solve_MIO( N, R, Historical, Partition_Path, epsilon_N, Kj, f, h, c, b, OutputFlag: bool = False ):
    """ Solve the MIO problem to receive the optimal solution for decision rules for each stage.
    
    Parameters
    ----------
    N : int
        The number of historical data points.
    R : int
        The number of retailers.
    Historical : array
        The historical data for constructing the uncertainty set.
    Partition_Path : array
        The partition path for each retailer.
    uncertainty_max : array
        The maximum values of each uncertainty set.
    uncertainty_min : array
        The minimum values of each uncertainty set.
    f : int
        The fixed shipping cost.
    h : int
        The holding cost.
    c : int
        The production cost.
    b: int
        The back order cost.
    OutputFlag : bool
        Whether to print the solving process of gurobi. Default is False.
    
    Returns
    ----------
    Q1_0: float
        The optimal decision rule for the warehouse in the first stage.
    Q1_opt: array
        The optimal decision rule for each retailer in the first stage.
    Q2_opt: array
        The optimal decision rule for each retailer in the second stage.
    z_opt: array
        The optimal decision rule for each retailer in the second stage.
    model.objVal: float
        The optimal objective function value.
    
    Reference
    ----------
    Dimitris Bertsimas, Shimrit Shtern, Bradley Sturt ( 2023 ) A Data-Driven Approach to Multistage Stochastic Linear
    Optimization. Management Science 69( 1 ):51-74. https://doi.org/10.1287/mnsc.2022.4352
    Page 67, Section 7.2, Problem (9)
    """
    
    with gp.Env( empty = True ) as env:
        env.setParam( 'OutputFlag', 1 ) if OutputFlag else env.setParam( 'OutputFlag', 0 )
        env.start()
        with gp.Model( "MIO", env = env ) as model:
            #! Parameters
            # Initialize the bi-M constant as the maximum value of demand in the historical data.
            M = 5 * np.max( Historical )
            K = len( Partition_Path )
            
            #! Decision variables
            # Q1_0: Decision rule for the warehouse in the first stage (t = 1)
            Q1_0 = model.addVar( name = "Warehouse", vtype = gp.GRB.CONTINUOUS, lb = 0 )
            
            # Q1: Decision rule for each retailer (r) in the first stage (t = 1)
            Q1 = model.addVars( R, name = "Q1", vtype = gp.GRB.CONTINUOUS, lb = 0 )
            
            # Q2: Decision rule for each retailer (r) for each partition path (k).
            Q2 = model.addVars( N, R, name = "Q2", vtype = gp.GRB.CONTINUOUS, lb = 0 )
            
            # z: Decision rule for each retailer (r) for each partition path (k).
            z = model.addVars( N, R, name = "z", vtype = gp.GRB.BINARY )
            
            # v: Auxiliary variable for second stage objective function (holding + back order + shipping) for each retailer (r) and historical data (j).
            v = model.addVars( N, R, name = "v", vtype = gp.GRB.CONTINUOUS )
            
            # u: Auxiliary variable for second stage objective function (holding + back order) for each retailer (r) and historical data (j) and uncertainty set (k).
            u = model.addVars( N, K, R, name = "u", vtype = gp.GRB.CONTINUOUS )
            
            #! Objective function
            obj = 0
            obj += ( c * ( Q1_0 + gp.quicksum( Q1[r] for r in range( R ) ) ) ) # Production cost
            obj += ( h * Q1_0 ) # Holding cost for the warehouse
            obj += ( ( gp.quicksum( v[j, r] for j in range( N ) for r in range( R ) ) ) / N ) # Average total cost of second stage for all uncertainty set
            
            model.update()
            
            model.setObjective( obj, gp.GRB.MINIMIZE )
            
            #! Constraints
            # Shipping quantity constraint (Total shipping quantity must less than or equal to the quantity in the warehouse)
            for k in range( K ):
                model.addConstr( gp.quicksum( Q2[k, r] for r in range( R ) ) <= Q1_0 )
            
            # Auxiliary variable constraint (For v_(j, r))
            for j in range( N ):
                for r in range( R ):
                    for k in Kj[j]:
                        model.addConstr( v[j, r] >= ( u[j, k, r] + ( f * z[k, r] ) ) )
            
            # Auxiliary variable constraint (For u_(j, k, r))
            for j in range( N ):
                for r in range( R ):
                    for k in Kj[j]:
                        ub, lb = Uncertainty_Bounds( Historical = Historical, epsilon_N = epsilon_N, Partition_Path = Partition_Path, idx_j = j, idx_k = k, idx_r = r )
                        
                        #* Situation 1: With back order for both first and second stage
                        model.addConstr( u[j, k, r] >= ( b * ( ub[1] + ub[0] - Q2[k, r] - Q1[r] ) ) - ( h * Q2[k, r] ), name = 'Situation 1' )
                        
                        #* Situation 2: With inventory for both first and second stage
                        model.addConstr( u[j, k, r] >= ( h * ( Q1[r] - lb[0] - lb[1] ) ), name = 'Situation 2' )
                        
                        #* Situation 3: Back order for first stage and inventory for second stage
                        model.addConstr( u[j, k, r] >= ( b * ( ub[0] - Q1[r] ) ) - ( h * lb[1] ), name = 'Situation 3' )
            
            # Big-M constraint
            for k in range( K ):
                for r in range( R ):
                    model.addConstr( z[k, r] * M >= Q2[k, r] )
            
            #! Solve the model
            start = time.time()
            # model.write( 'MIO.lp' )
            model.optimize()
            execution_time = time.time() - start
            
            if model.status == gp.GRB.OPTIMAL:
                print( f'Optimal solution found in {execution_time:.4f} seconds' )
                # Receive the optimal solution
                Q1_0 = Q1_0.x
                Q1_opt = np.array( [ Q1[r].x for r in range( R ) ] )
                Q2_opt = np.array( [ [ Q2[k, r].x for r in range( R ) ] for k in range( K ) ] )
                z_opt = np.array( [ [ z[k, r].x for r in range( R ) ] for k in range( K ) ] )
                return Q1_0, Q1_opt, Q2_opt, z_opt, model.objVal
            
            elif model.status == gp.GRB.INFEASIBLE:
                print( 'Model is infeasible.' )
                return None
            
            elif model.status == gp.GRB.INF_OR_UNBD:
                print( 'Model is infeasible or unbounded.' )
                return None
            
            elif model.status == gp.GRB.UNBOUNDED:
                print( 'Model is unbounded.' )
                return None
            
            else:
                print( 'Model is not solved.' )
                return None

def Generate_Q2_Sol( N, R, Q2_pool, z_pool, T1_demand, partition_path ):
    """ Generate the solution of the decision rule for the second stage based on the demand of the first stage.
    
    Parameters
    ----------
    Q2_pool : array
        The pool of decision rule for the second stage of each uncertainty set.
    z_pool : array
        The pool decision rule for the second stage of each uncertainty set.
    T1_demand : array
        The demand of the first stage.
    
    Returns
    ----------
    Q2_sol : array
        The solution of the decision rule for the second stage based on the demand of the first stage.
    z_sol : array
        The solution of the decision rule for the second stage based on the demand of the first stage.
    
    """
    Q2_sol, z_sol = np.zeros( ( N, R ) ), np.zeros( ( N, R ) )
    for data_idx in range( N ):
        for partition_idx in range( len( partition_path ) ):
            correct = 0
            for retailer_idx in range( R ):
                if T1_demand[data_idx, retailer_idx] >= partition_path[partition_idx][retailer_idx][0] and T1_demand[data_idx, retailer_idx] <= partition_path[partition_idx][retailer_idx][1]:
                    correct += 1
            
            if correct == R:
                Q2_sol[data_idx, :] = Q2_pool[partition_idx, :]
                z_sol[data_idx, :] = z_pool[partition_idx, :]
                break
    
    return Q2_sol, z_sol

def Compute_Cost( N, R, Q1_0, Q1, Q2, z, f, h, c, b, Demand ):
    """ Compute the total cost by using the optimal decision rule.
    
    Parameters
    ----------
    Q1_0: array
        The decision rule of the warehouse for the first stage.
    Q1 : array
        The decision rule for the first stage.
    Q2 : array
        The decision rule for the second stage.
    z : array
        The decision rule for the second stage.
    f : int
        The fixed shipping cost.
    h : int
        The holding cost.
    c : int
        The production cost.
    b: int
        The back order cost.
    Demand : array
        The demand data for evaluation.
    
    Returns
    ----------
    Average_Cost : float
        The average cost over all evaluation data.
    
    """
    Ending_Inventory = np.zeros( ( N, R ) )
    for data_idx in range( N ):
        for retailer_idx in range( R ):
            Ending_Inventory[data_idx, retailer_idx] = max( 0, Q1[retailer_idx] - Demand[data_idx, retailer_idx, 0] ) + \
                                                        Q2[data_idx, retailer_idx] - Demand[data_idx, retailer_idx, 1]
    
    Cost = np.zeros( N )
    for data_idx in range( N ):
        # Production cost
        Cost[data_idx] += c * ( Q1_0 + sum( Q1[retailer_idx] for retailer_idx in range( R ) ) )
        
        # Holding for warehouse
        Cost[data_idx] += h * ( Q1_0 - sum( Q2[data_idx, retailer_idx] for retailer_idx in range( R ) ) )
        
        # Back order for stage one
        Cost[data_idx] += b * ( sum( max( 0, Demand[data_idx, retailer_idx, 0] - Q1[retailer_idx] ) for retailer_idx in range( R ) ) )
        
        # Shipping cost
        Cost[data_idx] += f * sum( z[data_idx, retailer_idx] for retailer_idx in range( R ) )
        
        # Back order for stage two
        Cost[data_idx] += b * ( sum( max( 0, -Ending_Inventory[data_idx, retailer_idx] ) for retailer_idx in range( R ) ) )
        
        # Holding for stage two
        Cost[data_idx] += h * ( sum( max( 0, Ending_Inventory[data_idx, retailer_idx] ) for retailer_idx in range( R ) ) )
    
    # Compute the average cost over all evaluation data
    Average_Cost = np.mean( Cost )
    
    return Average_Cost

def Construct_Experiment( Historical_N, Num_Retailers, Evaluation_N, evaluation, f, h, c, b, epsilon_N, random_seed, save_result: bool = False, save_path: str = "" ):
    """ Construct the experiment data for the MIO problem.
    
    Parameters
    ----------
    Historical_N : int
        The number of historical data.
    Num_Retailers : int
        The number of retailers.
    Evaluation_N : int
        The number of evaluation data.
    f : int
        The fixed shipping cost.
    h : int
        The holding cost.
    c : int
        The production cost.
    b: int
        The back order cost.
    epsilon_N : float
        The uncertainty parameter.
    random_seed : int
        The random seed for the data generation.
    save_result : bool, optional
        Whether to save the result, by default False.
    save_path : str, optional
        The path to save the result, by default None.
    
    Returns
    ----------
    training_Cost : float
        The cost for training dataset. (Objective function value of problem (9))
    test_Average_Cost : float
        The average cost for testing dataset.
    """
    historical = Experiment_Data.Historical_Data( N = Historical_N, R = Num_Retailers, random_seed = random_seed )
    
    Partition_Path = Partition( N = Historical_N, R = Num_Retailers, Historical = historical )
    
    Kj = Construct_Kj( R = Num_Retailers, Historical = historical, Partition_Path = Partition_Path, epsilon_N = epsilon_N )
    
    Q1_0, Q1_opt, Q2_opt, z_opt, training_Cost = Solve_MIO( N = Historical_N, R = Num_Retailers, Historical = historical, Partition_Path = Partition_Path, epsilon_N = epsilon_N, \
                                                                Kj = Kj, f = f, h = h, c = c, b = b )
    
    test_Q2_sol, test_z_sol = Generate_Q2_Sol( N = Evaluation_N, R = Num_Retailers, Q2_pool = Q2_opt, 
                                            z_pool = z_opt, T1_demand = evaluation[:, :, 0], partition_path = Partition_Path )
    
    test_Average_Cost = Compute_Cost( N = Evaluation_N, R = Num_Retailers, Q1_0 = Q1_0, Q1 = Q1_opt, Q2 = test_Q2_sol, \
                                        z = test_z_sol, f = f, h = h, c = c, b = b, Demand = evaluation )
    
    if save_result:
        result_data = {}
        result_data['Warehouse'] = [Q1_0 for _ in range( Evaluation_N )]
        for retailer_idx in range( Num_Retailers ):
            result_data[f'Q1_{retailer_idx + 1}'] = [Q1_opt[retailer_idx] for _ in range( Evaluation_N )]
        for retailer_idx in range( Num_Retailers ):
            result_data[f'T1_Demand_{retailer_idx + 1}'] = evaluation[:, retailer_idx, 0]
        for retailer_idx in range( Num_Retailers ):
            result_data[f'Q2_{retailer_idx + 1}'] = test_Q2_sol[:, retailer_idx]
        for retailer_idx in range( Num_Retailers ):
            result_data[f'z_{retailer_idx + 1}'] = test_z_sol[:, retailer_idx]
        for retailer_idx in range( Num_Retailers ):
            result_data[f'T2_Demand_{retailer_idx + 1}'] = evaluation[:, retailer_idx, 1]
        
        df = pd.DataFrame( result_data )
        df.to_csv( save_path, index = False )
        
        Q2_pool = pd.DataFrame( Q2_opt )
        Q2_pool.to_csv( 'Result/Q2_Pool.csv', index = False )
    
    return training_Cost, test_Average_Cost