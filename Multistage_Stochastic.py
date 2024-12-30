from re import I
from unicodedata import name
import numpy as np, pandas as pd, gurobipy as gp
import Experiment_Data as Experiment_Data
import Time_Series_Forecasting as Time_Series_Forecasting

import os, time, math
import warnings
warnings.filterwarnings(  "ignore"  )

def Uncertainty_Bound( historical, epsilon_N, idx_j ):
    """ Construct the upper bound (u) and lower bound (l) for each stage.
    
    Parameters
    ----------
    historical : array
        The historical data for constructing the uncertainty set.
    epsilon : float
        The epsilon value for constructing the uncertainty set.
    idx_j : int
        The index of the historical data.
    
    Returns
    ----------
    ub : array
        The maximum values of each stage.
    lb : array
        The minimum values of each stage.
    
    """
    
    historical_j = historical[idx_j, :]
    ub = historical_j + epsilon_N
    lb = np.maximum( historical_j - epsilon_N, 0 )
    
    return ub, lb

def Solve_Model( N, T, historical, epsilon_N, c, h, x_bar, b, OutputFlag: bool = False ):
    with gp.Env( empty = True ) as env:
        env.setParam( 'OutputFlag', 1 ) if OutputFlag else env.setParam( 'OutputFlag', 0 )
        env.start()
        with gp.Model( "Model", env = env ) as model:
            
            #! Decision variables
            #* Original problem: x, X, y, Y, i, I
            # Production
            x = model.addVars( T, name = 'x', vtype = gp.GRB.CONTINUOUS )
            X = model.addVars( T, T - 1, name = 'X', vtype = gp.GRB.CONTINUOUS )
            
            # Holding / Backorder
            y = model.addVars( N, T + 1, name = 'y', vtype = gp.GRB.CONTINUOUS )
            
            # Dual variables
            alpha = model.addVars( T, name = 'alpha', vtype = gp.GRB.CONTINUOUS )
            beta = model.addVars( T, name = 'beta', vtype = gp.GRB.CONTINUOUS )
            
            #! Objective function
            obj = 0
            for j in range( N ):
                ub, lb = Uncertainty_Bound( historical, epsilon_N, j )
                obj += gp.quicksum( ( ( ( c[t] * x[t] ) + ( alpha[t] * ub[t] ) - ( beta[t] * lb[t] ) ) + y[j, t + 1] ) for t in range( T ) )
            obj = obj / N # Average over the uncertainty set
            
            model.setObjective( obj, gp.GRB.MINIMIZE )
            
            #! Constraints
            
            #* Constraints for dual variables
            for t in range( T ):
                model.addConstr( alpha[t] - beta[t] == gp.quicksum( ( c[s] * X[s, t] ) for s in range( t + 1, T ) ) )
            
            #* Constraints for holding & backorder
            for j in range( N ):
                ub, lb = Uncertainty_Bound( historical, epsilon_N, j )
                for t in range( T ):
                    # Holding
                    model.addConstr( y[j, t + 1] >= h[t] * gp.quicksum( ( ( x[i] + gp.quicksum( ( X[i, s] * ub[s] ) for s in range( i ) ) ) - lb[i] ) for i in range( t + 1 ) ), name = 'Holding' )
                    # Backorder
                    model.addConstr( y[j, t + 1] >= b[t] * gp.quicksum( (  ub[i] - ( x[i] + gp.quicksum( ( X[i, s] * lb[s] ) for s in range( i ) ) ) ) for i in range( t + 1 ) ), name = 'Backorder' )
            
            #* Capacity bound
            for j in range( N ):
                ub, lb = Uncertainty_Bound( historical, epsilon_N, j )
                for t in range( T ):
                    model.addConstr( x[t] + gp.quicksum( ( X[t, s] * ub[s] ) for s in range( t ) ) >= 0, name = 'Capacity_lb' )
                    model.addConstr( x[t] + gp.quicksum( ( X[t, s] * ub[s] ) for s in range( t ) ) <= x_bar[t], name = 'Capacity_ub' )
            
            model.update()
            
            #! Solve the model
            start = time.time()
            # model.write( 'Multistage_Stochastic.lp' )
            model.optimize()
            execution_time = time.time() - start
            
            if model.status == gp.GRB.OPTIMAL:
                # print( f'Optimal solution found in {execution_time:.4f} seconds' )
                x_opt = [x[i].x for i in range( T )]
                X_opt = np.zeros( ( T, T - 1 ) )
                for t in range( T ):
                    for s in range( t ):
                        X_opt[t, s] = X[t, s].x
                
                sol = []
                for v in model.getVars():
                    sol.append( [v.varName, v.x] )
                sol_df = pd.DataFrame( sol, columns = ['Variable', 'Value'] )
                return x_opt, X_opt, model.objVal, sol_df
            
            elif model.status == gp.GRB.INFEASIBLE:
                print( 'Model is infeasible.' )
                model.computeIIS()
                model.write( 'Multistage_Stochastic.ilp' )
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

def Generate_Decision( N, T, demand, x_opt, X_opt, capacity ):
    """ Generate the decision rules for each stage.
    
    Parameters
    ----------
    """
    decision_x = np.zeros( ( N, T ) )
    for j in range( N ):
        for t in range( T ):
            decision_x[j, t] = x_opt[t] + sum( X_opt[t, s] * demand[j, s] for s in range( t ) )
            if decision_x[j, t] > capacity:
                decision_x[j, t] = capacity
    return decision_x

def Compute_Cost( N, T, demand, decision_x, c, h, b ):
    """
    
    """
    # Inventory
    I = np.zeros( ( N, T + 1 ) )
    for j in range( N ):
        for t in range( T ):
            I[j, t + 1] = I[j, t] + decision_x[j, t] - demand[j, t]
    
    Cost = np.zeros( N )
    service_satisfy = np.zeros( N )
    for j in range( N ):
        for t in range( T ):
            Cost[j] += ( ( c[t] * decision_x[j, t] ) + max( [( h[t] * I[j, t + 1] ), ( -b[t] * I[j, t + 1] )] ) )
            
            if I[j, t + 1] >= 0:
                service_satisfy[j] += 1
        service_satisfy[j] = ( service_satisfy[j] / T ) * 100
    
    Average_Cost = np.sum( Cost ) / N
    Average_Service_Level = np.sum( service_satisfy ) / N
    return Average_Cost, Average_Service_Level

def Construct_Experiment( Historical_N, Evaluation_N, evaluation, T, alpha, mu, bound, \
                            capacity, c_t, h_t, b_t, b_T, epsilon_N, random_seed ):
    # Construct historical training data
    historical = Experiment_Data.Multistage_Inventory_Data( Historical_N, T, alpha, mu, bound, random_seed )
    
    # Construct each cost for each stage
    c, h, b, x_bar = np.zeros( T ), np.zeros( T ), np.zeros( T ), np.zeros( T )
    for t in range( T ):
        x_bar[t] = capacity
        c[t] = c_t
        h[t] = h_t
        b[t] = b_t
    b[-1] = b_T
    
    # Solve the optimization problem
    x_opt, X_opt, train_Average_Cost, sol_df = Solve_Model( Historical_N, T, historical, epsilon_N, c, h, x_bar, b )
    
    # Generate the decision rule for evaluation data
    test_decision = Generate_Decision( Evaluation_N, T, evaluation, x_opt, X_opt, capacity )
    
    # Compute the cost for evaluation data
    test_Average_Cost, Average_Service_Level = Compute_Cost( Evaluation_N, T, evaluation, test_decision, c, h, b )
    
    return train_Average_Cost, test_Average_Cost, sol_df

def Construct_Hybrid_Experiment( Historical_N, Evaluation_N, evaluation, T, alpha, mu, bound, \
                            capacity, c_t, h_t, b_t, b_T, epsilon_N, random_seed, hybrid_uncertainty = True, omega = 0.5, n_estimators = 50, max_depth = 10 ):
    # Construct historical training data
    historical = Experiment_Data.Multistage_Inventory_Data( Historical_N, T, alpha, mu, bound, random_seed )
    if hybrid_uncertainty:
        new_historical = np.zeros( ( Historical_N, T ) )
        MSE_rec = np.zeros( Historical_N )
        for j in range( Historical_N ):
            hybrid_uncertainty, _, hybrid_MSE = Time_Series_Forecasting.Construct_Hybrid_Uncertainty( demand = historical[j], omega = omega, random_seed = random_seed, \
                                                                                                        n_estimators = n_estimators, max_depth = max_depth )
            new_historical[j] = hybrid_uncertainty
            MSE_rec[j] = hybrid_MSE
        average_hybrid_MSE = np.mean( MSE_rec )
    else:
        new_historical = historical
        average_hybrid_MSE = None
    
    # Construct each cost for each stage
    c, h, b, x_bar = np.zeros( T ), np.zeros( T ), np.zeros( T ), np.zeros( T )
    for t in range( T ):
        x_bar[t] = capacity
        c[t] = c_t
        h[t] = h_t
        b[t] = b_t
    b[-1] = b_T
    
    # Solve the optimization problem
    x_opt, X_opt, train_Average_Cost, sol_df = Solve_Model( Historical_N, T, new_historical, epsilon_N, c, h, x_bar, b )
    
    # Generate the decision rule for evaluation data
    test_decision = Generate_Decision( Evaluation_N, T, evaluation, x_opt, X_opt, capacity )
    
    # Compute the cost for evaluation data
    test_Average_Cost, Average_Service_Level = Compute_Cost( Evaluation_N, T, evaluation, test_decision, c, h, b )
    
    return train_Average_Cost, test_Average_Cost, sol_df, average_hybrid_MSE