from scipy.stats import truncnorm
import numpy as np
import warnings
warnings.filterwarnings( "ignore" )

def generate_truncated_normal_vector( mean, std_dev, lower_bound, size, random_seed ):
    """Generates a random vector from a truncated normal distribution.
    
    Parameters:
    ---------
    mean : float
        The mean of the truncated normal distribution.
    std_dev : float
        The standard deviation of the truncated normal distribution.
    lower_bound : float
        The lower bound of the truncated normal distribution.
    size : int
        The size of the random vector to generate.
    random_seed : int
        The random seed for the random number generator.
    
    Returns:
    ---------
    vector : array
        The random vector from the truncated normal distribution.
    """
    
    # Calculate the a and b parameters for the truncnorm distribution
    a, b = ( lower_bound - mean ) / std_dev, np.inf 
    
    # Generate the random vector
    vector = truncnorm.rvs( a, b, loc = mean, scale = std_dev, size = size, random_state = random_seed )
    return vector

def Demand_Data( R, random_seed ):
    """Generate the demand data for each retailer R at each stage.
    
    Parameters
    ----------
    R : int
        The number of retailers.
    random_seed : int
        The random seed for the random number generator.
    
    Returns
    -------
    demand : array
        The demand data for each retailer R at each stage.
    
    Reference
    ---------
    Dimitris Bertsimas, Shimrit Shtern, Bradley Sturt (2023) A Data-Driven Approach to Multistage Stochastic Linear
    Optimization. Management Science 69(1):51-74. https://doi.org/10.1287/mnsc.2022.4352
    Page 67, Section 7.2
    """
    
    demand = np.zeros( ( R, 2 ) )
    
    #! First stage demand
    # Generate truncated normal distribution with mean 6 and std 2.5 bounded below by 0.
    demand[:,0] = generate_truncated_normal_vector( mean = 6, std_dev = 2.5, lower_bound = 0, size = R, random_seed = random_seed )
    
    #! Second stage demand
    # Generate truncated normal distribution with mean 2(6 - demand[i, 0])^2 and std 2.5 bounded below by 0.
    for i in range( R ):
        demand[i, 1] = generate_truncated_normal_vector( mean = 2*( 6 - demand[i, 0] )**2, std_dev = 2.5, lower_bound = 0, size = 1, random_seed = random_seed )
    
    return demand

def Historical_Data( N, R, random_seed: int = 320426 ):
    """ Generate the historical data for constructing the uncertainty set.
    
    Parameters
    ----------
    N : int
        The number of historical data.
    R : int
        The number of retailers.
    
    Returns
    -------
    historical : array
        The historical data for constructing the uncertainty set.
    
    Reference
    ---------
    Dimitris Bertsimas, Shimrit Shtern, Bradley Sturt (2023) A Data-Driven Approach to Multistage Stochastic Linear
    Optimization. Management Science 69(1):51-74. https://doi.org/10.1287/mnsc.2022.4352
    Page 67, Section 7.2
    """
    historical = np.zeros( ( N, R, 2 ) )
    
    rng = np.random.default_rng( random_seed )
    
    for i in range( N ):
        historical[i] = Demand_Data( R = R, random_seed = rng.integers( low = 0, high = 2**32 ) )
    
    return historical #! Index format: ( N, R, T )

def Evaluation_Data( N, R, random_seed: int = 320426 ):
    """ Generate the evaluation data for evaluating the performance of the method.
    
    Parameters
    ----------
    N : int
        The number of evaluation data.
    R : int
        The number of retailers.
    
    Returns
    -------
    evaluation : array
        The evaluation data for evaluating the performance of the method.
    
    Reference
    ---------
    Dimitris Bertsimas, Shimrit Shtern, Bradley Sturt (2023) A Data-Driven Approach to Multistage Stochastic Linear
    Optimization. Management Science 69(1):51-74. https://doi.org/10.1287/mnsc.2022.4352
    Page 67, Section 7.2
    """
    evaluation = np.zeros( ( N, R, 2 ) )
    
    rng = np.random.default_rng( random_seed )
    
    for i in range( N ):
        evaluation[i] = Demand_Data( R = R, random_seed = rng.integers( low = 0, high = 2**32 ) )
    
    return evaluation #! Index format: ( N, R, T )

def Multistage_Inventory_Data( N, T, alpha, mu, bound, random_seed: int = 320426 ):
    """ Generate the simulation data for multistage inventory problem.
    
    Parameters
    ----------
    T : int
        The number of periods.
    mu : float
        
    bound : float
        The demand bound.
    
    Returns
    ----------
    demand : array
        The demand data for multistage inventory problem.
    
    Reference
    ----------
    Dimitris Bertsimas, Shimrit Shtern, Bradley Sturt (2023) A Data-Driven Approach to Multistage Stochastic Linear
    Optimization. Management Science 69(1):51-74. https://doi.org/10.1287/mnsc.2022.4352
    Page 71, Section 8.2
    """
    
    # Independent random variables distributed uniformly over [-bound, bound]
    rng = np.random.default_rng( random_seed )
    
    demand = np.zeros( ( N, T ) )
    for data_idx in range( N ):
        sigma = rng.uniform( low = -bound, high = bound, size = T )
        for period_idx in range( T ):
            if period_idx == 0:
                demand[data_idx, period_idx] = sigma[period_idx] + mu
            else:
                demand[data_idx, period_idx] += ( sigma[period_idx] + sum( alpha * sigma[previous_idx] for previous_idx in range( period_idx ) ) + mu )
    return demand
