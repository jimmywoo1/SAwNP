import sys
sys.path.append('./np')
sys.path.append('./np/neural_process_models')
sys.path.append('./np/misc')

import numpy as np
import pandas as pd


def simulate(s: np.ndarray, reps: int) -> list:
    '''
    vectorized SAN simulation
    s: (num_scenarios, 5)
    '''
    sample_paths = []
    s = np.array(s)

    for rep in range(reps):
        rand = np.random.RandomState(rep)
        U = rand.random(s.shape)
        X = []
        
        # generate vectorized SAN paths
        for i in range(U.shape[0]):
            X.append(-np.log(1 - U[i]) * s[i])
        
        # compute max path
        X = np.atleast_2d(X)
        Y = np.max([X[:, 0] + X[:, 3], X[:, 0] + X[:, 2] + X[:, 4], X[:, 1] + X[:, 4]], axis=0)
        sample_paths.append(Y)
    
    return sample_paths

def simulate_bulk(s: np.ndarray, reps: int, path: str=None) -> list:
    '''
    vectorized SAN simulation
    s: (num_scenarios, 5)
    '''
    sample_paths = simulate(s, reps)

    # write data to path    
    df_dict = {f's{i+1}': s[:, i] for i in range(s.shape[1])}
    df_dict['mean'] = np.mean(sample_paths, axis=0)
    df_dict['95_CI'] = 1.96 * np.sqrt(np.var(sample_paths, ddof=1, axis=0)/reps)
    df = pd.DataFrame(df_dict)
    
    if path is not None:
        df.to_csv(path, index=False)

    return df

def parameter_generation(dims: int, lb: float, budget: float, num_parameters: int) -> np.ndarray:
    '''
    input parameter generator for
    SAN simulation
    '''
    params = np.random.uniform(0.5, 1, (num_parameters*5, dims))
    b_const = params[np.sum(params, axis=1) >= budget]
    l_const = np.array([v for v in b_const if np.all(v >= lb)])

    return l_const[:num_parameters]

def data_generation(path: str, dims: int=5, lb: float=5, budget: float=40, num_data_pts: int=50000) -> None:
    '''
    wrapper function for SAN simulation data generation
    '''
    params = parameter_generation(dims, lb, budget, num_data_pts)
    return simulate_bulk(params, 500, path)
    
if __name__ == "__main__":
    df = data_generation('./df_large.csv')

