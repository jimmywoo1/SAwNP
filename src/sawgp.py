import sys
sys.path.append('./np')
sys.path.append('./np/neural_process_models')
sys.path.append('./np/misc')

import json
import numpy as np
import torch 
import GPy
import scipy.optimize as opt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from typing import Callable

from neural_process_models.np import NP_Model
from neural_process_models.anp import ANP_Model
from generate_san import simulate


def smooth(scalars: list, weight: float) -> list:  
    '''
    exponential smoothing
    '''
    last = scalars[0]  
    smoothed = list()

    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)                      
        last = smoothed_val                                
        
    return smoothed

def quad_cost_fn(x: float, d: int=3, noise: bool=False) -> np.ndarray:
    '''
    (1d, 2d or 3d) quadratic objective function
    to be maximized
    '''
    twos = np.ones(d) + 1
    noise = np.random.normal if noise else 0
    return -0.05 * np.linalg.norm(x - twos) ** 2 + 10 + noise


def san_cost_fn(s: list, reps: int=1) -> float:
    '''
    wrapper around SAN simulation
    '''
    return np.mean(simulate(s, reps))


def box_constraint(x: list, bounds: list, d: int) -> list:
    '''
    clip d-dimensional input x between bounds[0] and bounds[1]
    '''
    if d == 1:
        x = max(min(x, np.array([bounds[0][1]])), np.array([bounds[0][0]])) 
    else:
        x = [max(min(v, b[1]), b[0]) for v, b in zip(x, bounds)]

    return x

def sum_constraint(x: list, limit: float) -> np.ndarray:
    '''
    impose sum constraint projection on vector x
    '''
    x = np.array(x)

    if np.sum(x) < limit:
        x = x * limit / np.sum(x) + 1e-6

    return x

def sample_from_circle(x: list, delta: float, bounds: list, d: int, sum_bound: float=None) -> list:
    '''
    sample from uniform circle of radius delta
    around d-dimensional input x with bounds
    '''
    u = np.random.normal(size=d)
    u /= np.linalg.norm(u)
    r = np.random.uniform(0, delta**(1/d))
    sample = x + r*u

    if sum_bound is not None:
        sample = sum_constraint(sample, sum_bound)
    
    return box_constraint(sample, bounds, d)


def expand(x_n: list, c_n: float, d: int, cost_fn: Callable, bounds: list, sum_bound: float=None) -> tuple:
    '''
    obtain all visited locations during single iteration
    of finite differences
    '''
    E = np.identity(d)
    X = [x_n]
    Y = [cost_fn(x_n)]

    for e_i in E:
        x_plus = x_n + c_n * e_i
        x_minus = x_n - c_n * e_i

        if sum_bound is not None:
            x_plus = sum_constraint(x_plus, sum_bound)
            x_minus = sum_constraint(x_minus, sum_bound)
        
        x_plus = box_constraint(x_plus, bounds, d)
        x_minus = box_constraint(x_minus, bounds, d)

        X.append(x_plus)
        X.append(x_minus)
        Y.append(cost_fn(x_plus))
        Y.append(cost_fn(x_minus))

    return X, Y


def finite_difference(x: list, c_n: float, cost_fn: Callable, d: int=1) ->list:
    '''
    finite differences of d-dimensional input x using c_n as step
    '''
    E = np.identity(d)
    grads = []

    for e_i in E:
        grads.append((cost_fn(x + c_n * e_i, d) - cost_fn(x - c_n * e_i, d)) / (2 * c_n))

    return grads


def sa_iteration(x_n: list, a_n: float, c_n: float, 
                 cost_fn: Callable, bounds: list, 
                 d: int, sum_bound: float=None,
                 obj='maximize') -> list:
    '''
    single iteration of stochastic approximation using finite differences
    '''
    grads = finite_difference(x_n, c_n, cost_fn, d)
    grad_update = a_n * np.array(grads) if obj == "maximize" else -a_n * np.array(grads)
    x_next = x_n + grad_update

    if sum_bound is not None:
        x_next = sum_constraint(x_next, sum_bound)

    return box_constraint(x_next, bounds, d)


def stochastic_approximation(x_init: list, num_samples: int, 
                             cost_fn: Callable, bounds: list, 
                             d: int, sum_bound: float=None,
                             obj='maximize') -> tuple:
    '''
    stochastic approximation algorithm, terminates once num_samples is
    satistfied 
    '''
    x_n = x_init
    m = 0
    n = 1
    history = []
    vals = []

    while m < num_samples:
        # sa iteration
        a_n = 0.1 / n
        c_n = n ** (-1/6)
        x_n = sa_iteration(x_n, a_n, c_n, cost_fn, bounds, d, sum_bound, obj)

        # update params
        m += 2
        n += 1
        history.append(cost_fn(x_n, d))
        vals.append(x_n)

    return vals, history

def anp_fn(x: np.ndarray, x_ctxt: np.ndarray, y_ctxt: np.ndarray, d: int) -> float:
    '''
    wrapper function to evaluate target d-dimensional vector x 
    using context x_ctxt and y_ctxt
    '''
    x = torch.tensor(x.reshape(1, -1, d), device=device, dtype=torch.float32)
    x_ctxt = torch.tensor(x_ctxt.reshape(1, -1, d), device=device, dtype=torch.float32)
    y_ctxt = torch.tensor(y_ctxt.reshape(1, -1, 1), device=device, dtype=torch.float32)
    pred, _, _, _, _ = np_model(x_ctxt, y_ctxt, x)

    return pred.squeeze().item()

def objective(x: np.ndarray, x_ctxt: np.ndarray, y_ctxt: np.ndarray, 
              d: int, sign: int) -> float:
    '''
    objective function to be minized for
    maximization/minimization of anp_fn 
    '''
    return sign * anp_fn(x, x_ctxt, y_ctxt, d)

def SAwGP(x_init: list, num_samples: int, cost_fn: Callable, 
          bounds: list, d: int, eta: int, delta: float, 
          gp: bool=True, sum_bound: float=None, obj: str="maximize", 
          cons: dict=None) -> tuple:
    '''
    implementation of SAwGP/SAwANP algorithm
    returns history of input vectors and
    corresponding function evaluations
    as tuple of lists
    '''
    x_n = x_init
    m = 0
    n = 1
    r = 1
    history = {}
    vals = []
    X = []
    Y = []
    sign = -1 if obj == "maximize" else 1

    while m < num_samples:
        print(f'cumulative samples: {m}')
        while n <= r * eta:
            a_n = 1 / n
            c_n = n ** (-1/6)
            
            # step 1 - sa iteration
            x_n = sa_iteration(x_n, a_n, c_n, cost_fn, bounds, d, sum_bound, obj)
            X_batch, Y_batch = expand(x_n, c_n, d, cost_fn, bounds, sum_bound)
            X += X_batch
            Y += Y_batch
            n += 1
            m += 2*d
            vals.append(x_n)
            history[m] = cost_fn(x_n)

        # fit model
        if gp:           
            kern = GPy.kern.RBF(5) * GPy.kern.Bias(5)
            gpr = GPy.models.GPRegression(np.array(X), np.array(Y).reshape(-1, 1), kern)
            gpr.optimize()

            def fmax(x):
                x = np.array(x).reshape(1, -1)
                mean, _ = gpr._raw_predict(x)
                return mean.squeeze()

            if cons is None:
                x_n = opt.minimize(lambda x: sign * fmax(x), x0=x_n, bounds=bounds, method='Powell', options={'maxiter':50000}).x
            else:
                x_n = opt.minimize(lambda x: sign * fmax(x), x0=x_n, bounds=bounds, constraints=cons, options={'maxiter':50000}).x
        else:
            x_ctxt = torch.tensor(X, device=device, dtype=torch.float32).view(1, -1, d)
            y_ctxt = torch.tensor(Y, device=device, dtype=torch.float32).view(1, -1, 1)

            if cons is None:
                x_n = opt.minimize(objective, x0=x_n, args=(x_ctxt, y_ctxt, d, sign), bounds=bounds, method='Powell', options={'maxiter':50000}).x
            else:
                x_n = opt.minimize(objective, x0=x_n, args=(x_ctxt, y_ctxt, d, sign), bounds=bounds, constraints=cons, options={'maxiter':50000}).x

        m += 1
        vals.append(x_n)
        history[m] = cost_fn(x_n)

        # sample from uniform ball
        x_d = sample_from_circle(x_n, delta, bounds, d)
        X.append(x_d)
        Y.append(cost_fn(x_d))
        r += 1

    return vals, history

def evaluate(x_init: list, num_samples: int, cost_fn: Callable, 
             bounds: list, d: int, reps: int=30, eta: int=None,  
             delta: float=None, method: str='SA', sum_bound: float=None,
             obj: str="maximize", cons: dict=None) -> tuple:
    '''
    evaluate opne of SA/SAwGP/SAwANP algorithms for desired
    number of replications
    '''

    res = []
    hist = []
    
    for i in range(reps):
        print(f'replication {i+1}')
        if method == "SA":
            r, h = stochastic_approximation(x_init, num_samples, cost_fn, bounds, d, sum_bound, obj)
        elif method == "SAwGP":
            r, h = SAwGP(x_init, num_samples, cost_fn, bounds, d, eta, delta, sum_bound=sum_bound, obj=obj, cons=cons)
        elif method == 'SAwNP':
            r, h = SAwGP(x_init, num_samples, cost_fn, bounds, d, eta, delta, gp=False, sum_bound=sum_bound, obj=obj, cons=cons)

        res.append(r)
        hist.append(h)
    
    return res, hist

def save_results(res: list, hist: list, num_samples: int, 
                 type: str, reps: int, out_path: str) -> None:
    '''
    saves simulation results to desired path
    '''

    mean_res = np.mean(res, axis=0).squeeze()

    if type == 'SA':
        mean_hist = np.mean(hist, axis=0)
        smooth_vals = [smooth(x, 0.9) for x in hist]
        conf = 1.96 * np.sqrt(np.var(hist, axis=0) / reps)
        xrange = np.arange(0, num_samples, 2).tolist()
    elif type == 'GP' or 'NP':
        hist_vals = np.array([list(x.values()) for x in hist])
        mean_hist = np.mean(hist_vals, axis=0)
        smooth_vals = [smooth(x, 0.9) for x in hist_vals]
        conf = 1.96 * np.sqrt(np.var(hist_vals, axis=0) / reps)
        xrange = list(hist[0].keys())
    
    # smoothed mean and 95% CI
    conf_smooth = 1.96 * np.sqrt(np.var(smooth_vals, axis=0) / reps)
    mean_smooth = np.mean(smooth_vals, axis=0)

    out = {'x': xrange, 
           'res': mean_res.tolist(), 
           'hist': mean_hist.tolist(), 
           'conf': conf.tolist(),
           'hist_smooth': mean_smooth.tolist(),
           'conf_smooth': conf_smooth.tolist()}
    
    with open(out_path, 'w') as file:
        json.dump(out, file)

if __name__ == "__main__":
    device = torch.device('cuda')

    # simple cost_fn params
    eta = 20
    delta = 1
    reps = 30

    # 1d evaluation
    x_init = -8
    bounds = [(-10, 10)]
    d = 1
    num_samples = 1000
    results_dir = "./results/quad_fn/"
    model_path = "./anp_1d.pth"
    latent_dim = 128
    mlp_size = [128, 128, 128, 128]

    # ANP
    np_model = ANP_Model(x_dim=d,
                        y_dim=1,
                        mlp_hidden_size_list=mlp_size,
                        latent_dim=latent_dim,
                        use_rnn=False,
                        use_self_attention=False,
                        use_deter_path=True).to(device)
    np_model.load_state_dict(torch.load(model_path))

    np_res_1d, np_hist_1d = evaluate(x_init, num_samples, quad_cost_fn, bounds, d, reps=30, eta=eta, delta=delta, method='SAwNP')
    save_results(np_res_1d, np_hist_1d, num_samples, 'NP', reps, f'{results_dir}1d/np.json')

    # SA - 1D
    sa_res_1d, sa_hist_1d = evaluate(x_init, num_samples, quad_cost_fn, bounds, d, reps=reps, method='SA')
    save_results(sa_res_1d, sa_hist_1d, num_samples, 'SA', reps, f'{results_dir}1d/sa.json')

    # # GP - 1D    
    gp_res_1d, gp_hist_1d = evaluate(x_init, num_samples, quad_cost_fn, bounds, d, reps=5, eta=eta, delta=delta, method='SAwGP')
    save_results(gp_res_1d, gp_hist_1d, num_samples, 'GP', reps, f'{results_dir}1d/gp.json')

    # 2d evaluation
    x_init = [-8, -8]
    bounds = [(-10, 10), (-10, 10)]
    d = 2
    num_samples = 2000
    results_dir = "./results/quad_fn/"
    model_path = "./anp_2d.pth"
    latent_dim = 128
    mlp_size = [128, 128, 128, 128]

    # ANP
    np_model = ANP_Model(x_dim=d,
                        y_dim=1,
                        mlp_hidden_size_list=mlp_size,
                        latent_dim=latent_dim,
                        use_rnn=False,
                        use_self_attention=False,
                        use_deter_path=True).to(device)
    np_model.load_state_dict(torch.load(model_path))

    np_res_2d, np_hist_2d = evaluate(x_init, num_samples, quad_cost_fn, bounds, d, reps=30, eta=eta, delta=delta, method='SAwNP')
    save_results(np_res_2d, np_hist_2d, num_samples, 'NP', reps, f'{results_dir}2d/np.json')


    sa_res_2d, sa_hist_2d = evaluate(x_init, num_samples, quad_cost_fn, bounds, d, reps=reps, method='SA')
    save_results(sa_res_2d, sa_hist_2d, num_samples, 'SA', reps, f'{results_dir}2d/sa.json')

    gp_res_2d, gp_hist_2d = evaluate(x_init, num_samples, quad_cost_fn, bounds, d, reps=5, eta=eta, delta=delta, method='SAwGP')
    save_results(gp_res_2d, gp_hist_2d, num_samples, 'GP', reps, f'{results_dir}2d/gp.json')

    # 3d evaluation
    x_init = [-8, -8, -8]
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    d = 3
    num_samples = 3000
    results_dir = "./results/quad_fn/"
    model_path = "./anp_3d.pth"
    latent_dim = 256
    mlp_size = [256, 256, 256, 256]

    # ANP
    np_model = ANP_Model(x_dim=d,
                        y_dim=1,
                        mlp_hidden_size_list=mlp_size,
                        latent_dim=latent_dim,
                        use_rnn=False,
                        use_self_attention=False,
                        use_deter_path=True).to(device)
    np_model.load_state_dict(torch.load(model_path))

    np_res_3d, np_hist_3d = evaluate(x_init, num_samples, quad_cost_fn, bounds, d, reps=30, eta=eta, delta=delta, method='SAwNP')
    save_results(np_res_3d, np_hist_3d, num_samples, 'NP', reps, f'{results_dir}3d/np.json')

    sa_res_3d, sa_hist_3d = evaluate(x_init, num_samples, quad_cost_fn, bounds, d, reps=reps, method='SA')
    save_results(sa_res_3d, sa_hist_3d, num_samples, 'SA', reps, f'{results_dir}3d/sa.json')

    gp_res_3d, gp_hist_3d = evaluate(x_init, num_samples, quad_cost_fn, bounds, d, reps=30, eta=eta, delta=delta, method='SAwGP')
    save_results(gp_res_3d, gp_hist_3d, num_samples, 'GP', reps, f'{results_dir}3d/gp.json')

    # SAN evaluation
    eta = 20
    delta = 1
    x_init = [1.0, 1.0, 1.0, 1.0, 1.0]
    bounds = [(0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.5, 1.0)]
    budget_bound = 4
    d = 5
    num_samples = 1000
    reps = 30
    results_dir = "./results/san/"
    model_path = "./anp_san.pth"
    latent_dim = 256
    mlp_size = [256, 256, 256, 256]

    sa_res_san, sa_hist_san = evaluate(x_init, num_samples, san_cost_fn, bounds, d, reps, eta, delta, sum_bound=budget_bound, obj="minimize")
    save_results(sa_res_san, sa_hist_san, num_samples, 'SA', reps, f'{results_dir}sa_1.json')

    # linear constraints for SAN
    A = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    B = np.array([4])
    cons = [{'type': 'ineq', 'fun': lambda x: A @ x - B}]
    
    gp_res_san, gp_hist_san = evaluate(x_init, num_samples, san_cost_fn, bounds, d, reps, eta, delta, method='SAwGP', sum_bound=budget_bound, obj="minimize", cons=cons) 
    save_results(gp_res_san, gp_hist_san, num_samples, 'GP', reps, f'{results_dir}gp_1.json')
    
    np_res_san, np_hist_san = evaluate(x_init, num_samples, san_cost_fn, bounds, d, reps, eta, delta, method='SAwNP', sum_bound=budget_bound, obj="minimize", cons=cons) 
    save_results(np_res_san, np_hist_san, num_samples, 'NP', reps, f'{results_dir}np_1.json')

    