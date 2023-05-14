import sys
sys.path.append('./np')
sys.path.append('./np/neural_process_models')
sys.path.append('./np/misc')

import numpy as np
import torch
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from neural_process_models.anp import ANP_Model
from generate_san import simulate_bulk, parameter_generation


def sin_cost_fn(x: float, noise: bool=False) -> np.ndarray:
    '''
    sinusoidal objective function
    to be maximized
    '''
    if np.abs(x) <= 0.33:
        return 1.35 * np.cos(12*np.pi*x)
    if np.abs(x) <= 0.66:
        return 1.35
    return 1.35/2 * np.cos(24*np.pi*x) + 1.35/2


def cost_fn(x: float, d: int=3, noise: bool=False) -> np.ndarray:
    '''
    (1d, 2d or 3d) quadratic objective function
    to be maximized
    '''
    twos = np.ones(d) + 1
    noise = np.random.normal if noise else 0
    return -0.05 * np.linalg.norm(x - twos) ** 2 + 10 + noise

if __name__ == "__main__":
    dims = 5
    lb = 0.5
    budget = 4

    # hyperparameters
    epochs = 1000
    latent_dim = 256
    mlp_size = [256, 256, 256, 256]
    batch_size = 100
    min_ctxt_size = batch_size * 0.2
    max_ctxt_size = batch_size 
    device = torch.device("cuda")

    # ANP
    np_model = ANP_Model(x_dim=dims,
                        y_dim=1,
                        mlp_hidden_size_list=mlp_size,
                        latent_dim=latent_dim,
                        use_rnn=False,
                        use_self_attention=False,
                        use_deter_path=True).to(device)

    optim = torch.optim.Adam(np_model.parameters(), lr=1e-4)
    scheduler = StepLR(optim, step_size=500, gamma=0.1)

    x_cols = ['s1', 's2', 's3', 's4', 's5']
    y_col = ['mean']

    for epoch in range(epochs):
        np_model.train()
        # plt.clf()

        # simulate training data
        params = parameter_generation(dims, lb, budget, batch_size * 2)
        df = simulate_bulk(params, 100)

        ctxt_size = np.random.randint(min_ctxt_size, batch_size)
        x_train = df[x_cols].to_numpy()[:batch_size]
        x_test = df[x_cols].to_numpy()[batch_size:]
        y_train = df[y_col].to_numpy()[:batch_size]
        y_test = df[y_col].to_numpy()[batch_size:] 

        # subset and convert to tensor
        x_ctt = torch.tensor(x_train[:ctxt_size].reshape(1, -1, 5), 
                             device=device, 
                             dtype=torch.float32)
        y_ctt = torch.tensor(y_train[:ctxt_size].reshape(1, -1, 1),
                             device=device,
                             dtype=torch.float32)
        x_tgt = torch.tensor(x_train[ctxt_size:].reshape(1, -1, 5),
                             device=device,
                             dtype=torch.float32)
        y_tgt = torch.tensor(y_train[ctxt_size:].reshape(1, -1, 1),
                             device=device,
                             dtype=torch.float32)

        # forward pass
        mu, sigma, log_p, kl, loss = np_model(x_ctt, y_ctt, x_tgt, y_tgt)
        loss_val = loss.item()

        # backprop
        optim.zero_grad()
        loss.backward()
        optim.step()

        # evaluate on test data
        np_model.eval()
        x_ctt_eval = torch.tensor(x_train.reshape(1, -1, 5), device=device, dtype=torch.float32)
        y_ctt_eval = torch.tensor(y_train.reshape(1, -1, 1), device=device, dtype=torch.float32)
        x_tgt_eval = torch.tensor(x_test.reshape(1, -1, 5), device=device, dtype=torch.float32)
        np_pred, sigma, logp, kl, loss = np_model(x_ctt_eval, y_ctt_eval, x_tgt_eval)
        np_pred = np_pred.cpu().detach().squeeze().numpy()
        np_rmse = mean_squared_error(y_test, np_pred, squared=False)

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}: Loss: {loss_val} RMSE: {np_rmse}')
        scheduler.step()

    torch.save(np_model.state_dict(), "./anp_san.pth")