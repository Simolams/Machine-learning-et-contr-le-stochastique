
import torch
from torch import nn
from torch import optim
import numpy as np
from numpy.random import default_rng
rng = default_rng()

################### paramètres ############################

S0 = 40  # Prix initial
K = 40  # Prix d'exercice
T = 1    # Maturité en années
r = 0.06 # Taux d'intérêt sans risque
sigma = 0.4 # Volatilité
N = 10  # Nombre de pas
dt = T / N  # Intervalle de temps
M = int(2**10) # Nombre de chemins MC
lambda_ = 0.05 # paramètre de régularisation
creterio  = nn.MSELoss()  ## LOSS FUNCTION

g = lambda x : np.maximum(K-x,0) # payoff function

R = lambda x : x-x*np.log(x) # Regularization function
optimal_price = 5.31


def generate_black_scholes_paths(size_path, size_sample): 
    dt = T/size_path
    gaussien_inc = np.sqrt(dt)*rng.standard_normal(size=(size_path, size_sample))
    sample = np.zeros(shape=(size_path+1, size_sample))
    sample[0] = S0
    for n in range(1, size_path+1):
        sample[n] = sample[n-1] * np.exp((r - 0.5 * sigma**2)*dt + sigma*gaussien_inc[n-1])
    return sample


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def Phi_NN(K,N,lr = 0.0001) :

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(1, 21),
                nn.ReLU(),
                nn.Linear(21, 21),
                nn.ReLU(),
                nn.Linear(21, 21),
                nn.ReLU(),
                nn.Linear(21, 21),
                nn.ReLU(),
                nn.Linear(21, 21),
                nn.ReLU(),
                nn.Linear(21, 1),
                nn.LeakyReLU(negative_slope=0.01)
            )

        def forward(self, x):
            out = self.linear_relu_stack(x)
            payoff = torch.max(K - x, torch.tensor(0.0))
            #payoff = (K-x)*torch.sigmoid(K - x)
            return out  +payoff
        

    # Define the optimizers and the networks : 

    Phi_functions = [NeuralNetwork().to(device) for n in range(1,N)]

    optimizers = [optim.Adam(Phi_functions[n-1].parameters(), lr) for n in range(1,N)]

    return Phi_functions,optimizers


######################## 

def payoff_phi(n, x): 
    return np.exp(-r*n*T/N) * np.maximum(K-x, 0)


def calculate_price(Phi_functions,data_train):

    
    payoffs = np.empty_like(data_train)
    for n in range(0, N):
        payoffs[:,n] = payoff_phi(n, data_train[:,n])
    
    payoff_opt = payoffs[:,-1].copy() # Payoffs optimaux à l'instant t = N

    for n in range(N-1,0,-1):

        xx = torch.tensor(data_train[:, n], dtype=torch.float32, device=device).reshape(-1, 1)
        with torch.no_grad():
                continuation_function = Phi_functions[n-1](xx).cpu().numpy().flatten()

        stop_at_n = payoffs[:,n] >= continuation_function
        payoff_opt[stop_at_n] = payoffs[stop_at_n,n].copy() # Payoffs optimaux à l'instant t = n


    payoff_opt.mean()
    

    return payoff_opt.mean(),payoff_opt

#############################


def train_evaluate(n_iteration,lambda_,Phi_functions,optimizers,data_train):


    Losses = []
    pi_values = np.ones_like(data_train)
    P = []

    for n in range(n_iteration): 

        S_samples = generate_black_scholes_paths(N-1,M).T
        #S_samples = data_train.copy()
        V_values = np.zeros_like(S_samples)
        V_values[:,-1] = g(S_samples[:,-1])


        print(f'{n+1} Iteration ...')
        

        total_loss = 0
        for l in range(N-2,-1,-1):
            
        
            #Calculate the TD-error :
            
            ## construct target vector
            X = S_samples[:,l]
            Y = g(S_samples[:,l])*pi_values[:,l]*dt + lambda_*R(pi_values[:,l])*dt+np.exp(-r*dt)*V_values[:,l+1]*(1-pi_values[:,l]*dt)

            xx = torch.tensor(X, dtype=torch.float32, device=device).reshape(-1, 1)
            yy = torch.tensor(Y, dtype=torch.float32, device=device)

            #print(f"Start training V_{l} ...")
            # entrainement "one step Gadient descent" :
            prediction = Phi_functions[l](xx)
            loss = creterio(prediction, yy.reshape(-1, 1))
            optimizers[l].zero_grad()
            loss.backward()
            optimizers[l].step()
            #print(f"model trained ! ...")
            total_loss+=loss

            # Calculate the new V using the updated parameters 
            with torch.no_grad():
                new_y =Phi_functions[l](xx).cpu().numpy().flatten()
            
            V_values[:,l] = new_y
        

        pi_values = np.ones_like(S_samples)
        pi_values[:,:-1] = np.exp(np.clip(-(V_values[:,:-1]-g(S_samples[:,:-1])) / lambda_, -10, 10))
        
        if n%1 == 0 :
            data_test = generate_black_scholes_paths(N-1,M).T
            price,_ = calculate_price(Phi_functions,data_test)
            relative_error = np.abs(optimal_price-price)/price

        
        P.append(relative_error)
        print(f"Epochs {n+1} finished !")
        print(f"TOTAL LOSS : {total_loss/N}")
        Losses.append((total_loss/N).detach().cpu().numpy())


    return P
