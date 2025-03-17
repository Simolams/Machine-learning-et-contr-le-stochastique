import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def pde_solver_regularization(r,sigma,K,T,lambda_,Nx,Nt,x_min,x_max,max_iter):

    dx = (x_max - x_min) / Nx
    x = np.linspace(x_min, x_max, Nx+1)

    # Time discretization
    dt = T / Nt
    # Construct L_x matrix coefficients
    a = 0.5 * sigma**2 / dx**2 - (r - 0.5 * sigma**2) / (2 * dx)  # Lower diagonal
    b = -sigma**2 / dx**2 - r  # Main diagonal
    c = 0.5 * sigma**2 / dx**2 + (r - 0.5 * sigma**2) / (2 * dx)  # Upper diagonal

    # Banded matrix for efficient solving
    A_banded = np.zeros((3, Nx-1))
    A_banded[0, 1:] = c  # Upper diagonal
    A_banded[1, :] = 1/dt - b  # Main diagonal (implicit time step)
    A_banded[2, :-1] = a  # Lower diagonal

    # Initial condition (final condition for backward in time)
    eps = 1e-6  # Small value to prevent numerical issues
    h = np.maximum(K - np.exp(x), eps)  # Put option-like function
    nu = h.copy()

    # Newton iteration parameters
    tol = 1e-6
    max_iter = 50
    damping = 0.5  # Damping factor for Newton updates


    # Time stepping (backward in time)
    for n in reversed(range(Nt)):
        u_old = nu.copy()
        for k in range(max_iter):  # Newton iteration
            u_old = np.maximum(u_old, h - lambda_)  # Prevent instability
            
            # Compute f(u) (residual)
            f = (u_old[1:-1] - nu[1:-1]) / dt \
                - (a * u_old[:-2] + b * u_old[1:-1] + c * u_old[2:]) \
                + lambda_ * np.exp(-(u_old[1:-1] - h[1:-1]) / lambda_)

            # Compute the Jacobian J(u)
            d_main = 1 / dt - b - np.exp(-(u_old[1:-1] - h[1:-1]) / lambda_)  # Main diagonal
            d_lower = -a * np.ones(Nx-2)  # Lower diagonal
            d_upper = -c * np.ones(Nx-2)  # Upper diagonal
            
            # Construct sparse tridiagonal Jacobian matrix
            J = diags([d_lower, d_main, d_upper], offsets=[-1, 0, 1], format='csc')

            # Solve for du: J * du = -f
            du = spsolve(J, -f)

            # Apply damping to ensure stability
            u_old[1:-1] += damping * du

            # Check for NaN or convergence
            if np.any(np.isnan(u_old)):
                print("NaN encountered! Stopping.")
                break
            if np.linalg.norm(du) < tol:
                break

        nu = u_old.copy()
    

    return nu




def penalty_pde_solver(r,sigma,K,T,lambda_,Nx,Nt,x_min,x_max) : 

    x = np.linspace(x_min, x_max, Nx+1)
    dx = (x_max - x_min) / Nx

    # Time discretization
    dt = T / Nt
    t = np.linspace(0, T, Nt+1)

    # Construct L_x matrix coefficients
    a = 0.5 * sigma**2 / dx**2 - (r - 0.5 * sigma**2) / (2 * dx)  # Lower diagonal
    b = -sigma**2 / dx**2 - r  # Main diagonal
    c = 0.5 * sigma**2 / dx**2 + (r - 0.5 * sigma**2) / (2 * dx)  # Upper diagonal

    # Initial condition (final condition for backward in time)
    eps = 1e-6  # Small value to prevent numerical issues
    h = np.maximum(K - np.exp(x), eps)  # Payoff function (like American put)
    nu = h.copy()  # Initialize with the obstacle condition

    # Newton iteration parameters
    tol = 1e-6
    max_iter = 50
    damping = 0.5  # Damping factor for Newton updates
  
    
    # Time stepping (backward in time)
    for n in reversed(range(Nt)):
        u_old = nu.copy()
        for k in range(max_iter):  # Newton iteration
            penalty = np.maximum(0, h[1:-1] - u_old[1:-1])  # Penalty function
            
            # Compute f(u) (residual)
            f = (u_old[1:-1] - nu[1:-1]) / dt \
                - (a * u_old[:-2] + b * u_old[1:-1] + c * u_old[2:]) \
                + lambda_ * penalty

            # Compute the Jacobian J(u)
            d_main = 1 / dt - b - lambda_ * (penalty > 0)  # Main diagonal
            d_lower = -a * np.ones(Nx-2)  # Lower diagonal
            d_upper = -c * np.ones(Nx-2)  # Upper diagonal
            
            # Construct sparse tridiagonal Jacobian matrix
            J = diags([d_lower, d_main, d_upper], offsets=[-1, 0, 1], format='csc')

            # Solve for du: J * du = -f
            du = spsolve(J, -f)

            # Apply damping to ensure stability
            u_old[1:-1] += damping * du

            # Enforce obstacle condition manually
            #u_old = np.maximum(u_old, h)

            # Check for NaN or convergence
            if np.any(np.isnan(u_old)):
                print("NaN encountered! Stopping.")
                break
            if np.linalg.norm(du) < tol:
                break

        nu = u_old.copy()

    
    return nu




import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
def policy_iteration_solver(r,sigma,K,T,lambda_,Nx,Nt,xmin,xmax,max_iter,tol,clip = True) : 


        u_list = []
        dx = (xmax - xmin) / (Nx - 1)
        dt = T / Nt
        K = 1

        # Space grid
        x = np.linspace(xmin, xmax, Nx)


        # Time discretization
        dt = T / Nt  

        # Construct Lx operator
        def Lx_operator(Nx, dx, sigma, r):
                alpha = 0.5 * sigma**2 / dx**2
                beta = (r - 0.5 * sigma**2) / (2 * dx)
                
                diagonal = (-2 * alpha + r) * np.ones(Nx)
                lower = (alpha - beta) * np.ones(Nx - 1)
                upper = (alpha + beta) * np.ones(Nx - 1)
                
                Lx = diags([lower, diagonal, upper], offsets=[-1, 0, 1]).tocsc()
                return Lx

        Lx = Lx_operator(Nx, dx, sigma, r)

        # Terminal condition h(x)
        h = np.maximum(K - np.exp(x), 0)  # Put option-like function
        # Initialize policy π using equation (3.6)
        U = np.zeros((Nt,Nx))
        U[-1,:] = h
        pi = np.ones_like(U)

        for iteration in range(max_iter):

                print(f"Policy Iteration {iteration+1}")
        
                # Initialize the value function with the terminal condition
                U_new = np.zeros_like(U)
                U_new[-1,:] = h

                # Time-stepping (backward Euler)
                for n in range(Nt-1,0,-1):

                        # Compute Hamiltonian term H(x, π, u)
                        H = (h) * pi[n-1,:] + lambda_ * (pi[n-1,:] - pi[n-1,:] * np.log(pi[n-1,:]))

                        # Solve the system (I - dt * Lx - dt * diag(pi)) u = u_old - dt * H
                        J = (np.eye(Nx) - dt * Lx.toarray() + dt * np.diag(pi[n-1,:]))
                        rhs = U[n,:] + dt * H
                        U_new[n-1,:] = spsolve(J, rhs)
                
                #new_pi = np.exp((-U_new + h) / lambda_).copy()
                if clip : 
                     
                    new_pi = np.exp(np.clip((-U_new + h) / lambda_, -10, 10)) # Clipping for stability
                
                else : 
                     new_pi = np.exp((-U_new+h)/lambda_)
                     
        
                # Convergence check
                if np.max(np.abs(new_pi - pi)) < tol:
                        print("Policy Converged!")
                        break

                # Update policy
                pi = new_pi.copy()
                U = U_new.copy()

                u_list.append(U[0,:])
        
        return x,u_list
        
    

    
