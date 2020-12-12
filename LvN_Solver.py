import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import pandas as pd
from scipy import linalg,integrate
from numpy import random
from functools import partial
from multiprocessing import Pool

def Off_diagonal_disorder(N):
    E =1
    diag = np.empty(N)
    diag.fill(E)
    H = np.diag(diag,k=0)
    V =random.random(N)
    H[0][1] =  1* V[0]
    H[N-1][N-2] = 1*V[N-1]
    for i in range(1,N-1):
        H[i][i+1] = 1*V[i]
        H[i][i-1] = 1*V[i]    
    return(H)
            
def Site_disorder(N):
    diag =random.random(N)
    H = np.diag(diag,k=0)
    V = 1
    H[0][1] =  V
    H[N-1][N-2] = V
    for i in range(1,N-1):
        H[i][i+1] = V
        H[i][i-1] = V 
    # set periodic boundary conditions
    H[0][N-1] = V
    H[N-1][0] = V                
    return(H)
    

def Lindbladian_super_operator(N):
    d = np.empty(N**2)
    d.fill(1)
    iter_list = [i*(N+1) for i in range(N)]
    for i in iter_list:
        d[i] = 0    
    D = np.diag(d,k=0)    
    return(D)        


def rho0(N,i):
    psi0_bra = np.zeros(N)
    psi0_bra[i-1] = 1
    psi0_ket = np.vstack(psi0_bra)
    rho0_mat = psi0_ket*psi0_bra
    rho0_vec = rho0_mat.reshape((N**2,))
    return(rho0_vec)

def rho0_delocalised(N):
    psi0_bra =  np.empty(N)
    psi0_bra.fill(1.0)
    psi0_bra = 1/N * psi0_bra
    psi0_ket = np.vstack(psi0_bra)
    rho0_mat = psi0_ket*psi0_bra
    rho0_vec = rho0_mat.reshape((N**2,))
    return(rho0_vec)       
 

def Liouvillian(H,N):
    I = np.eye(N)
    #print(H)
    L = -1j*(np.kron(H,I)-np.kron(I,H))
    return(L)

def LvN_solver_norm_preserving(L,P0,T,tSteps):
    func = lambda t,y : np.dot(L,y)
    integrator = integrate.ode(func)
    integrator.set_integrator('zvode',method='adams',with_jacobian=True)
    integrator.set_initial_value(P0,0.0)
    rho_t = []
    dt = T/tSteps
    while integrator.successful() and integrator.t < T:
        rho_ti = integrator.integrate(integrator.t+dt)
        norm  = linalg.norm(rho_ti)
        rho_ti = 1/norm * rho_ti
        rho_t.append(rho_ti)
    return(np.array(rho_t))

def LvN_solver_off_diag(L,P0,T,tSteps):
    func = lambda t,y : np.dot(L,y)
    integrator = integrate.ode(func)
    integrator.set_integrator('zvode',method='BFDS',with_jacobian=True)
    integrator.set_initial_value(P0,0.0)
    rho_t = []
    dt = T/tSteps
    while integrator.successful() and integrator.t < T:
        rho_ti = integrator.integrate(integrator.t+dt)
        norm  = linalg.norm(rho_ti)
        rho_ti = 1/norm * rho_ti
        rho_t.append(rho_ti)
    return(np.array(rho_t))


def LvN_solver(L,P0,T,tSteps):
    func = lambda t,y : np.dot(L,y)
    integrator = integrate.ode(func)
    integrator.set_integrator('zvode',method='adams',with_jacobian=True)
    integrator.set_initial_value(P0,0.0)
    print(P0)
    rho_t = []
    dt = T/tSteps
    while integrator.successful() and integrator.t < T:
        rho_ti = integrator.integrate(integrator.t+dt)
        #norm  = linalg.norm(rho_ti)
        #rho_ti = 1/norm * rho_ti
        rho_t.append(rho_ti)
    return(np.array(rho_t))


def Disorder_average(N,K,params,Method,env):
    i_site,T,tSteps = params
    P0 = rho0(N,i_site)
    def Loop():
        if env =='pure_dephasing':
            rate = 0.1
            if Method =='site_disorder':
                L_dephasing = Lindbladian_super_operator(N)
                rho_t = np.empty((tSteps,N**2),dtype='complex128')
                for i in range(K-1):
                    H  = Site_disorder(N)
                    L = Liouvillian(H,N) - rate*L_dephasing
                    rho_t += LvN_solver_norm_preserving(L,P0,T,tSteps)
                rho_t = rho_t/K
                return(rho_t)

            elif Method =='off_diagonal_disorder':
                L_dephasing = Lindbladian_super_operator(N)
                rho_t = np.empty((tSteps,N**2),dtype='complex128')
                for i in range(K-1):
                    H  = Off_diagonal_disorder(N)
                    L = Liouvillian(H,N) - rate*L_dephasing
                    rho_t += LvN_solver(L,P0,T,tSteps)
                rho_t = rho_t/K
                norm  = linalg.norm(rho_t)
                rho_t = 1/norm * rho_t
                return(rho_t)
            else:
                print('Method Error: Please check your method *arg')    
        
        elif env =='isolated':
            if Method =='site_disorder':
                rho_t = np.empty((tSteps,N**2),dtype='complex128')
                for i in range(K-1):
                    H  = Site_disorder(N)
                    L = Liouvillian(H,N)
                    rho_t += LvN_solver(L,P0,T,tSteps)
                rho_t = rho_t/K
                return(rho_t)

            elif Method =='off_diagonal_disorder':
                rho_t = np.empty((tSteps,N**2),dtype='complex128')
                for i in range(K-1):
                    H  = Off_diagonal_disorder(N)
                    L = Liouvillian(H,N)
                    rho_t += LvN_solver(L,P0,T,tSteps)
                rho_t = rho_t/K
                return(rho_t)
            else:
                print('Method Error: Please check your method *arg')    
        else:
            print('env Error: Please check your env *arg')        

    rho_t = Loop()             
    return(rho_t)

class Disorder_average_parallelisation(object):
    """docstring for Disorder_average_parallelisation"""
    def __init__(self, N, params, Method, env ,rate=0.0):
        super(Disorder_average_parallelisation, self).__init__()
        self.N = N
        self.i_site,self.T,self.tSteps = params
        #self.P0 = rho0(self.N,self.i_site)
        self.P0 = rho0_delocalised(self.N)
        self.env = env
        print(self.env)
        self.Method = Method
        if self.env == 'pure_dephasing':
            print(True)
            self.rate = rate 
            self.L_dephasing = Lindbladian_super_operator(N)

    def run_site_disorder_dephasing(self,i):
        H  = Site_disorder(self.N)
        L = Liouvillian(H,self.N) - self.rate*self.L_dephasing
        rho_t = LvN_solver(L,self.P0,self.T,self.tSteps)
        #print('cycle')
        return(rho_t)
    
    def run_off_diagonal_disorder_dephasing(self,i):
        H  = Off_diagonal_disorder(self.N)
        L = Liouvillian(H,self.N) - self.rate*self.L_dephasing
        rho_t = LvN_solver_norm_preserving(L,self.P0,self.T,self.tSteps)
        #norm  = linalg.norm(rho_t)
        #rho_t = 1/norm * rho_t
        #print('cycle')
        return(rho_t)  
    
    def run_site_disorder_isolated(self,i):
        H  = Site_disorder(self.N)
        L = Liouvillian(H,self.N)
        rho_t = LvN_solver_norm_preserving(L,self.P0,self.T,self.tSteps)
        #print('cycle')
        return(rho_t)       
    
    def run_off_diagonal_disorder_isolated(self,i):
        H  = Off_diagonal_disorder(self.N)
        L = Liouvillian(H,self.N)
        rho_t = LvN_solver_norm_preserving(L,self.P0,self.T,self.tSteps)
        #print('cycle')
        return(rho_t)
    
    def run_decision_tree(self,K):
        pool = Pool(6)
        if self.env =='pure_dephasing':
            if self.Method =='site_disorder':
                rho = np.array(pool.map(partial(self.run_site_disorder_dephasing),range(K))).sum(axis=0)
                rho = rho/K
                return(rho)
    
            elif self.Method =='off_diagonal_disorder':
                rho = np.array(pool.map(partial(self.run_off_diagonal_disorder_dephasing),range(K))).sum(axis=0)
                rho = rho/K
                return(rho)
    
            else:
                print('Method Error: Please check your method *arg')    
            
        elif self.env =='isolated':
            if Method =='site_disorder':
                rho = np.array(pool.map(partial(self.run_site_disorder_isolated),range(K))).sum(axis=0)
                rho =  rho/K
                return(rho)
    
            elif self.Method =='off_diagonal_disorder': 
                rho = np.array(pool.map(partial(self.run_off_diagonal_disorder_isolated),range(K))).sum(axis=0)
                rho = rho/K
                return(rho)
                
            else:
                print('Method Error: Please check your method *arg')    
        else:
            print('env Error: Please check your env *arg')        
             


def Time_average(rho_list,T,file_name,trial):
    print(rho_list)
    tSteps,N2 = rho_list.shape 
    N = int(np.sqrt(N2))
    n=0
    time = np.linspace(0,T,tSteps)
    tavgs = []
    fig = plt.figure()
    save_data = []
    for i in range(N):
        Populations = np.abs([p[n] for p in rho_list ])
        save_data.append(Populations)
        tavgs.append(np.trapz(Populations,time)/T)
        plt.plot(time,Populations,label = i )
        n+= (N+1)
    info_str = ('N :'+str(N)+'\n'+
        'T = '+str(T)+'\n'+
        'dt = '+str(T/(tSteps-1)))
    plt.text(0,1.0 ,info_str,fontsize=12)
    plt.xlabel('time (a.u.)')
    plt.ylabel(r'$\rho_{nn}$')
    #plt.ylim(0,1)
    plt.legend()
    fig.savefig('population_data_'+str(N)+'_trial_'+str(trial))

    save_data = np.array(save_data).T
    df = pd.DataFrame(save_data)
    df.to_csv(file_name)
    #np.savetxt(file_name,save_data)
    #plt.show()    
    return(tavgs)
    
def main(N,K,params,Method,env,file_name,trail):
    #Method = 'site_disorder'
    #env = 'pure_dephasing'
    dt = params[1]/params[2]
    time = np.arange(0,params[1],dt)
    pops = Disorder_average(N,K,params,Method,env)
    tavg = Time_average(pops,params[1],file_name,trial)
    return(tavg)

def main_parallelised(N,K,params,Method,env,file_name,trial):
    dt = params[1]/params[2]
    time = np.arange(0,params[1],dt)
    rate = 0.01
    Disorder_class = Disorder_average_parallelisation(N,params,Method,env,rate)
    pops = Disorder_class.run_decision_tree(K)
    tavg = Time_average(pops,params[1],file_name,trial)
    return(tavg)
    
if __name__ == '__main__':
    N = int(sys.argv[2])
    K = int(sys.argv[4])
    T = int(sys.argv[6])
    #population_data = dict()
    tSteps = T*10
    Method = sys.argv[10]
    env = sys.argv[12]
    rate = sys.argv[14]
    file_name = sys.argv[16]
    trial = sys.argv[18] 
    print(Method,env)
    i_site = int(N/2 +1)

    params = (i_site,T,tSteps)
    print('works!',N,params)
    #tavg_data = main(N,K,params,Method, env,file_name,trial)
    tavg_data = main_parallelised(N,K,params,Method,env,file_name,trial)
    np.savetxt(file_name+'_population_data',tavg_data)
    #print('Total Probability for consistency check, (this should be 1.0):',sum(tavg_data))
    fig = plt.figure(figsize=(16,12))
    plt.xticks(range(1,N+1))
    plt.plot(range(1,N+1),tavg_data,':',marker='o')
    plt.xlabel('n')
    plt.ylabel(r'$\langle \rho_{nn} \rangle_{\Theta,\tau} $')
    info_str2 = ('N :'+str(N)+'\n'+
        'Ensemble ; '+str(K)+'\n'+
        'T = '+str(T)+'\n'+
        ' dt = '+str(tSteps)+'\n'+
        'Method :'+Method +'\n'+
        'Enivornment: '+env)
    plt.text(0,1.0,info_str2,fontsize=12)
    #plt.ylim(0,1)

    plt.title('Disorder Averaged Populations')
    fig.savefig(file_name+'.png')
    #plt.show()
    #print(tavg_data)

    
