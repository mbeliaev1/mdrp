import numpy as np
import cyipopt

class mdrp_prob:

    def __init__(self, prob_setup):
        '''
        Initializing the mdrp non-linear program to be solved.
        All parameters are np arrays
        All units and rates are in hours

        NOTE:
        Main optimization methods use 'x' in vector form
        Helpers will output items in matrix form for indexing convenience

        Parameters required:
            s_time     - I(row) x J(col), service time of ea. (order,mode) pair
            t_time     - I(row) x J(col), travel time of ea. (order,mode) pair
            cost       - I(row) x J(col), cost of ea. (order,mode) pair
            beta       - I(row) x J(col), portion of couriers available of ea. (order,mode) pair
            N          - J(row), total available couriers for each mode
            k          - J(row), unit of time for uniform distribution of couriers
            mu         - J(row), average rate of order completions by coruier mode
            max_rho    - vector, maximum server utilization
            demand     - scalar, between 0 and 1, scales the demand by a function of max demand
        '''

        self.s_time  = prob_setup['s_time']
        self.t_time  = prob_setup['t_time']
        self.cost    = prob_setup['cost']
        self.beta    = prob_setup['beta']
        self.N_veh   = prob_setup['N']
        self.k       = prob_setup['k']
        self.mu      = prob_setup['mu']
        self.max_rho = prob_setup['max_rho']

        self.n = self.t_time.shape[0]
        self.m = self.t_time.shape[1]
        
        self.rate = prob_setup['rate']
        if prob_setup['scale']:
            self.demand = self.get_max_demand()*prob_setup['demand']*self.rate
        else:
            self.demand = prob_setup['demand']*self.rate
        self.tot_demand = self.demand.sum()

        self.alpha_0 = prob_setup['alpha_0']
        # iterate as delta can be 2d
        self.delta = np.zeros_like(prob_setup['delta'])
        for i in range(self.n):
            self.delta[i] = prob_setup['delta'][i]/self.demand[i]

    # HELPERS 
    def vec_to_mat(self, vec):
        '''
        Converts matrix to 
        ''' 
        return vec.reshape(self.n, self.m)

    def mat_to_vec(self, mat):
        '''
        Converts matrix to 
        ''' 
        return mat.flatten()

    def get_max_demand(self,):
        '''
        returns possible max demand
        '''
        # return np.sum(self.N_veh*self.mu*self.max_rho)/self.n
        return np.sum(self.N_veh*self.mu*self.max_rho)

    def get_util(self, flow):
        '''
        return vector of length n_modes
        server utilization rho for mode j 
        '''
        return flow.sum(axis=0)/(self.mu*self.N_veh)

    def get_latency(self, flow):
        '''
        latency vector \ell_{i,j}
        '''
        p_time = self.k/(1 + self.beta*self.N_veh*(1-self.get_util(flow)))
        lat = p_time + self.s_time + self.t_time
        return lat
        
    def guess_init(self):
        '''
        Initial guess is just splitting demand equally per order
        '''
        flow = np.zeros((self.n,self.m))
        flow = ((self.N_veh*self.mu*self.max_rho)/np.sum(self.N_veh*self.mu*self.max_rho))*np.tile(self.demand,(self.m,1)).T
        return self.mat_to_vec(flow)

    def get_bounds(self,):
        # zero lb for flow 
        lb = np.zeros(self.n*self.m)
        # total demand is ub for flow
        # ub = np.ones_like(lb)*self.demand
        ub = np.tile(self.demand, (self.m,1)).T.flatten()

        # two constriants, utilization and demand
        cl = np.zeros(self.m+self.n)
        cu = np.zeros_like(cl)

        # first cons is utilization (ineq)
        cu[0:self.m] = self.max_rho

        # next cons is demand (eq)
        cu[self.m:self.m+self.n] = self.demand
        cl[self.m:self.m+self.n] = self.demand

        return lb, ub, cl, cu
    
    def get_dlat_df(self, rho):
        '''
        computes derivative information:
        d lat_{i,j} / d flow_{i',j'}

        NOTE: using i,j in code to index over i',j'

        Returns vector as (m x n) by (m x n) matrix where
        dlat[i,j] = d lat_{i//m,i mod m} / d flow_{j//m,j mod m}
        '''

        denom = 1 + self.beta*self.N_veh*(1-rho)
        w = (self.k*self.beta)/self.mu
        value = w/(denom**2)

        # only when j = j'
        row_j = np.tile(np.arange(self.m),self.n)
        mat_j = np.tile(row_j, (self.n*self.m,1))
        mask = (mat_j.T == mat_j)
        # mask is symmetric, but multiplication broadcasts over row 
        dlat = (value.flatten()*mask).T

        return dlat 
        
    # MAIN OPTIMIZATION METHODS
    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        flow = self.vec_to_mat(x)
        lat  = self.mat_to_vec(self.get_latency(flow))

        return np.inner(lat,x)/self.tot_demand

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        flow    = self.vec_to_mat(x)
        rho     = self.get_util(flow)

        # util cons (ineq)
        cons_1 = rho
        # demand (eq)
        cons_2 = flow.sum(axis=1)
        
        return np.concatenate((cons_1,cons_2))

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        flow    = self.vec_to_mat(x)
        lat     = self.get_latency(flow)
        rho     = self.get_util(flow)
        dlat_df = self.get_dlat_df(rho)

        dl_df   = self.mat_to_vec(lat) + (dlat_df.T@x)

        return  dl_df/self.tot_demand
    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        # redundant first constraint
        flow    = self.vec_to_mat(x)
        rho     = self.get_util(flow)

        jac = np.zeros((self.m+self.n,self.m*self.n))

        # util cons (ineq) over modes
        for j in range(self.m):
            dg_j = (np.arange(0,self.m)*np.ones_like(flow)==j)*(1/(self.mu*self.N_veh))
            jac[j,:] = dg_j.flatten()

        # demand (eq) over orders
        for i in range(self.n):
            dh_i = (np.arange(0,self.n)*np.ones_like(flow.T)==i)
            jac[self.m+i] = (dh_i.T).flatten()

        return jac

    # def hessianstructure(self):
    #     #
    #     # The structure of the Hessian
    #     # Note:
    #     # The default hessian structure is of a lower triangular matrix. Therefore
    #     # this function is redundant. I include it as an example for structure
    #     # callback.
    #     #

    #     return np.nonzero(np.tril(np.ones((self.m*self.n, self.m*self.n))))

    # NOTE: OLD HESSIAN is actually faster
    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        # will compute hessian the longway
        flow = self.vec_to_mat(x)
        rho = self.get_util(flow)
                 
        H = np.zeros((self.m*self.n,self.m*self.n))
        # iterate over first derivative
        row_count = 0
        for i_p in range(self.n):
            for j_p in range(self.m):
                # computing each row as a matrix will simplify indexing
                row_h = np.zeros((self.n,self.m))

                # compute second derivative component
                denom = 1 + (self.beta[:,j_p]*self.N_veh[j_p])*(1-rho[j_p])
                numer = (2*self.k[j_p]*flow[:,j_p]*(self.beta[:,j_p]**2))/(self.mu[j_p]**2)
                row_h[:,j_p] = (numer/(denom**3)).sum()

                # compute part that depends on i_p, jp
                denom = 1 + (self.beta[i_p,j_p]*self.N_veh[j_p])*(1-rho[j_p])
                numer = (self.k[j_p]*self.beta[i_p,j_p])/(self.mu[j_p])
                row_h[:,j_p] += numer/(denom**2)

                # compute the part that depends on i
                denom = 1 + (self.beta[:,j_p]*self.N_veh[j_p])*(1-rho[j_p])
                numer = (self.k[j_p]*self.beta[:,j_p])/(self.mu[j_p])
                row_h[:,j_p] += numer/(denom**2)

                # row_h /= (self.n*self.demand)
                row_h /= (self.tot_demand)
                H[row_count] = self.mat_to_vec(row_h)
                row_count += 1
        
        return obj_factor*H

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        #
        # Example for the use of the intermediate callback.
        #
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))

    # NOTE
    # SLOWER CODE BUT EASIER TO READ SAVED FOR LATER
    # def slow_hessian(self, x, lagrange, obj_factor):
    #     #
    #     # The callback for calculating the Hessian
    #     #
    #     #NOTE: constraints have zero 2nd derivative

    #     flow     = self.vec_to_mat(x)
    #     lat      = self.get_latency(flow)
    #     rho      = self.get_util(flow)
    #     dlat_df  = self.get_dlat_df(rho)

    #     # will compute hessian the longway
    #     H = np.zeros((self.m*self.n,self.m*self.n))

    #     # iterate over first derivative (ip, jp)
    #     for i in range(self.n):
    #         for j in range(self.m):
    #             # H[i*self.m+j,:] += d2lat_df[:,i*self.m+j,:].T@x
    #             # get instance of second derivative and specific ij
    #             # NOTE: This requireid for masking matrices not to get too large
    #             d2lat_ipjp = self.get_d2lat_df(rho, i, j)
    #             H[i*self.m+j,:] += d2lat_ipjp.T@x
    #             H[i*self.m+j,:] += dlat_df[:,i*self.m+j]
    #             H[i*self.m+j,:] += dlat_df[i*self.m+j,:]

    #     H /= (self.n*self.demand)
    #     return obj_factor*H

    # # DERIVATIVE METHODS
    # def get_d2lat_df(self, rho, ip ,jp):
    #     '''
    #     computes derivative information:
    #     d^2 lat_{i,j} / d flow_{i'',j''} d flow_{i',j'}
    #     '''

    #     denom = 1 + self.beta*self.N_veh*(1-rho)
    #     w = (2*self.k*(self.beta**2))/(self.mu**2)
    #     value = w/(denom**3)

    #     # only when j = j' = j''
    #     row_j = np.tile(np.arange(self.m),self.n)
    #     mat_j = np.tile(row_j, (self.n*self.m,1))
    #     # (j compared to jpp) * (j compared to jp)
    #     mask = (mat_j.T == mat_j)*(mat_j == jp)
    #     # mask is symmetric, but multiplication broadcasts over row 
    #     d2lat = (value.flatten()*mask).T

    #     # NOTE: METHOD WITHOUT INDEX
    #     # this crashes the memory 

    #     # row_j = np.tile(np.arange(self.m),self.n)
    #     # mat_jpp = np.tile(row_j, (self.n*self.m,self.n*self.m,1))
    #     # mat_jp = np.transpose(mat_jpp,[0,2,1])
    #     # mat_j = np.transpose(mat_jpp,[2,1,0])

    #     # mask = (mat_j == mat_jp)*(mat_j == mat_jpp)
    #     # d2lat = value.flatten()*mask
    #     # d2lat = np.transpose(d2lat,[2,1,0])
    #     return d2lat