import numpy as np
import cyipopt

class mdrp_prob_elastic:

    def __init__(self, prob_setup):
        '''
        Initializing the mdrp non-linear program to be solved.
        All parameters are np arrays
        All units and rates are in hours
        ELASTIC version:
        - x is now a concatenation of flow and price
        - permutations required 

        NOTE:
        Main optimization methods use 'x' in vector form
        Helpers will output items in matrix form for indexing convenience

        NOTE:
        Whenever _sig is the suffix of a variable, we are referreing to a permuted
        version of the variable based on the ordering of the latencies (sigma)

        Parameters required:
            s_time     - I(row) x J(col), service time of ea. (order,mode) pair
            t_time     - I(row) x J(col), travel time of ea. (order,mode) pair
            cost       - I(row) x J(col), cost of ea. (order,mode) pair
            beta       - I(row) x J(col), portion of couriers available of ea. (order,mode) pair
            N          - J(row), total available couriers for each mode
            k          - J(row), unit of time for uniform distribution of couriers
            mu         - J(row), average rate of order completions by coruier mode
            max_rho    - scalar, maximum server utilization
            max_cost   - scalar, maximum cost
            demand     - scalar, between 0 and 1, scales the demand by a function of max demand

            alpha_0    
            delta
            gamma
            min_price
        '''

        self.s_time   = prob_setup['s_time']
        self.t_time   = prob_setup['t_time']
        self.cost     = prob_setup['cost']
        self.beta     = prob_setup['beta']
        self.N        = prob_setup['N']
        self.k        = prob_setup['k']
        self.mu       = prob_setup['mu']
        self.max_rho  = prob_setup['max_rho']
        self.max_cost = prob_setup['max_cost']

        self.n = self.t_time.shape[0]
        self.m = self.t_time.shape[1]
        
        if prob_setup['scale']:
            self.demand = self.get_max_demand()*prob_setup['demand']
        else:
            self.demand = prob_setup['demand']

        # variables introduced from elastic formulation
        self.alpha_0 = prob_setup['alpha_0']
        self.delta   = prob_setup['delta']/self.demand
        self.gamma   = prob_setup['gamma']
        self.min_tau = prob_setup['min_tau']
        # price/time tradeoff
        self.alpha     = lambda x: self.alpha_0 + self.delta*x
        self.alpha_inv = lambda x: (x-self.alpha_0)/self.delta
    # HELPERS
    # def _vec_to_mat(self, vec):
    #     '''
    #     Converts matrix to 
    #     ''' 
    #     return vec.reshape(self.n, self.m)

    # def _mat_to_vec(self, mat):
    #     '''
    #     Converts matrix to 
    #     ''' 
    #     return mat.flatten()

    def get_permutation(self,lat):
        '''
        For each order i, we want to sort j by latency

        NOTE: This permutation is only applied variables over (i,j)
        sigma[i,j]: the j'th fastest mode for order i  
        '''
        return np.argsort(lat)

    def un_permute(self, sigma, mat):
        '''
        takes permuted matrix max and ordering, 
        and returns matrix in original order

        NOTE: Assume mat has n rows, as it should!
        '''
        new_mat = np.zeros_like(mat)
        new_mat[np.arange(self.n),sigma.T] = mat.T
        return new_mat
    def permute(self, sigma, mat):
        '''
        permutes mat using sigma as ordering

        NOTE: Assume mat has n rows, as it should!
        '''
        return mat[np.arange(self.n),sigma.T].T

    def split_x(self,x):
        '''
        x is concat of [flow, price]
        returns flow as matrix and price and vector
        '''
        f = x[0:self.m*self.n].reshape(self.n,self.m)
        tau = x[self.m*self.n:]

        return f, tau

    def merge_x(self,flow,tau):
        '''
        combine flow and price (min) into one vector
        flow:  orders x modes
        price: modes
        '''
        x = np.concatenate((flow.flatten(),tau))
        return x

    def guess_init(self):
        '''
        Initial guess having a minimum price of zero, 
        and splitting demand equally.
        '''
        # x_flow = (self.demand*np.ones(self.n*self.m))/self.m
        flow = np.zeros((self.n,self.m))
        flow[:,] = self.N*self.mu/np.dot(self.N,self.mu)*self.demand

        lat = self.get_latency(flow)
        sigma = self.get_permutation(lat)
        lat_sig = self.permute(sigma,lat)
        # get estimate of tau corresponding to this
        tau = (self.gamma-lat_sig[:,-1])/self.alpha(flow.sum(axis=1))

        x = np.concatenate((flow.flatten(),tau))
        

        # NOTE: CONSTANT PERMUTATION
        self.sigma = sigma

        return x

    def get_bounds(self,):
        # zero lower bound for price and flow
        lb = np.zeros(self.n*self.m+self.n)
        # total demand and max cost as upper bound
        ub = np.concatenate((np.ones(self.n*self.m)*self.demand,
                             np.ones(self.n)*self.max_cost))

        # two constraints, utilization (over modes) and demand (over orders)
        cl = np.zeros(self.m+self.n)
        cu = np.zeros_like(cl)

        # utilization is inequality constraint
        cu[0:self.m] = self.max_rho

        # demand is equality constraint
        cu[self.m:] = 0
        # NOTE: Relaxation
        # cu[self.m:] = 2

        return lb, ub, cl, cu

    def get_util(self, flow):
        '''
        return vector of length m
        server utilization rho for mode j 
        '''
        return flow.sum(axis=0)/(self.mu*self.N)

    def get_latency(self, flow):
        '''
        latency vector \ell_{i,j}

        # NOTE: unordered as it is based on flow variable
        '''
        p_time = self.k/(1 + self.beta*self.N*(1-self.get_util(flow)))
        lat = p_time + self.s_time + self.t_time
        return lat
    
    def get_prices(self, flow_sig, lat_sig, tau):
        '''
        returns prices assuming latency & flow is ordered
        '''
        prices_sig = np.zeros_like(flow_sig)
        prices_sig[:,-1] = tau
        # iterate down from j=J-1 
        a_j = flow_sig[:,0:self.m-1].sum(axis=1)
        for j in range(self.m-2,-1,-1):
            diff = (lat_sig[:,j+1]-lat_sig[:,j])/self.alpha(a_j) 
            prices_sig[:,j] = prices_sig[:,j+1] + diff
            a_j -= flow_sig[:,j]

        return prices_sig

    def get_max_demand(self,):
        '''
        returns possible max demand
        '''
        return (np.inner(self.N,self.mu)*self.max_rho)/self.n

    #--------------------------------------------------------------#
    # DERIVATIVE FUNCTIONS # 
    #--------------------------------------------------------------#

    def get_dlat_df(self, sigma, beta_sig, rho):
        '''
        computes derivative information:
        d lat_{i,j} / d flow_{i',j'}

        NOTE: using i,j in code to index over i',j'

        NOTE: Assumes both indeces correspond to sorted versions
        Returns vector as (m x n) by (m x n) matrix where
        dlat[i,j] = d lat_{i//m,i mod m} / d flow_{j//m,j mod m}
        '''

        dlat = np.zeros((self.n*self.m,self.n*self.m))

        denom = 1 + beta_sig*self.N[sigma]*(1-rho[sigma])
        w = (self.k/self.mu)[sigma]*beta_sig  
        value = w/(denom**2)

        # NOTE: for loop iteration for computing matching indeces of permutations
        # maybe can be shortened
        for i in range(self.n):
            for j in range(self.m):
                mask = sigma==sigma[i,j]
                dlat[:,i*self.m + j] = (value*mask).flatten()

        return dlat

    def get_d2lat_df(self, sigma, beta_sig, rho):
        '''
        computes derivative information:
        d^2 lat_{i,j} / d flow_{i',j'} d flow_{i'',j''}
        '''

        d2lat_df = np.zeros((self.n*self.m,self.n*self.m,self.n*self.m))
        # compute value as before
        denom = 1 + beta_sig*self.N[sigma]*(1-rho[sigma])
        w = (self.k/(self.mu**2))[sigma]*(2*(beta_sig**2))  
        value = w/(denom**3)

        for i in range(self.n):
            for j in range(self.m):
                for i_p in range(self.n):
                    for j_p in range(self.m):
                        mask = (sigma==sigma[i,j])*(sigma==sigma[i_p,j_p])
                        d2lat_df[:,i*self.m + j,i_p*self.m + j_p] = (value*mask).flatten()

        return d2lat_df

    def get_dt_df(self, lat_sig, flow_sig, dlat_df):
        '''
        computes derivative information:
        d tau_{i,j} / d flow_{i',j'}
        '''

        dt_df = np.zeros((self.n*self.m,self.n*self.m))

        for i in range(self.n):
            for j in range(self.m):
                left = np.zeros(self.n*self.m)
                for k in range(j,self.m-1):
                    numer = dlat_df[i*self.m+k+1,:] - dlat_df[i*self.m+k,:]
                    denom = self.alpha(flow_sig[i,0:k].sum())
                    left += numer/denom

                dt_df[i*self.m + j,:] = left.flatten()
                # must compute right part seperately for variable wrt 
                for j_p in range(self.m):
                    right = 0
                    for k in range(max(j,j_p),self.m-1):
                        numer = self.delta*(lat_sig[i,k+1] - lat_sig[i,k])
                        denom = self.alpha(flow_sig[i,0:k].sum())**2
                        right += numer/denom

                    # only when i = i_p
                    dt_df[i*self.m + j,i*self.m + j_p] -= right

        return dt_df
        

    def get_d2t_df(self, lat_sig, flow_sig, dlat_df, d2lat_df):
        '''
        computes second derivative information:
        d^2 tau_{i,j} / d flow_{i',j'}^2
        '''

        d2t_df = np.zeros((self.n*self.m,self.n*self.m,self.n*self.m))
        # PART 1
        # iterate over l variable
        for i in range(self.n):
            for j in range(self.m):
                # 2nd derivative term is matrix
                left = np.zeros((self.n*self.m, self.n*self.m))
                for k in range(j,self.m-1):
                    numer = d2lat_df[i*self.m+k+1] - d2lat_df[i*self.m+k]
                    denom = self.alpha(flow_sig[i,0:k].sum())
                    left += numer/denom
                d2t_df[i*self.m + j] = left

                # 1st derivative term is vector 
                # refer to second derivative as i_pp, j_pp
                for j_pp in range(self.m):
                    right = np.zeros(self.n*self.m)
                    for k in range(max(j,j_pp),self.m-1):
                        numer = self.delta*(dlat_df[i*self.m+k+1] - dlat_df[i*self.m+k])
                        denom = self.alpha(flow_sig[i,0:k].sum())**2
                        right += numer/denom

                    # only when i = i_pp
                    d2t_df[i*self.m + j,:,i*self.m + j_pp] -= right
        
        # PART 2
        # iterate over l variable
        for i in range(self.n):
            for j in range(self.m):
                # refer to first derivative as i_p, j_p
                for j_p in range(self.m):
                    left = np.zeros(self.n*self.m)
                    for k in range(max(j,j_p),self.m-1):
                        # dlat in this case is wrt i_pp, j_pp
                        numer = self.delta*(dlat_df[i*self.m+k+1] - dlat_df[i*self.m+k])
                        denom = self.alpha(flow_sig[i,0:k].sum())**2
                        left += numer/denom
                    # only when i = i_p
                    d2t_df[i*self.m + j,i*self.m + j_p,:] -= left
            
                for j_p in range(self.m):
                    for j_pp in range(self.m):
                        right = 0
                        for k in range(max(j,j_p,j_pp),self.m-1):
                            numer = 2*(self.delta**2)*(lat_sig[i,k+1] - lat_sig[i,k])
                            denom = self.alpha(flow_sig[i,0:k].sum())**3
                            right += numer/denom
                        # only when i = i_pp and i = i_p
                        d2t_df[i*self.m + j,i*self.m + j_p,i*self.m + j_pp] += right
    
        return d2t_df

    #--------------------------------------------------------------#
    # MAIN OPTIMIZATION METHODS # 
    # MUST TAKE INPUT X #
    # Each method will fetch a set amount of parameters from x
    #--------------------------------------------------------------#

    def objective(self, x):
        '''
        The callback for calculating the objective
        Requires update to:
        '''
        # fetch parameters
        flow, tau  = self.split_x(x)
        lat        = self.get_latency(flow)
        # rho        = self.get_util(flow)
    
        # sigma      = self.get_permutation(lat)
        sigma = self.sigma 

        lat_sig    = self.permute(sigma, lat)
        flow_sig   = self.permute(sigma, flow)
        # cost_sig   = self.permute(sigma, self.cost)
        prices_sig = self.get_prices(flow_sig, lat_sig, tau)
        prices     = self.un_permute(sigma,prices_sig)
        # beta_sig   = self.permute(sigma, self.beta)
        # dlat_df    = self.get_dlat_df(sigma, beta_sig, rho)
        # dt_df      = self.get_dt_df(lat_sig, flow_sig, dlat_df)
        # end fetch

        # cleaner than using np inner
        return (flow*(prices-self.cost)).sum()

    def constraints(self, x):
        '''
        The callback for calculating the constraints
        '''
        # fetch parameters
        flow, tau  = self.split_x(x)
        lat        = self.get_latency(flow)
        rho        = self.get_util(flow)

        # sigma      = self.get_permutation(lat)
        sigma      = self.sigma

        lat_sig    = self.permute(sigma, lat)
        # flow_sig   = self.permute(sigma, flow)
        # cost_sig   = self.permute(sigma, self.cost)
        # prices_sig = self.get_prices(flow_sig, lat_sig, tau)
        # prices     = self.un_permute(sigma,prices_sig)
        # beta_sig   = self.permute(sigma, self.beta)
        # dlat_df    = self.get_dlat_df(sigma, beta_sig, rho)
        # dt_df      = self.get_dt_df(lat_sig, flow_sig, dlat_df)
        # end fetch

        # util over modes
        cons_1 = rho
        # demand (elastic)
        cons_2 = self.alpha_inv((self.gamma - lat_sig[:,-1])/tau)
        cons_2 -= flow.sum(axis=1)
        
        return np.concatenate((cons_1,cons_2))
        # return cons_1, cons_2


    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        # fetch parameters
        flow, tau  = self.split_x(x)
        lat        = self.get_latency(flow)
        rho        = self.get_util(flow)

        # sigma      = self.get_permutation(lat)
        sigma      = self.sigma

        lat_sig    = self.permute(sigma, lat)
        flow_sig   = self.permute(sigma, flow)
        cost_sig   = self.permute(sigma, self.cost)
        prices_sig = self.get_prices(flow_sig, lat_sig, tau)
        # prices     = self.un_permute(sigma,prices_sig)
        beta_sig   = self.permute(sigma, self.beta)
        dlat_df    = self.get_dlat_df(sigma, beta_sig, rho)
        dt_df      = self.get_dt_df(lat_sig, flow_sig, dlat_df)
        # end fetch

        grad_flow = np.zeros((self.n,self.m))
        grad_tau = np.zeros(self.n)

        # gradient wrt flow variable 
        grad_flow += prices_sig - cost_sig
        grad_flow += (flow_sig.flatten() @ dt_df).reshape(self.n,self.m)
        # NOTE: Un permute 
        grad_flow = self.un_permute(sigma, grad_flow)

        # gradient wrt price variable
        grad_tau += flow_sig.sum(axis=1)

        return np.concatenate((grad_flow.flatten(),grad_tau))

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        # fetch parameters
        flow, tau  = self.split_x(x)
        lat        = self.get_latency(flow)
        rho        = self.get_util(flow)

        # sigma      = self.get_permutation(lat)
        sigma      = self.sigma

        lat_sig    = self.permute(sigma, lat)
        # flow_sig   = self.permute(sigma, flow)
        # cost_sig   = self.permute(sigma, self.cost)
        # prices_sig = self.get_prices(flow_sig, lat_sig, tau)
        # prices     = self.un_permute(sigma,prices_sig)
        beta_sig   = self.permute(sigma, self.beta)
        dlat_df    = self.get_dlat_df(sigma, beta_sig, rho)
        # dt_df      = self.get_dt_df(lat_sig, flow_sig, dlat_df)
        # end fetch

        jac = np.zeros((self.m+self.n,self.n*self.m+self.n))

        # server constraints
        # dg1_df
        for j in range(self.m):
            dg1j_df = (np.arange(0,self.m)*np.ones_like(flow)==j)*(1/(self.mu*self.N))
            jac[j,0:self.n*self.m] = dg1j_df.flatten()
        
        # dg1_dtau
        jac[0:self.m,self.n*self.m:] = 0

        # dg2_df
        for i in range(self.n):
            # self.m-1 is last
            dlat_iJ_df = dlat_df[i*self.m+self.m-1]
            val = (-1*dlat_iJ_df)/(tau[i]*self.delta)
            # val *= (sigma[i,-1]==sigma).flatten() # redundant 
            val[i*self.m:i*self.m+self.m] -= 1
            # ORDER
            val_ordered = self.un_permute(sigma, val.reshape(self.n,self.m))
            jac[self.m+i,0:self.n*self.m] = val_ordered.flatten()

        # dg2_dtau
        for i in range(self.n):
            val = (lat_sig[i,-1] - self.gamma)/(self.delta*(tau[i]**2))
            jac[self.m+i,self.n*self.m+i] = val

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

    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        # fetch parameters
        flow, tau  = self.split_x(x)
        lat        = self.get_latency(flow)
        rho        = self.get_util(flow)

        # sigma      = self.get_permutation(lat)
        sigma      = self.sigma

        lat_sig    = self.permute(sigma, lat)
        flow_sig   = self.permute(sigma, flow)
        cost_sig   = self.permute(sigma, self.cost)
        prices_sig = self.get_prices(flow_sig, lat_sig, tau)
        prices     = self.un_permute(sigma,prices_sig)
        beta_sig   = self.permute(sigma, self.beta)
        dlat_df    = self.get_dlat_df(sigma, beta_sig, rho)
        dtau_df    = self.get_dt_df(lat_sig, flow_sig, dlat_df)
        # second derivative information for hessian
        d2lat_df   = self.get_d2lat_df(sigma, beta_sig, rho)
        d2t_df     = self.get_d2t_df(lat_sig, flow_sig, dlat_df, d2lat_df)
        # end fetch

        # for notation
        n = self.n
        m = self.m

        # compute second derivative of objective in 4 parts
        H_obj = np.zeros((n*m+n,n*m+n))
        
        #-----------------------------------------------#
        # top left
        df_df = np.zeros((n*m,n*m))
        # first term is transposed matrix
        df_df += dtau_df.T
        # second term is summation sacled by x
        for i in range(n):
            for j in range(m):
                df_df += flow[i,j]*d2t_df[i*m+j]
        # UNPERMUTE
        mat = np.zeros_like(df_df)
        for i_row in range(n*m):
            row = df_df[i_row].reshape(n,m)
            mat[i_row] = self.un_permute(sigma, row).flatten()

        # print(mat.shape, H_obj[0:n*m,0:n*m].shape)
        H_obj[0:n*m,0:n*m] = mat
        #-----------------------------------------------#
        # top right (ones for identical orders)
        df_dt = np.zeros((n*m,n))
        for i in range(n):
            for j in range(m):
                df_dt[i*m+j,i]=1
        # no permutation needed as indep of j
        H_obj[0:n*m,n*m:n*m+n] = df_dt
        #-----------------------------------------------#
        # bot left (identical to top right)
        H_obj[n*m:n*m+n,0:n*m] = df_dt.T
        #-----------------------------------------------#
        # bot right (is zero)
        dt_dt = np.zeros((n,n))
        H_obj[n*m:n*m+n,n*m:n*m+n] = dt_dt
        #-----------------------------------------------#
        H = obj_factor*H_obj


        # first set of constraints has zero Hessian
        for i_cons in range(m):
            H_cons1 = np.zeros((n*m+n,n*m+n))
            H += lagrange[i_cons]*H_cons1

        # second set of constraints
        for i_cons in range(n):
            H_cons2 = np.zeros((n*m+n,n*m+n))
            #-----------------------------------------------#
            # top left
            df_df = np.zeros((n*m,n*m))
            df_df -= (d2lat_df[i_cons*m+m-1])/(tau[i_cons]*self.delta)
            # UNPERMUTE
            mat = np.zeros_like(df_df)
            for i_row in range(n*m):
                row = df_df[i_row].reshape(n,m)
                mat[i_row] = self.un_permute(sigma, row).flatten()
            H_cons2[0:n*m,0:n*m] = mat
            #-----------------------------------------------#
            # bot left (easier to permute)
            dt_df = np.zeros((n,n*m))
            val = dlat_df[i_cons*m+m-1]/(self.delta*(tau[i_cons]**2))
            dt_df[i_cons,:] = val
            # UNPERMUTE
            mat = np.zeros_like(dt_df)
            for i_row in range(n):
                row = dt_df[i_row].reshape(n,m)
                mat[i_row] = self.un_permute(sigma, row).flatten()

            H_cons2[n*m:n*m+n,0:n*m] = mat
            #-----------------------------------------------#
            # top right (identical to bot left)
            H_cons2[0:n*m,n*m:n*m+n] = mat.T
            #-----------------------------------------------#
            # bot right 
            dt_dt = np.zeros((n,n))
            val = (2*(self.gamma-lat_sig[i_cons,-1]))/(self.delta*(tau[i]**3))
            dt_dt[i_cons,i_cons] = val
            H_cons2[n*m:n*m+n,n*m:n*m+n] = dt_dt
            #-----------------------------------------------#
            H += lagrange[m+i_cons]*H_cons2

        return H
        # return

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