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
            max_rho    - scalar, maximum server utilization
            max_cost   - scalar, maximum cost
            demand     - scalar, between 0 and 1, scales the demand by a function of max demand
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

        self.n_orders = self.t_time.shape[0]
        self.n_modes = self.t_time.shape[1]
        
        if prob_setup['scale']:
            self.demand = self._get_max_demand()*prob_setup['demand']
        else:
            self.demand = prob_setup['demand']
    # HELPERS 
    def get_bounds(self,):
        lb = np.zeros(self.n_orders*self.n_modes)
        ub = np.ones_like(lb)*self.demand

        cl = np.zeros(1+self.n_modes+self.n_orders)
        cu = np.zeros_like(cl)

        # first cons is cost constraint
        cu[0] = self.max_cost

        # next n_modes constraints are server constraints
        cu[1:1+self.n_modes] = self.max_rho

        # next n_orders constraints are flow 
        cu[1+self.n_modes:1+self.n_modes+self.n_orders] = self.demand
        cl[1+self.n_modes:1+self.n_modes+self.n_orders] = self.demand

        return lb, ub, cl, cu

    def _get_latency(self, x):
        '''
        latency vector \ell_{i,j}
        '''
        p_time = self.k/(1 + ((self.beta*self.N)/(1-self._get_util(x))))
        lat = p_time + self.s_time + self.t_time
        return lat
    
    def _get_max_demand(self,):
        '''
        returns possible max demand
        '''
        return (np.inner(self.N,self.mu)*self.max_rho)/self.n_orders

    def _get_util(self, x):
        '''
        return vector of length n_modes
        server utilization rho for mode j 
        '''
        mat_x = self._vec_to_mat(x)
        return mat_x.sum(axis=0)/(self.mu*self.N)


    def _vec_to_mat(self, vec):
        '''
        Converts matrix to 
        ''' 
        return vec.reshape(self.n_orders, self.n_modes)

    def _mat_to_vec(self, mat):
        '''
        Converts matrix to 
        ''' 
        return mat.flatten()
        
    def guess_init(self):
        '''
        Initial guess is just splitting demand equally per order
        '''
        x = (self.demand*np.ones(self.n_orders*self.n_modes))/self.n_modes
        return x


    # MAIN OPTIMIZATION METHODS
    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        lat = self._mat_to_vec(self._get_latency(x))

        return np.inner(lat,x)/(self.n_orders*self.demand)

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        mat_x = self._vec_to_mat(x)
        lat = self._get_latency(x)
        
        denom = 1 + (self.beta*self.N)*(1-self._get_util(x)) 
        numer = (self.k*mat_x)/(self.N*self.mu)
        left_part  = (numer/(denom**2)).sum(axis=0)
        grad = self._mat_to_vec(left_part + lat)

        return grad/self.n_orders

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        cons = []
        # cost contraints
        cost_cons = np.inner(self._mat_to_vec(self.cost),x)
        cons.append(cost_cons)

        # server constraints
        utils = self._get_util(x)
        for j in range(self.n_modes):
            cons.append(utils[j])

        # flow constraints
        mat_x = self._vec_to_mat(x)
        flows = mat_x.sum(axis=1)
        for i in range(self.n_orders):
            cons.append(flows[i])

        return np.asarray(cons)

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        # redundant first constraint
        jac = np.zeros(self.n_orders*self.n_modes)

        # cost constraint
        jac = np.concatenate(([jac],[self._mat_to_vec(self.cost)])) 

        # server constraints
        for j in range(self.n_modes):
            values = np.zeros((self.n_orders,self.n_modes))
            values[:,j] = 1/(self.N[j]*self.mu[j])
            # print(jac.shape,values.shape)
            jac = np.concatenate((jac,[self._mat_to_vec(values)]))

        # flow constraints
        for i in range(self.n_orders):
            values = np.zeros((self.n_orders,self.n_modes))
            values[i,:] = 1
            # print(jac.shape,values.shape)
            jac = np.concatenate((jac,[self._mat_to_vec(values)]))

        return jac[1:]

    # def hessianstructure(self):
    #     #
    #     # The structure of the Hessian
    #     # Note:
    #     # The default hessian structure is of a lower triangular matrix. Therefore
    #     # this function is redundant. I include it as an example for structure
    #     # callback.
    #     #

    #     return np.nonzero(np.tril(np.ones((self.n_modes*self.n_orders, self.n_modes*self.n_orders))))

    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        # will compute hessian the longway
        H = np.zeros((self.n_modes*self.n_orders,self.n_modes*self.n_orders))

        mat_x = self._vec_to_mat(x)
        
        # lat = self._get_latency(x)
        # denom = 1 + (self.beta*self.N)*(1-self._get_util(x)) 
        # numer = (self.k*mat_x)/(self.N*self.mu)
        # left_part  = (numer/(denom**2)).sum(axis=0)
        # grad = self._mat_to_vec(left_part + lat)    
        # Each row of the hessian matrix corresponds to the gradient of
        # the (i,j)th coordinate of the gradient of f
         
        row_count = 0
        for i_p in range(self.n_orders):
            util = self._get_util(x)
            for j_p in range(self.n_modes):
                # computing each row as a matrix will simplify indexing
                row_h = np.zeros((self.n_orders,self.n_modes))

                # compute part that sums over all i
                denom = 1 + (self.beta[:,j_p]*self.N[j_p])*(1-util[j_p])
                numer = (2*self.k[j_p]*mat_x[:,j_p])/((self.N[j_p]*self.mu[j_p])**2)
                row_h[:,j_p] = (numer/(denom**3)).sum()

                # compute part that depends on i_p
                denom = 1 + (self.beta[i_p,j_p]*self.N[j_p])*(1-util[j_p])
                numer = self.k[j_p]/(self.N[j_p]*self.mu[j_p])
                row_h[:,j_p] += numer/(denom**2)

                # compute the part that depends on i
                denom = 1 + (self.beta[:,j_p]*self.N[j_p])*(1-util[j_p])
                numer = self.k[j_p]/(self.N[j_p]*self.mu[j_p])
                row_h[:,j_p] += numer/(denom**2)

                row_h /= self.n_orders
                H[row_count] = self._mat_to_vec(row_h)
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