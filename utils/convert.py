import numpy as np
from utils.helpers import read_instance_information, traveltime
import pandas as pd


def convert_instance(path, parameters):
    '''
    Given path to grubhub instance along with parameter dict,
    this function will output the corresponding prob setup required by 
    the non lienar program class
    Parameters:

    '''
    prob_setup = {}
    if path != 'test':
        orders,restaurants,couriers,instanceparams,locations,meters_per_minute,\
        pickup_service_minutes,dropoff_service_minutes,target_click_to_door,\
        pay_per_order,guaranteed_pay_per_hour = read_instance_information(path)

        n_modes = len(parameters['speeds'])

        # service time
        s_t = (pickup_service_minutes+dropoff_service_minutes)/60
        s_time = np.array([s_t, s_t*parameters['u_ratios'][1], s_t*parameters['u_ratios'][2]])
        t_time = np.array

        # travel time
        n_orders = len(orders)
        t_time = np.zeros((n_orders,n_modes))
        temp = []
        for order in orders.index:
            temp.append(traveltime(order,orders.at[order,'restaurant'],meters_per_minute,locations)/60)
        for j in range(n_modes):
            t_time[:,j] = np.asarray(temp)/parameters['speeds'][j]

        # costs just simple constant model
        cost = np.zeros_like(t_time)
        mode_costs = parameters['mode_costs']
        for j in range(n_modes):
            cost[:,j] = mode_costs[j]*np.ones(n_orders)

        # beta, N, k, mu All together.
        N = np.array(parameters['N'])

        car_locs = np.array([couriers.x,couriers.y]).T
        car_locs = car_locs[np.random.choice(10,5)] # sample N
        # N = np.array(parameters['N_ratios'])*len(car_locs)
        # print(N)
        k = np.ones(n_modes)*10/60 # 10 minutes around center

        # lets make drone couriers (unifromly around entire grid)
        # n_drones = len(car_locs)*parameters['N_ratios'][1]
        n_drones = N[1]
        grid_x = [restaurants.x.min(),restaurants.x.max()]
        grid_y = [restaurants.y.min(),restaurants.y.max()]
        drone_locs = np.array([grid_x[0]+np.random.rand(n_drones)*(grid_x[1]-grid_x[0]),
                            grid_y[0]+np.random.rand(n_drones)*(grid_y[1]-grid_y[0])]).T

        # n_droids = len(car_locs)*parameters['N_ratios'][2]
        n_droids = N[2]
        grid_x = [restaurants.x.mean()-0.5*restaurants.x.std(),restaurants.x.mean()+0.5*restaurants.x.std()]
        grid_y = [restaurants.y.mean()-0.5*restaurants.y.std(),restaurants.y.mean()+0.5*restaurants.y.std()]
        droid_locs = np.array([grid_x[0]+np.random.rand(n_droids)*(grid_x[1]-grid_x[0]),
                            grid_y[0]+np.random.rand(n_droids)*(grid_y[1]-grid_y[0])]).T

        all_locs = [car_locs, drone_locs, droid_locs]
        # beta
        beta = np.zeros_like(t_time)
        for i in range(1,n_orders+1):
            rest = orders.at['o%d'%i,'restaurant']
            loc = [restaurants.at[rest,'x'],restaurants.at[rest,'y']]
            for j in range(n_modes):
                count = 0
                for cour_loc in all_locs[j]:
                    dist = np.sqrt(((cour_loc-loc)**2).sum())
                    # print((dist/(meters_per_minute*parameters['speeds'][j]))/60)
                    if (dist/(meters_per_minute*parameters['speeds'][j]))/60 < k[j]:
                        count+=1
                beta[i-1,j] = count/N[j]
        # average order time
        mu = 1/(((k/(1+beta*N*(0.1))) + s_time + t_time).mean(axis=0))

        prob_setup['s_time']   = s_time
        prob_setup['t_time']   = t_time
        prob_setup['cost']     = cost
        prob_setup['beta']     = beta
        prob_setup['N']        = N
        prob_setup['k']        = k
        prob_setup['mu']       = mu
        # code here
    
    else:
        # testing code here
        num_i = 5
        num_j = 3
        # prob_setup['s_time']   = np.array([[5,5,5,5,5],
        #                            [5,5,5,5,5],
        #                            [5,5,5,5,5]]).T/60
        prob_setup['s_time']   = np.array([5,5,5])/60
        prob_setup['t_time']   = np.array([[45,30,10,10,10],
                                        [15,10,10,8,5],
                                        [20,25,30,20,20]]).T/60
        prob_setup['cost']     = np.array([[10,10,10,10,10],
                                        [5,5,5,5,5],
                                        [5,5,5,5,5]]).T
        prob_setup['beta']     = np.array([[0.5,0.5,0.5,0.5,0.5],
                                        [0.5,0.5,0.5,0.5,0.5],
                                        [0.5,0.5,0.5,0.5,0.5]]).T
        prob_setup['N']        = np.ones(num_j)*10
        prob_setup['k']        = np.ones(num_j)*10/60
        prob_setup['mu']       = np.array([2,4,3])
    
    return prob_setup