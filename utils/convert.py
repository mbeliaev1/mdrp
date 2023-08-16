import numpy as np
from utils.helpers import read_instance_information, traveltime
import pickle 

def convert_chi(path, parameters):
    '''
    Given path to grubhub instance along with parameter dict,
    this function will output the corresponding prob setup required by 
    the non lienar program class
    Parameters:
    NOTE: Equal rate for each order
    '''
    prob_setup = {}
    airport = 75
    vertiport = 31
    # seed for reproducing
    rng = np.random.default_rng(parameters['seed'])
    data = pickle.load(open(parameters['data_path'],'rb'))
    n_modes = len(parameters['N'])

    rate      = []
    lat_car   = []
    lat_drone = []
    min_dph   = []
    avg_dph   = []
    # masks for future indexing
    from_air    = []
    vert_to_air = []
    air_to_vert = []

    # iterate over pikcup locations
    for pickup in range(len(data['Rate'][0])):
        # if not picked up at aiport, only care about dropoffs to the aiport
        if pickup != airport:
            
            from_air.append(False)
            air_to_vert.append(False)
            if pickup == vertiport:
                vert_to_air.append(True)
            else:
                vert_to_air.append(False)

            # rate is equivalent
            rate.append(data['Rate'][pickup,airport])
            # car latency is direct trip length
            lat_car.append(data['Latency'][pickup,airport,1,0])
            # drone will go to vertiport first (unless already there)
            pick_to_vert = data['Latency'][pickup,vertiport,1,0]*(pickup != vertiport)
            lat_drone.append(pick_to_vert+10/60) # 10 min constant added

            # take care of value here (for car)
            min_dph.append(data['Value'][pickup,airport,1,0] - data['Value'][pickup,airport,0,1])
            avg_dph.append(data['Value'][pickup,airport,1,0])

        # if picked up at airport, include entire row
        else:
            # iterate over every dropoff location
            for dropoff in range(len(data['Rate'][0])):
                
                from_air.append(True)
                vert_to_air.append(False)
                if dropoff == vertiport:
                    air_to_vert.append(True)
                else:
                    air_to_vert.append(False)

                rate.append(data['Rate'][airport,dropoff])
                # car latency is direct trip length
                lat_car.append(data['Latency'][airport,dropoff,1,0])
                # drone will go to vertiport first (unless already there)
                vert_to_drop = data['Latency'][vertiport,dropoff,1,0]*(dropoff != vertiport)
                lat_drone.append(vert_to_drop+10/60) # 10 min constant added

                # take care of value here (for car)
                min_dph.append(data['Value'][airport,dropoff,1,0] - data['Value'][airport,dropoff,0,1])
                avg_dph.append(data['Value'][airport,dropoff,1,0])

    rate      = np.array(rate)
    rate /= rate.sum()
    lat_car   = np.array(lat_car)
    lat_drone = np.array(lat_drone)
    min_dph     = np.array(min_dph)
    avg_dph     = np.array(avg_dph)
    from_air    = np.array(from_air)
    vert_to_air = np.array(vert_to_air)
    air_to_vert = np.array(air_to_vert)

    n_orders = len(rate)

    # TRAVEL TIME
    t_time = np.zeros((n_orders,n_modes))
    t_time[:,0] = lat_drone
    t_time[:,1] = lat_car
    t_time[:,2] = lat_car

    N = np.array(parameters['N'])
    k = np.array([0,5,5])/60 # 5 minutes away for cars, zero for drones

    beta = np.zeros((n_orders,n_modes))
    # from airport always available 
    beta[from_air] = 1

    B_max = parameters['Beta_max']
    B_min = parameters['Beta_min']

    theta = (np.log(B_max)-np.log(B_min))/(rate[~from_air].max()-rate[~from_air].min())
    c     = np.log(B_min)-rate[~from_air].min()*theta
    beta[~from_air]  = np.tile(np.exp(theta*rate[~from_air]+c),(n_modes,1)).T

    # service time is zero
    s_time = np.zeros((n_orders,n_modes))
    mu = 1/(((k/(1+beta*N*(1-np.array(parameters['max_rho'])))) + 0 + t_time).mean(axis=0))

    delta = np.zeros((n_orders, n_modes))
    alpha_0 = np.zeros((n_orders, n_modes))

    for i in range(n_orders):
        max_dph = 2*avg_dph[i]*parameters['alpha_scale'] - min_dph[i]
        delta[i,:] = (1/min_dph[i])-(1/max_dph)
        alpha_0[i,:] = 1/max_dph


    #NOTE: delta still needs to be scaled by demand to be proper slope
    prob_setup['delta']   =  delta
    prob_setup['alpha_0'] =  alpha_0
    prob_setup['rate']     = rate
    prob_setup['avg_dph']  = avg_dph
    prob_setup['from_air']   = from_air
    prob_setup['vert_to_air']   = vert_to_air
    prob_setup['air_to_vert']   = air_to_vert

    prob_setup['s_time']   = s_time
    # prob_setup['q_time']   = q_time
    prob_setup['t_time']   = t_time
    prob_setup['cost']     = parameters['cost']
    prob_setup['beta']     = beta
    prob_setup['N']        = N
    prob_setup['k']        = k
    prob_setup['mu']       = mu
    # code here
    
    return prob_setup


def convert_instance(path, parameters):
    '''
    Given path to grubhub instance along with parameter dict,
    this function will output the corresponding prob setup required by 
    the non lienar program class
    Parameters:
    NOTE: Equal rate for each order
    '''
    prob_setup = {}
    # seed for reproducing
    rng = np.random.default_rng(parameters['seed'])

    orders,restaurants,couriers,instanceparams,locations,meters_per_minute,\
    pickup_service_minutes,dropoff_service_minutes,target_click_to_door,\
    pay_per_order,guaranteed_pay_per_hour = read_instance_information(path)

    n_modes = len(parameters['speeds'])

    # service time
    s_t = (pickup_service_minutes+dropoff_service_minutes)/60
    s_time = np.array([s_t, s_t*parameters['u_ratios'][1], s_t*parameters['u_ratios'][2]])

    # travel time
    n_orders = len(orders)
    t_time = np.zeros((n_orders,n_modes))
    for i in range(n_orders):
        o_i = orders.index[i]
        for j in range(n_modes):
            # t_time[:,j] = np.asarray(temp)/parameters['speeds'][j]
            t_time[i,j] = traveltime(o_i,orders.at[o_i,'restaurant'],parameters['speeds'][j],locations)/60

    # beta, N, k, mu All together.
    N = np.array(parameters['N'])
    k = np.ones(n_modes)*10/60 # 10 minutes around center
    # sample the car locs directly
    n_cars = N[0]
    car_locs = np.array([couriers.x,couriers.y]).T
    car_locs = car_locs[rng.choice(car_locs.shape[0],n_cars)] # sample N

    # lets make drone couriers (unifromly around entire grid)
    # n_drones = len(car_locs)*parameters['N_ratios'][1]
    n_drones = N[1]
    grid_x = [restaurants.x.min(),restaurants.x.max()]
    grid_y = [restaurants.y.min(),restaurants.y.max()]
    drone_locs = np.array([grid_x[0]+rng.random(n_drones)*(grid_x[1]-grid_x[0]),
                        grid_y[0]+rng.random(n_drones)*(grid_y[1]-grid_y[0])]).T

    # n_droids = len(car_locs)*parameters['N_ratios'][2]
    n_droids = N[2]
    grid_x = [restaurants.x.mean()-0.5*restaurants.x.std(),restaurants.x.mean()+0.5*restaurants.x.std()]
    grid_y = [restaurants.y.mean()-0.5*restaurants.y.std(),restaurants.y.mean()+0.5*restaurants.y.std()]
    droid_locs = np.array([grid_x[0]+rng.random(n_droids)*(grid_x[1]-grid_x[0]),
                        grid_y[0]+rng.random(n_droids)*(grid_y[1]-grid_y[0])]).T

    all_locs = [car_locs, drone_locs, droid_locs]
    # beta
    beta = np.zeros_like(t_time)
    for i in range(1,n_orders+1):
        rest = orders.at['o%d'%i,'restaurant']
        loc = [restaurants.at[rest,'x'],restaurants.at[rest,'y']]
        for j in range(n_modes):
            count = 0
            # count number of vehicles within range for ea mode
            for cour_loc in all_locs[j]:
                dist = np.sqrt(((cour_loc-loc)**2).sum())
                if (dist/(parameters['speeds'][j]))/60 < k[j]:
                    count+=1
            beta[i-1,j] = count/N[j]
    # average order time
    mu = 1/(((k/(1+beta*N*0.1)) + s_time + t_time).mean(axis=0))


    # Price preference is identical for each order
    #NOTE: delta still needs to be scaled by demand to be proper slope
    prob_setup['delta']   =  (1/parameters['min_dph'])-(1/parameters['max_dph'])*np.ones(n_orders)
    prob_setup['alpha_0'] =  (1/parameters['max_dph'])*np.ones(n_orders)

    prob_setup['s_time']   = s_time
    prob_setup['t_time']   = t_time
    # only used internally to check
    # prob_setup['q_time']   = np.array(q_time)
    prob_setup['cost']     = parameters['cost']
    prob_setup['beta']     = beta
    prob_setup['N']        = N
    prob_setup['k']        = k
    prob_setup['mu']       = mu
    prob_setup['rate']     = np.ones(n_orders)/n_orders
    # code here
    
    return prob_setup