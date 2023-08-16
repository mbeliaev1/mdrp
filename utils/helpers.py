import numpy as np
from utils.nlp_mdrp import mdrp_prob
import pandas as pd

import os
import sys


def alpha_func(alpha_0, delta, x):
    return alpha_0 + delta*x

def get_prices(result):
    mdrp    = result['mdrp']
    x       = result['sol']

    flow       = mdrp.vec_to_mat(np.round(x,4))
    tot_demand = mdrp.tot_demand
    # alphas = []
    # for i in range(mdrp.n):    
    #     alphas.append(lambda x: mdrp.alpha_0[i] + mdrp.delta[i]*x)

    # First compute prices assuming cheapest option is zero
    prices = np.zeros_like(flow)
    lat = mdrp.get_latency(flow)
    for i in range(mdrp.n):
        j_sort = np.argsort(-lat[i])
        # j starts at cheapest option (longest latency)
        prices[i,j_sort[0]] = 0
        prev_j = j_sort[0]
        user_a = mdrp.demand[i] - flow[i,j_sort[0]]
        # iterate over rest of the price options
        for j in j_sort[1:]:
            prices[i,j] = prices[i,prev_j] + (lat[i,prev_j]-lat[i,j])/(alpha_func(mdrp.alpha_0[i],mdrp.delta[i],user_a))
            user_a -= flow[i,j]
            prev_j = j
    # Now compute the cheapest option based on cost constraint
    # cost per order used for meal delivery
    op_cost = np.inner(flow.sum(axis=0),mdrp.cost)
    min_price = (op_cost - np.inner(mdrp.mat_to_vec(prices),x))/tot_demand
    prices += min_price

    return prices, min_price

def eval_sol(result):
    '''
    Inputs
        x          - solution as vector outputted by mdrp
        mdrp       - nlp_mdrp instance for the problem setting
                     contains useful methods
    '''
    x          = result['sol']
    mdrp       = result['mdrp'] 
    parameters = result['parameters']  
    
    mode_names = ['car','drone','droid']
    flow       = mdrp.vec_to_mat(np.round(x,4))
    tot_demand = mdrp.tot_demand


    print('\n\nProblem Evlatuation')
    print('--------------'*4)
    print('Orders Demanded (order/hour): %.2f'%(tot_demand))
    print('--------------'*4)

    # Setup table
    print('Mode \t\t|', end="")
    for j in range(mdrp.m):
        print('%s\t|'%mode_names[j],end="")
    print('%s\t|'%'total',end="")
    print('\n----------------|',end="")
    for _ in range(mdrp.m+1):
        print('-------|',end="")

    # N
    print('\nN (# veh.)\t|',end="")
    for j in range(mdrp.m):
        print('%d\t|'%(mdrp.N_veh[j]),end="")

    # Orders
    orders_j = flow.sum(axis=0)/tot_demand # sum over orders per mode
    print('\nOrders (%)\t|',end="")
    for j in range(mdrp.m):
        print('%.2f\t|'%(orders_j[j]*100),end="")
    # tot_util = mat_x.sum(axis=0).sum(axis=0)/(mdrp.mu*mdrp.N).sum()
    print('%.2f\t|'%(orders_j*100).sum(),end="")

    # Utilization
    rho = mdrp.get_util(flow)
    print('\nUtilization (%)\t|',end="")
    for j in range(mdrp.m):
        print('%.2f\t|'%rho[j],end="")
    tot_util = flow.sum(axis=0).sum(axis=0)/(mdrp.mu*mdrp.N_veh).sum()
    print('%.2f\t|'%tot_util,end="")
    
    # Delivery Time
    lat = mdrp.get_latency(flow)*60
    print('\nD Time(min.)\t|',end="")
    for j in range(mdrp.m):
        lat_j = np.inner(lat[:,j],flow[:,j])/(orders_j[j]*tot_demand)
        print('%.2f\t|'%lat_j,end="")
    tot_lat = mdrp.objective(x)*60
    print('%.2f\t|'%tot_lat,end="")

    # Prices
    prices, min_price = get_prices(result)
    print('\nOrder Price\t|',end="")
    for j in range(mdrp.m):    # NOTE: REMOVE ROUNDING FOR FULL RESULT
        price_j = np.inner(prices[:,j],flow[:,j])/(orders_j[j]*tot_demand)
        print('%.2f\t|'%price_j,end="")
    tot_prices = np.inner(mdrp.mat_to_vec(prices),x)/(tot_demand)
    print('%.2f\t|'%tot_prices,end="")

    # Cost
    # NOTE: REMOVE ROUNDING FOR FULL RESULT
    print('\nOp. Cost ($/hr)\t|',end="")
    tot_cost = 0
    for j in range(mdrp.m):
        cost_j = mdrp.cost[j]*flow[:,j].sum()
        print('%.2f\t|'%(cost_j),end="")
        tot_cost += cost_j
    print('%.2f\t|'%tot_cost,end="")

    # Money Collected
    print('\nRevenue ($/hr)\t|',end="")
    tot_rev = 0
    for j in range(mdrp.m):
        rev_j = np.sum(prices[:,j]*flow[:,j])
        print('%.2f\t|'%(rev_j),end="")
        tot_rev += rev_j
    print('%.2f\t|'%tot_rev,end="")


    # Delivery Distance in meters (t_time in hours, speed in m/min)
    distance = (mdrp.t_time*60)*parameters['speeds']
    # print(pph)
    print('\nDistance\t|',end="")
    for j in range(mdrp.m):
        dist_j = np.inner(distance[:,j],flow[:,j])/(orders_j[j]*tot_demand*1000)
        # convert to miles
        dist_j *= 0.621371
        print('%.2f\t|'%dist_j,end="")
    tot_dist = np.inner(distance.flatten(),(flow).flatten())/(tot_demand*1000)
    # convert to miles
    tot_dist *= 0.621371
    print('%.2f\t|'%tot_dist,end="\n")


    print('--------------'*4)
    print('Minimum Price ($): %.2f'%(min_price))
    print('--------------'*4)

    return prices, min_price

    # print(mat_x[:,0])

def traveltime(origin_id,destination_id,meters_per_minute,locations):
    dist=np.sqrt((locations.at[destination_id,'x']-locations.at[origin_id,'x'])**2\
                +(locations.at[destination_id,'y']-locations.at[origin_id,'y'])**2)
    tt=np.ceil(dist/meters_per_minute)
    return tt

def read_instance_information(instance_dir):
    orders=pd.read_table(os.path.join(instance_dir,'orders.txt'))
    restaurants=pd.read_table(os.path.join(instance_dir,'restaurants.txt'))
    couriers=pd.read_table(os.path.join(instance_dir,'couriers.txt'))
    instanceparams=pd.read_table(os.path.join(instance_dir,'instance_parameters.txt'))

    order_locations=pd.DataFrame(data=[orders.order,orders.x,orders.y]).transpose()
    order_locations.columns=['id','x','y']
    restaurant_locations=pd.DataFrame(data=[restaurants.restaurant,restaurants.x,restaurants.y]).transpose()
    restaurant_locations.columns=['id','x','y']
    courier_locations=pd.DataFrame(data=[couriers.courier,couriers.x,couriers.y]).transpose()
    courier_locations.columns=['id','x','y']
    locations=pd.concat([order_locations,restaurant_locations,courier_locations])
    locations.set_index('id',inplace=True)

    orders.set_index('order',inplace=True)
    couriers.set_index('courier',inplace=True)
    restaurants.set_index('restaurant',inplace=True)

    meters_per_minute=instanceparams.at[0,'meters_per_minute']
    pickup_service_minutes=instanceparams.at[0,'pickup service minutes']
    dropoff_service_minutes=instanceparams.at[0,'dropoff service minutes']
    target_click_to_door=instanceparams.at[0,'target click-to-door']
    pay_per_order=instanceparams.at[0,'pay per order']
    guaranteed_pay_per_hour=instanceparams.at[0,'guaranteed pay per hour']
    return orders,restaurants,couriers,instanceparams,locations,meters_per_minute,\
           pickup_service_minutes,dropoff_service_minutes,target_click_to_door,\
           pay_per_order,guaranteed_pay_per_hour


