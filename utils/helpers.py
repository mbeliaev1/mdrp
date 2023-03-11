import numpy as np
from utils.nlp_mdrp import mdrp_prob
import pandas as pd

import os
import sys


def get_prices(x, mdrp,eval_setup):
    alpha = lambda a : (1/eval_setup['max_dph']) + ((1/eval_setup['min_dph'])-(1/eval_setup['max_dph']))*a 
    # dollar per hour-->
    # 5$ per hour -> 1/5 0.2
    # 100$ per hour -> 1/100 0.01

    mat_x = np.round(mdrp._vec_to_mat(x)/mdrp.demand,3)
    # mat_x = mdrp._vec_to_mat(x)
    prices = np.zeros_like(mat_x)
    lat = mdrp._get_latency(x)
    for i in range(mdrp.n_orders):
        j_sort = np.argsort(-lat[i])
        # j starts at cheapest option (longest latency)
        prices[i,j_sort[0]] = eval_setup['min_price']
        prev_j = j_sort[0]
        user_a = 1 - mat_x[i,j_sort[0]]
        for j in j_sort[1:]:
            prices[i,j] = prices[i,prev_j] + (lat[i,prev_j]-lat[i,j])/(alpha(user_a))
            user_a -= mat_x[i,j]
            prev_j = j

    return prices

def eval_sol(x,mdrp,eval_setup):
    '''
    Inputs
        x          - solution as vector outputted by mdrp
        mdrp       - nlp_mdrp instance for the problem setting
                     contains useful methods
        mode_names - list of strings defining the name of the modes
    '''

    mat_x = mdrp._vec_to_mat(x)
    print('Problem Evlatuation')
    print('--------------'*4)
    demand = mdrp.demand*mdrp.n_orders
    print('Orders Demanded (order/hour): %.2f'%(demand))
    print('Maximum Cost ($/hour): %.2f'%mdrp.max_cost)
    print('--------------'*4)
    # Setup table
    print('Mode \t\t\t|', end="")
    for j in range(mdrp.n_modes):
        print('%s\t|'%eval_setup['mode_names'][j],end="")
    print('%s\t|'%'total',end="")
    print('\n----------------|',end="")
    for _ in range(mdrp.n_modes+1):
        print('-------|',end="")
    # Orders
    orders_j = mat_x.sum(axis=0)/demand # sum over orders per mode
    print('\nOrders (%)\t\t|',end="")
    for j in range(mdrp.n_modes):
        print('%.2f\t|'%(orders_j[j]),end="")
    # tot_util = mat_x.sum(axis=0).sum(axis=0)/(mdrp.mu*mdrp.N).sum()
    print('%.2f\t|'%orders_j.sum(),end="")

    # Cost
    print('\nOp. Cost ($/hr)\t|',end="")
    for j in range(mdrp.n_modes):
        cost_j = np.inner(mdrp.cost[:,j],mat_x[:,j])
        print('%.2f\t|'%(cost_j),end="")
    tot_cost = np.inner(mdrp._mat_to_vec(mdrp.cost),x)
    print('%.2f\t|'%tot_cost,end="")

    # Utilization
    util = mdrp._get_util(x)
    print('\nUtilization (%)\t|',end="")
    for j in range(mdrp.n_modes):
        print('%.2f\t|'%util[j],end="")
    tot_util = mat_x.sum(axis=0).sum(axis=0)/(mdrp.mu*mdrp.N).sum()
    print('%.2f\t|'%tot_util,end="")
    
    # Delivery Time
    lat = mdrp._get_latency(x)*60
    print('\nD Time(min.)\t|',end="")
    for j in range(mdrp.n_modes):
        lat_j = np.inner(lat[:,j],mat_x[:,j])/(orders_j[j]*demand)
        print('%.2f\t|'%lat_j,end="")
    tot_lat = mdrp.objective(x)*60
    print('%.2f\t|'%tot_lat,end="")

    # Prices
    prices = get_prices(x, mdrp, eval_setup)
    print('\nOrder Price\t\t|',end="")
    for j in range(mdrp.n_modes):
        price_j = np.inner(prices[:,j],mat_x[:,j])/(orders_j[j]*demand)
        print('%.2f\t|'%price_j,end="")
    tot_prices = np.inner(prices.flatten(),x)/(demand)
    print('%.2f\t|'%tot_prices,end="")
    # print("")
    # print(np.round(mat_x,2))
    # print(prices)

    # Price per hour
    pph = prices/(lat/60)
    # print(pph)
    print('\nPrice per hour\t|',end="")
    for j in range(mdrp.n_modes):
        price_j = np.inner(pph[:,j],mat_x[:,j])/(orders_j[j]*demand)
        print('%.2f\t|'%price_j,end="")
    tot_prices = np.inner(pph.flatten(),x)/(demand)
    print('%.2f\t|'%tot_prices,end="")

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