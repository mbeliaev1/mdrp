import numpy as np
from utils.nlp_mdrp import mdrp_prob
import pandas as pd

import os
import sys


def alpha_func(alpha_0, delta, x):
    return alpha_0 + delta*x

def guess_price(alpha_0, delta, lat, flow, tau_min):

    prices = np.zeros_like(flow)
    m = prices.shape[1]
    n = prices.shape[0]

    # set min prices
    prices[:,m-1] = tau_min
    for i in range(n):
        user_a = flow[i,:].sum() - flow[i,m-1]

        for j in range(m-2,-1,-1):
            left = (lat[i,j+1]-lat[i,j])/alpha_func(alpha_0[i,j],delta[i,j],user_a)
            right = (alpha_func(alpha_0[i,j+1],delta[i,j+1],user_a)/alpha_func(alpha_0[i,j],delta[i,j],user_a))*prices[i,j+1]
            prices[i,j] = left + right
            user_a -= flow[i,j]
    revenue = np.inner(prices.flatten(),flow.flatten())
    return prices, revenue


def get_prices(result):
    mdrp    = result['mdrp']
    x       = result['sol']

    flow       = mdrp.vec_to_mat(np.round(x,4))
    tot_demand = mdrp.tot_demand

    # initialize
    prices = np.zeros_like(flow)
    lat = mdrp.get_latency(flow)
    op_cost = np.inner(mdrp.N_veh,mdrp.cost)

    # find slope & intercept
    _, rev_0 = guess_price(mdrp.alpha_0, mdrp.delta, lat, flow, 0)
    _, rev_1 = guess_price(mdrp.alpha_0, mdrp.delta, lat, flow, 1)
    # find min price
    # NOTE: can be vector or scalar
    min_users = flow.sum(axis=1) - flow[:,-1]/2
    min_price = (1/alpha_func(mdrp.alpha_0[:,-1], mdrp.delta[:,-1],min_users))*lat[:,-1]
    # solve
    prices, _ = guess_price(mdrp.alpha_0, mdrp.delta, lat, flow, min_price)
    # is_nash = check_nash(mdrp.alpha_0, mdrp.delta, lat, flow, prices)
    # print('is nash:',is_nash)
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
    
    mode_names = ['UAT','AV','T']
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
    for j in range(mdrp.m):
        price_j = np.inner(prices[:,j],flow[:,j])/(orders_j[j]*tot_demand)
        print('%.2f\t|'%price_j,end="")
    tot_prices = np.inner(mdrp.mat_to_vec(prices),x)/(tot_demand)
    print('%.2f\t|'%tot_prices,end="")

    # Cost
    # NOTE: REMOVE ROUNDING FOR FULL RESULT
    print('\nOp. Cost ($/hr)\t|',end="")
    tot_cost = 0
    for j in range(mdrp.m):
        cost_j = mdrp.cost[j]*mdrp.N_veh[j]
        print('%.0f\t|'%(cost_j),end="")
        tot_cost += cost_j
    print('%.0f\t|'%tot_cost,end="")

    # Profit COllected
    # NOTE: REMOVE ROUNDING FOR FULL RESULT
    print('\nProfit ($/hr)\t|',end="")
    tot_profit = 0
    for j in range(mdrp.m):
        cost_j = mdrp.cost[j]*mdrp.N_veh[j]
        rev_j = np.sum(prices[:,j]*flow[:,j])
        profit_j = rev_j - cost_j
        print('%.0f\t|'%(profit_j),end="")
        tot_profit += profit_j
    print('%.0f\t|'%tot_profit,end="")
    # print("")
    # print(np.round(mat_x,2))
    # print(prices)

    # Delivery Distance in meters (t_time in hours, speed in m/min)
    # distance = (mdrp.t_time*60)*parameters['speeds']
    # # print(pph)
    # print('\nDistance\t|',end="")
    # for j in range(mdrp.m):
    #     dist_j = np.inner(distance[:,j],flow[:,j])/(orders_j[j]*tot_demand*1000)
    #     print('%.2f\t|'%dist_j,end="")
    # tot_dist = np.inner(distance.flatten(),(flow*parameters['speeds']).flatten())/(tot_demand*1000)
    # print('%.2f\t|'%tot_dist,end="\n")


    print('--------------'*4)
    print('Minimum Price ($): %.2f'%(min_price.mean()))
    print('--------------'*4)

    return prices, min_price

    # print(mat_x[:,0])

