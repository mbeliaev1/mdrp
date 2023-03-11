import argparse
import json
# PATHING
import os
import sys
root = os.path.abspath(os.curdir)
sys.path.append(root)

import numpy as np
import cyipopt

from utils.convert import convert_instance
from utils.nlp_mdrp import mdrp_prob
from utils.helpers import *

def main(args):
    # Setup the directory and logger
    dir_list = ['%s'%'results', 
                '%s'%args.inst_path, 
                '%s'%args.demand]
    save_dir = ''
    for dirname in dir_list:
        save_dir += '%s/'%dirname
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

    sys.stdout = open(os.path.join(save_dir,'log.txt'), "w")
    print('Intiailzing')
    print('--------'*8)
    print(vars(args))
    print('--------'*8)
    setup_path = os.path.join(save_dir,'setup.json')
    with open(setup_path, 'w') as f:
        json.dump(vars(args),f,indent=4)    
    
    # Load the instance
    parameters = {}
    parameters['speeds'] = [1,2,0.3]
    parameters['N_ratios'] = [1,1,1]
    parameters['u_ratios'] = [1,0.2,0.2]

    isinstance_path = os.path.join('instances/',args.inst_path)
    prob_setup = convert_instance(isinstance_path,parameters)
    # add additional parameters
    prob_setup['max_rho'] = args.max_rho
    prob_setup['max_cost'] = args.max_cost
    prob_setup['demand'] = args.demand

    # print('Prob Setup')
    # print('--------'*8)
    # for param in prob_setup.keys():
    #     print('Parameter: %s \nValue:\n'%param,prob_setup[param])
    #     print('-----------------------------')
    # print('--------'*8)

    # SOLVE NLP
    mdrp = mdrp_prob(prob_setup)
    lb, ub, cl, cu = mdrp.get_bounds()
    x0 = mdrp.guess_init()

    nlp = cyipopt.Problem(
        n=len(x0),
        m=len(cl),
        problem_obj=mdrp,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
        )

    # nlp.addOption('derivative_test', 'second-order')
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('tol', 1e-4)

    #
    # Scale the problem (Just for demonstration purposes)
    #
    # nlp.set_problem_scaling(
    #     obj_scaling=1,
    #     x_scaling=[1, 1, 1, 1]
    #     )
    # nlp.add_option('nlp_scaling_method', 'user-scaling')

    #
    # Solve the problem
    #
    x, info = nlp.solve(x0)

    # print("Solution of the primal variables: x=%s\n" % repr(x))

    # print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))

    # print("Objective=%s\n" % repr(info['obj_val']))

    # Eval
    eval_setup = {}
    eval_setup['max_dph'] = args.max_dph
    eval_setup['min_dph'] = args.min_dph
    eval_setup['min_price'] = args.min_price
    eval_setup['mode_names'] = ['cars','drones','droids']

    eval_sol(x,mdrp,eval_setup)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--min_dph",
        type=float,
        default=5.0,
        help="mininum dollar/hour tradeoff for customers"
    )

    parser.add_argument(
        "--max_dph",
        type=float,
        default=100.0,
        help="maximum dollar/hour tradeoff for customers"
    )

    parser.add_argument(
        "--demand",
        type=float,
        default=0.99,
        help="Portion of maximum demand the system can take (b.w 0 and 1)"
    )

    parser.add_argument(
        "--max_rho",
        type=float,
        default=0.9,
        help="Maximum server utilization for all courier modalities"
    )

    parser.add_argument(
        "--max_cost",
        type=float,
        default=10000,
        help="Maximum system cost"
    )

    parser.add_argument(
        "--min_price",
        type=float,
        default=5,
        help="Minimum price per order"
    )

    parser.add_argument(
        "--inst_path",
        type=str,
        default='test',
        help="load_directory for instance path"
    )

    # parser.add_argument(
    #     "--out_dir",
    #     type=str,
    #     default='test',
    #     help="output directory for result folder"
    # )

    # parser.add_argument(
    #     "--no_adv",
    #     action = 'store_true',
    #     help="removes the adversarial training component, but still performs robust test with perturb"
    # )
    args = parser.parse_args()
    main(args)
