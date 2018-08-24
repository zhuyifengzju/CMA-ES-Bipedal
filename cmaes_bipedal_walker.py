import numpy as np
import gym
import cma
from copy import copy

import multiprocessing as mp
from itertools import product

import argparse

import pickle

N_HIDDEN_1 = 100

env = gym.make('BipedalWalker-v2')
state = env.reset()
np.random.seed(0)
state_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]

def parse_args():
    parser = argparse.ArgumentParser(description='CMA-ES on BipedalWalker')

    parser.add_argument(
        '-n',
        '--iterations',
        default=1000,
        type=int,
        help='Number of training iterations')
    parser.add_argument(
        '--render',
        action='store_true',
        help='Display rendering while training'
    )

    # CMA-ES hyperparameters
    parser.add_argument(
        '--seed',
        default=0,
        type=int,
        help='Random seed',
    )
    parser.add_argument(
        '--popsize',
        default=16,
        type=int,
        help='Population size',
    )


    return parser.parse_args()
    
    
    
def reshape(theta, state_shape, action_shape):

    w1_length = state_shape * N_HIDDEN_1
    b1_length = N_HIDDEN_1
    w2_length = N_HIDDEN_1 * action_shape
    b2_length = action_shape
    
    w1_theta = np.copy(theta[0:w1_length])
    b1_theta = np.copy(theta[w1_length:w1_length+b1_length])
    w2_theta = np.copy(theta[w1_length+b1_length:w1_length+b1_length+w2_length])
    b2_theta = np.copy(theta[-b2_length:].copy())
    
    w1_theta = np.reshape(w1_theta, (state_shape, N_HIDDEN_1))
    b1_theta = np.reshape(b1_theta, (N_HIDDEN_1,))
    w2_theta = np.reshape(w2_theta, (N_HIDDEN_1, action_shape))
    b2_theta = np.reshape(b2_theta, (action_shape,))
    return (w1_theta, b1_theta, w2_theta, b2_theta)

def f(theta, render=False):

    theta = reshape(np.copy(theta), state_shape, action_shape)
    gamma = 1

    N_TRIALS = 10
    total_reward = 0
    for i in range(N_TRIALS):
        is_terminal = False

        cur_state = env.reset()   
        steps = 0
        episode_reward = 0
        while not is_terminal:
            cur_state = np.squeeze(cur_state)

            h1 = np.matmul(np.expand_dims(cur_state, axis=0), theta[0]) + theta[1]
            h2 = np.tanh(np.matmul(h1, theta[2]) + theta[3])
            action = np.squeeze(h2)

            next_state, reward, is_terminal, _ = env.step(action)
            
            if render is True and i==0:
                env.render()

            episode_reward = reward + gamma * episode_reward                
            cur_state = next_state
            steps += 1

        
        total_reward += episode_reward

    # CMA minimizes function value, so negative sign is needed
    return -total_reward / N_TRIALS
    

def main():

    args = parse_args()
    
    N_THETA = state_shape * N_HIDDEN_1 + N_HIDDEN_1 + N_HIDDEN_1 * action_shape + action_shape
    
    es = cma.CMAEvolutionStrategy(N_THETA * [0], 0.1, {'popsize': args.popsize, 'seed': args.seed})
    
    
    for i in range(args.iterations):
        solutions = es.ask()


        with mp.Pool(mp.cpu_count()) as pool:
            rewards = pool.starmap(f, product(solutions))

        es.tell(solutions, rewards)

        # Rendering 3 policies every 50 iterations
        if i % 10 == 0:
            if args.render is True:
                [f(solution, render=True) for solution in solutions[0:3]]
            dict = {
                'iteration': i,
                'mean': es.mean,
                'sigma': es.sigma,
                'popsize': args.popsize,
                'seed': args.seed
            }
            np.save(f'Weights/Iteration-{i}.npy', dict)
            
        es.disp()


    pickle.dump(es, open(f'Weights/Final-saved-es-object.pkl', 'wb'))
    
    env.close()

if __name__ == '__main__':
    main()
