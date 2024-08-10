from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np
from copy import deepcopy
from collections import defaultdict


def value_iteration(mdp: MDP, U_init: np.ndarray, epsilon: float=10 ** (-3)) -> np.ndarray:
    correct_U = deepcopy(U_init)
    delta = float('inf')  

    while delta >= epsilon * (1 - mdp.gamma) / mdp.gamma:
        U_update = deepcopy(correct_U)
        delta = 0

        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                
                reward = mdp.board[row][col]
                
                if reward == "WALL":
                    continue

                if (row, col) in mdp.terminal_states:
                    U_update[row][col] = float(reward)
                    
                else:
                    utilities = []
                    for action in mdp.actions:
                        #Evaluate utility considering all possible results from an action
                        next_state = mdp.step((row, col), action)
                        utility = sum(
                            mdp.transition_function[action][i] * correct_U[next_state[0]][next_state[1]]
                            for i, direction in enumerate(mdp.actions)
                        )
                        utilities.append(utility)

                    U_update[row][col] = float(reward) + mdp.gamma * max(utilities)

                if abs(U_update[row][col] - correct_U[row][col]) > delta:
                    delta = abs(U_update[row][col] - correct_U[row][col])

        correct_U = U_update

    return correct_U



def get_policy(mdp: MDP, U: np.ndarray) -> np.ndarray:

    policy = [[None for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]

    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            
            reward = mdp.board[row][col]

            if reward == "WALL" or (row, col) in mdp.terminal_states:
                policy[row][col] = None
                
            else:
                best_action = None
                max_utility = float('-inf')
                
                for action in mdp.actions:
                    #Evaluate utility considering all possible results from an action
                    next_state = mdp.step((row, col), action)
                    expected_utility = sum(
                        mdp.transition_function[action][i] * U[next_state[0]][next_state[1]]
                        for i in range(4)
                    )
                    #Find best utility and action 
                    if expected_utility > max_utility:
                        max_utility = expected_utility
                        best_action = action

                policy[row][col] = str(best_action)
    
    return policy


def policy_evaluation(mdp: MDP, policy: np.ndarray) -> np.ndarray:

    correct_U = np.zeros((mdp.num_row, mdp.num_col))
    c=0
    while True:
        delta = 0
        U_update = np.copy(correct_U)
        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                
                reward = mdp.board[row][col]

                if reward == "WALL" or (row, col) in mdp.terminal_states:
                    continue
                
                #Evaluate utility for the current state according to the given policy
                step = Action(policy[row][col])  
                next_state = mdp.step((row, col), step)
                
                expected_utility = sum(
                    mdp.transition_function[step][i] * correct_U[next_state[0]][next_state[1]]
                    for i in range(4)
                )
                U_update[row][col] = float(reward) + mdp.gamma * expected_utility

                delta = max(delta, abs(U_update[row][col] - correct_U[row][col]))

        
        correct_U = U_update
        #Stop when the numbers don't change anymore
        if delta == 0:  
            break
    
    return correct_U.tolist()


def policy_iteration(mdp: MDP, policy_init: np.ndarray) -> np.ndarray:
    policy = deepcopy(policy_init)
    stable = False
    
    #Run until there is no change in optimal policy
    while not stable:
        correct_U = policy_evaluation(mdp, policy)
        stable = True

        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                
                if mdp.board[row][col] == "WALL" or (row, col) in mdp.terminal_states:
                    policy[row][col] = None
                    continue

                current_action = Action[policy[row][col]] if policy[row][col] else None
                best_action = None
                max_value = float('-inf')
                
                #Find optimal action according to policy evaluation
                for action in Action:
                    total = 0
                    for i, direction in enumerate(Action):
                        next_state = mdp.step((row, col), direction)
                        probability = mdp.transition_function[action][i]  
                        total += probability * correct_U[next_state[0]][next_state[1]]

                    if total > max_value:
                        max_value = total
                        best_action = action

                if best_action and best_action != current_action:
                    policy[row][col] = str(best_action)  
                    stable = False

    return policy

def adp_algorithm(
    sim: Simulator, 
    num_episodes: int,
    num_rows: int = 3, 
    num_cols: int = 4, 
    actions: List[Action] = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT] 
) -> Tuple[np.ndarray, Dict[Action, Dict[Action, float]]]:
    """
    Runs the ADP algorithm given the simulator, the number of rows and columns in the grid, 
    the list of actions, and the number of episodes.

    :param sim: The simulator instance.
    :param num_rows: Number of rows in the grid (default is 3).
    :param num_cols: Number of columns in the grid (default is 4).
    :param actions: List of possible actions (default is [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]).
    :param num_episodes: Number of episodes to run the simulation (default is 10).
    :return: A tuple containing the reward matrix and the transition probabilities.
    
    NOTE: the transition probabilities should be represented as a dictionary of dictionaries, so that given a desired action (the first key),
    its nested dicionary will contain the condional probabilites of all the actions. 
    """
    
    # Initialize transition counters with zeros for all (action, actual_action) pairs
    transition_counts = {action: {act: 0 for act in actions} for action in actions}
    rewards = np.zeros((num_rows, num_cols, len(actions)))  
    
    for episode_gen in sim.replay(num_episodes):
        for state, reward, action, actual_action in episode_gen:
            if action is not None:
                row, col = state
                action_index = list(Action).index(action)
                
                #Fill the reward matrix
                rewards[row, col, action_index] = reward  
                
            if action is not None and actual_action is not None:
                transition_counts[action][actual_action] += 1

    #Calculate transition probabilities
    transition_probabilities = {action: {} for action in actions}
    for action in actions:
        total_action_count = sum(transition_counts[action].values())
        if total_action_count > 0:
            for actual_action in actions:
                transition_probabilities[action][actual_action] = transition_counts[action][actual_action] / total_action_count
        else:
            for actual_action in actions:
                transition_probabilities[action][actual_action] = 0.0

    return rewards, transition_probabilities
