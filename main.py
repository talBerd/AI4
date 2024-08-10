
from mdp_rl_implementation import value_iteration, get_policy, policy_evaluation, policy_iteration, adp_algorithm
from mdp import MDP
from simulator import Simulator

def example_driver():
    """
    This is an example of a driver function, after implementing the functions
    in "mdp_rl_implementation.py" you will be able to run this code with no errors.
    """

    mdp = MDP.load_mdp()

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@ The board and rewards @@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    
    mdp.print_rewards()

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@@@@ Value iteration @@@@@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    U = [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]

    print("\nInitial utility:")
    mdp.print_utility(U)
    print("\nFinal utility:")
    U_new = value_iteration(mdp, U)
    mdp.print_utility(U_new)
    print("\nFinal policy:")
    policy = get_policy(mdp, U_new)
    mdp.print_policy(policy)

    
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@@@@ Policy iteration @@@@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    print("\nPolicy evaluation:")
    U_eval = policy_evaluation(mdp, policy)
    mdp.print_utility(U_eval)

    policy = [['UP', 'UP', 'UP', 0],
              ['UP', 'WALL', 'UP', 0],
              ['UP', 'UP', 'UP', 'UP']]

    print("\nInitial policy:")
    mdp.print_policy(policy)
    print("\nFinal policy:")
    policy_new = policy_iteration(mdp, policy)
    mdp.print_policy(policy_new)

    print("Done!")
    


def adp_example_driver():

    sim = Simulator()

    
    reward_matrix, transition_probabilities = adp_algorithm(sim,num_episodes=10)
  
    print("Reward Matrix:")
    print(reward_matrix)
    
    print("\n Transition Probabilities:")
    for action, probs in transition_probabilities.items():
        print(f"{action}: {probs}")

    
if __name__ == '__main__':
    # run our example
    example_driver()
    adp_example_driver()
