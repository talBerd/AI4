
from typing import Optional, Generator, Tuple, Any, Union
from mdp import MDP, Action
class Simulator():
    def __init__(self, output_file: str='simulator_results.txt'):
        self.output_file = output_file
        
    def replay(self, num_episodes: Optional[int] = None, result_file: Optional[str] = None):
        '''
        Returns a nested generator object, the first one generates episodes, the second one generates (s, r, a, a_actual) tuples.
        :param num_episodes: number of episodes to replay. If None, all episodes are replayed. If the number of episodes in the file is less than num_episodes, all episodes are replayed.
        '''
        if result_file is None:
            result_file = self.output_file
        
        with open(result_file) as f:
            for i, episode in enumerate(f):
                if num_episodes is not None and i >= num_episodes:
                    break
                yield self.step_generator(episode.strip())

    def step_generator(self, line: str) -> Generator[Tuple[Any, float, Optional[Action], Optional[Action]], None, None]:
        '''
        Generates (state, reward, action, actual_action) tuples from a single line of the result file.
        NOTE: For the last step, the actions are None, None.
        '''
        elements = line.split(';')
        state_str = elements[0]
        state = tuple(map(int, state_str.strip('()').split(',')))
        
        for i in range(1, len(elements), 4):
            reward = float(elements[i])
            if i + 3 >= len(elements):
                # Last step, return just state and reward
                yield state, reward, None, None
                break

            action = Action[elements[i + 1]]
            actual_action = Action[elements[i + 2]]
            next_state_str = elements[i + 3]
            next_state = tuple(map(int, next_state_str.strip('()').split(',')))
            yield state, reward, action, actual_action
            state = next_state


if __name__ == '__main__':
    #example usage
    sim = Simulator()
    
    for episode_index, episode_gen in enumerate(sim.replay(num_episodes=1)):
        print(f"@@@@    episode {episode_index}   @@@@@")
        for step_index, step in enumerate(episode_gen):
            state, reward, action, actual_action = step
            print(f"Step {step_index}: state={state}, reward={reward}, action={action}, actual_action={actual_action}")

  
  
  