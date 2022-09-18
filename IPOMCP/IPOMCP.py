from enviroment.nodes import *
from enviroment.abstract_classes import *
import matplotlib.pyplot as plt
from ipomcp_config import get_config


class IPOMCP:

    def __init__(self,
                 root_sampling,
                 environment_simulator,
                 exploration_policy,
                 reward_function,
                 seed: int):
        """

        :param root_sampling: Generative model for sampling root IS particles
        :param environment_simulator: Generative model for simulating environment dynamics
        :param exploration_policy: An exploration policy class
        :param reward_function: reward function for computation of accept value
        """
        self.root_sampling = root_sampling
        self.environment_simulator = environment_simulator
        self.action_exploration_policy = exploration_policy
        self.reward_function = reward_function
        self.config = get_config()
        self.seed = seed
        self.tree = dict()
        self.history_node = None
        self.exploration_bonus = float(self.config.get_from_env("exploration_bonus"))
        self.depth = float(self.config.get_from_env("planning_depth"))
        self.n_iterations = int(self.config.get_from_env("mcts_number_of_iterations"))
        self.softmax_temperature = float(self.config.softmax_temperature)
        self.pruning_epsilon = 0.01

    def plan(self, offer, counter_offer,
             first_move=False):
        """
        
        :param first_move:
        :param offer:  action_{t-1}
        :param counter_offer: observation_{t}
        :return: action_node
        """
        previous_counter_offer = self.root_sampling.history[-2]
        current_history_length = len(self.root_sampling.history)
        base_node = HistoryNode(None, Action(previous_counter_offer), self.action_exploration_policy)
        offer_node = base_node.add_action_node(Action(offer))
        self.history_node = offer_node.add_history_node(Action(counter_offer), self.action_exploration_policy)
        self.root_sampling.update_distribution(Action(offer), Action(counter_offer), first_move)
        root_samples = self.root_sampling.sample(self.seed, n_samples=self.n_iterations)
        for i in range(self.n_iterations):
            persona = root_samples[i]
            self.environment_simulator.reset_persona(persona, current_history_length)
            nested_belief = self.environment_simulator.opponent_model.belief_distribution.get_belief()
            interactive_state = InteractiveState(None, persona, nested_belief)
            self.history_node.particles.append(interactive_state)
            self.simulate(i, interactive_state, self.history_node, self.depth, self.seed, True)
        return self.history_node.children, \
               np.c_[self.history_node.children_qvalues, self.history_node.children_visited[:, 1]]

    def simulate(self, trail_number, interactive_state: InteractiveState,
                 history_node: HistoryNode, depth,
                 seed: int, tree: bool):
        if depth <= 0:
            return 0.0, True
        history_node.compute_deterministic_actions_reward(self.reward_function)
        action_node = history_node.select_action(interactive_state,
                                                 history_node.parent.action,
                                                 history_node.observation,
                                                 tree)
        action_node.append_particle(interactive_state)
        # If the selected action is terminal
        if action_node.action.is_terminal:
            history_node.increment_visited()
            action_node.increment_visited()
            return self._halting_action_reward(action_node.action, history_node.observation.value), True
        new_interactive_state, observation, q_value, reward, log_prob = \
            self.environment_simulator.act(interactive_state,
                                           action_node.action,
                                           history_node.observation, seed)
        new_observation_flag = True
        if str(observation.value) in action_node.children:
            new_observation_flag = False
            new_history_node = action_node.children[str(observation.value)]
        else:
            new_history_node = action_node.add_history_node(observation, self.action_exploration_policy,
                                                            is_terminal=observation.is_terminal)
        new_history_node.particles.append(interactive_state)
        if observation.is_terminal:
            history_node.increment_visited()
            action_node.increment_visited()
            action_node.update_q_value(reward)
            return reward, observation.is_terminal

        if new_observation_flag:
            action_node.children[str(new_history_node.observation)] = new_history_node
            future_reward, is_terminal = self.simulate(trail_number, new_interactive_state, new_history_node, depth - 1, seed, False)
            total = reward + future_reward
        else:
            future_reward, is_terminal = self.simulate(trail_number, new_interactive_state, new_history_node, depth - 1,
                                                       seed, True)
            total = reward + future_reward
        # TODO(Nitay) - warp up the method
        history_node.increment_visited()
        action_node.increment_visited()
        action_node.update_q_value(total)
        return total, observation.is_terminal

    def rollout(self, interactive_state: InteractiveState, history_node: HistoryNode, depth):
        if depth <= 0:
            return 0.0
        history_node.compute_deterministic_actions_reward(self.reward_function)
        action_node = history_node.rollout_policy(interactive_state, history_node.parent.action,
                                                  history_node.observation, int(depth))
        history_node.children[str(action_node.action)] = action_node
        action_node.append_particle(interactive_state)

        if action_node.action.is_terminal:
            return self._halting_action_reward(action_node.action, history_node.observation)

        new_interactive_state, observation, q_value, reward, log_prob = \
            self.environment_simulator.act(interactive_state,
                                           history_node.previous_observation,
                                           action_node.action,
                                           history_node.observation)

        new_history_node = action_node.add_history_node(observation, self.action_exploration_policy,
                                                        observation.is_terminal)
        if observation.is_terminal:
            history_node.increment_visited()
            action_node.increment_visited()
            action_node.update_q_value(reward)
            return reward, observation.is_terminal

        future_reward = self.rollout(new_interactive_state, new_history_node, depth - 1)
        total = reward + future_reward
        return total

    def prune_tree(self, action_node: ActionNode, observation):  # TODO should we add dispose?
        nearest_observation = action_node.children.get(observation) or action_node.children[
            min(action_node.children.keys(), key=lambda k: abs(k - observation))]
        if abs(nearest_observation - observation) < self.pruning_epsilon:
            return action_node.children[nearest_observation]
        return None

    def q_values_hist(self):
        x = np.array([k for k, c in self.history_node.children.items()])
        y = np.array([c.q_value for k, c in self.history_node.children.items()])
        plt.hist(y)
        plt.show()

    def _add_nodes(self, tree_node, node: TreeNode):
        if not node:
            return

        for k, c in node.children.items():
            new_tree_node = tree_node.add_child(name=str(c))
            self._add_nodes(new_tree_node, c)

    def plot_q_values(self, iteration_number):
        plt.scatter(np.array(self.history_node.children_values).astype(float)[2:], self.history_node.children_qvalues[2:])
        plt.title(f'Iteration number {iteration_number}')
        plt.savefig(f'{iteration_number}.png', bbox_inches='tight')

    def _halting_action_reward(self, action, observation):
        reward = 0.0
        if action.value == -2:
            reward = self.reward_function(observation)
        return reward
