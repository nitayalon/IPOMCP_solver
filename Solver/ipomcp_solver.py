from IPOMCP_solver.Solver.nodes import *
from IPOMCP_solver.Solver.abstract_classes import *
from IPOMCP_solver.Solver.ipomcp_config import get_config
from IPOMCP_solver.utils.memoization_table import MemoizationTable
import time


class IPOMCP:

    def __init__(self,
                 agent_dom_level: int,
                 root_sampling: BeliefDistribution,
                 environment_simulator: EnvironmentModel,
                 memoization_table: Optional[MemoizationTable],
                 exploration_policy,
                 reward_function,
                 planning_parameters: dict,
                 seed: int):
        """

        :param root_sampling: Generative model for sampling root IS particles
        :param environment_simulator: Generative model for simulating environment dynamics
        :param memoization_table: Q-values memoization data for querying
        :param exploration_policy: An exploration policy class
        :param reward_function: reward function for computation of accept value
        :param planning_parameters: dict, additional parameters for planning
        """
        self.root_sampling = root_sampling
        self.environment_simulator = environment_simulator
        self.memoization_table = memoization_table
        self.action_exploration_policy = exploration_policy
        self.reward_function = reward_function
        self.planning_parameters = planning_parameters
        self.seed = seed
        self.config = get_config()
        self.tree = dict()
        self.history_node = None
        self.action_node = None
        self.exploration_bonus = float(self.config.get_from_env("uct_exploration_bonus"))
        self.depth = float(self.config.get_from_env("planning_depth"))
        self.n_iterations = self.compute_number_of_planning_iterations(
            int(self.config.get_from_env("mcts_number_of_iterations")), agent_dom_level)
        self.discount_factor = float(self.config.get_from_env("discount_factor"))
        self.softmax_temperature = float(self.config.softmax_temperature)
        self.name = "IPOMCP"
        self.x_ipomdp_model = False

    @staticmethod
    def compute_number_of_planning_iterations(number_of_iterations, agent_dom_level):
        if agent_dom_level == 2:
            return number_of_iterations // 2
        return number_of_iterations

    def reset(self, iteration_number: int):
        self.history_node = None
        self.action_node = None

    def plan(self, offer: Action, counter_offer: Action,
             iteration_number, update_belief):
        """
        
        :param iteration_number:
        :param offer:  action_{t-1}
        :param counter_offer: observation_{t}
        :return: action_node
        :param update_belief:
        """
        action_length = len(self.root_sampling.history.actions)
        observation_length = len(self.root_sampling.history.observations)
        current_opponent_persona = self.environment_simulator.get_persona()
        if self.action_node is None or str(counter_offer) not in self.action_node.children:
            previous_counter_offer = self.root_sampling.history.get_last_observation()
            base_node = HistoryNode(None, previous_counter_offer, 1.0, self.action_exploration_policy)
            offer_node = base_node.add_action_node(offer)
            self.history_node = offer_node.add_history_node(counter_offer, 1.0, self.action_exploration_policy)
            self.root_sampling.update_distribution(offer, counter_offer, iteration_number)
        else:
            self.history_node = self.action_node.children[str(counter_offer)]
            self.root_sampling.update_distribution_from_particles(self.history_node.particles, offer, counter_offer,
                                                                  iteration_number)
        root_samples = self.root_sampling.sample(self.seed, n_samples=self.n_iterations)
        # Check if we already have Q-values for this setting:
        query_parameters = {'trial': iteration_number,
                            'threshold': self.planning_parameters['threshold'],
                            'belief': self.root_sampling.belief_distribution}
        q_values = self.memoization_table.query_table(query_parameters)
        if not q_values.empty:
            return self.history_node.children, None, np.c_[q_values, np.repeat(10, q_values.shape[0])]
        print(f'Empty Q-value data, iteration: {iteration_number}')
        self.action_exploration_policy.update_belief(self.root_sampling.belief_distribution)
        iteration_times = []
        depth_statistics = []
        for i in range(self.n_iterations):
            persona = Persona(root_samples[i], None)
            self.environment_simulator.reset_persona(persona, action_length, observation_length,
                                                     self.root_sampling.opponent_model.belief.belief_distribution,
                                                     iteration_number)
            nested_belief = self.environment_simulator.opponent_model.belief.get_current_belief()
            interactive_state = InteractiveState(State(str(iteration_number), False), persona, nested_belief)
            start_time = time.time()
            _, _, depth = self.simulate(i, interactive_state, self.history_node, 0, self.seed, iteration_number)
            end_time = time.time()
            iteration_time = end_time - start_time
            iteration_times.append([persona, iteration_time])
            depth_statistics.append([persona, depth])
        self.environment_simulator.reset_persona(current_opponent_persona, action_length, observation_length,
                                                 self.root_sampling.opponent_model.belief.belief_distribution,
                                                 iteration_number)
        # Reporting iteration time
        if self.config.report_ipocmp_statistics:
            iteration_time_for_logging = pd.DataFrame(iteration_times)
            iteration_time_for_logging.columns = ["persona", "time"]
        if self.config.output_planning_tree:
            optimal_tree, optimal_tree_beliefs = self.extract_max_q_value_trajectory(self.history_node)
            optimal_tree_table = pd.DataFrame(optimal_tree, columns=['node_type', 'parent_id', 'self_id', 'parent_value',
                                                                     'self_value', 'probability', 'q_value'])
        else:
            optimal_tree_table = None
        # update buffer with new information
        self.memoization_table.update_table(self.history_node.children_qvalues,
                                            np.array([offer.value, counter_offer.value]),
                                            query_parameters['belief'],
                                            game_parameters={'trial': query_parameters['trial'],
                                                             'seed': self.planning_parameters['seed'],
                                                             'threshold': query_parameters['threshold']})
        return self.history_node.children, optimal_tree_table, \
               np.c_[self.history_node.children_qvalues, self.history_node.children_visited[:, 1]]

    def simulate(self, trail_number, interactive_state: InteractiveState,
                 history_node: HistoryNode, depth,
                 seed: int, iteration_number):
        action_node = history_node.select_action(interactive_state,
                                                 history_node.parent.action,
                                                 history_node.observation,
                                                 True, iteration_number)

        if depth >= self.depth or iteration_number >= self.config.task_duration:
            reward = self.environment_simulator.reward_function(history_node.observation.value,
                                                                action_node.action.value)
            return reward, True, depth

        # If the selected action is terminal
        if action_node.action.is_terminal:
            history_node.increment_visited()
            action_node.increment_visited()
            return self._halting_action_reward(action_node.action, history_node.observation.value), True, depth

        # Since the nested beliefs are deterministic given the action:
        # If we already expanded this action in this history we resample from interactive state:
        if str(interactive_state) in action_node.particles.keys():
            new_interactive_state = action_node.particles[str(interactive_state)]
            observation, reward, observation_probability = \
                self.environment_simulator.step_from_is(new_interactive_state, history_node.observation,
                                                        action_node.action, seed)
        else:
            new_interactive_state, observation, reward, observation_probability = \
                self.environment_simulator.step(history_node,
                                                action_node,
                                                interactive_state,
                                                seed, self.environment_simulator.compute_iteration(iteration_number))
            action_node.append_particle(new_interactive_state)
        # Update reward for interactive state: R(is)
        history_node.update_reward(action_node.action, reward, observation_probability)
        new_observation_flag = True
        if str(observation.value) in action_node.children:
            new_observation_flag = False
            new_history_node = action_node.children[str(observation.value)]
        else:
            new_history_node = action_node.add_history_node(observation, observation_probability,
                                                            self.action_exploration_policy,
                                                            is_terminal=observation.is_terminal)
        # Append the new interactive state to the history node for PF
        new_history_node.append_particle(new_interactive_state, observation_probability)
        if observation.is_terminal:
            history_node.increment_visited()
            action_node.increment_visited()
            new_history_node.increment_visited()
            action_node.update_q_value(reward)
            return reward, observation.is_terminal, depth

        if new_observation_flag:
            action_node.children[str(new_history_node.observation)] = new_history_node
            if interactive_state.persona.persona[1]:
                future_reward = 0.0
            else:
                future_reward, is_terminal, depth = self.rollout(trail_number, new_interactive_state,
                                                                 action_node.action, observation, depth + 1,
                                                                 seed, iteration_number + 1)
        else:
            if interactive_state.persona.persona[1]:
                future_reward = 0.0
            else:
                future_reward, is_terminal, depth = self.simulate(trail_number, new_interactive_state, new_history_node,
                                                                  depth + 1, seed, iteration_number + 1)
        total = reward + self.discount_factor * future_reward
        history_node.increment_visited()
        action_node.increment_visited()
        action_node.update_q_value(total)
        return total, observation.is_terminal, depth

    def rollout(self, trail_number, interactive_state: InteractiveState, last_action: Action, observation: Action,
                depth, seed: int,
                iteration_number) -> [float, bool, int]:
        if depth >= self.depth or iteration_number >= self.config.task_duration:
            future_value = self.environment_simulator.compute_future_values(observation.value, last_action.value,
                                                                            iteration_number, self.config.task_duration)
            return future_value, True, depth
        action, _ = self.action_exploration_policy.sample(interactive_state,
                                                          last_action.value, observation.value,
                                                          iteration_number)
        if action.is_terminal:
            reward = self._halting_action_reward(action, observation.value)
            return reward, True, depth
        new_interactive_state, observation, reward, observation_probability = \
            self.environment_simulator.rollout_step(interactive_state, action, observation, seed, iteration_number + 1,
                                                    last_action)
        if observation.is_terminal:
            return reward, observation.is_terminal, depth
        else:
            if interactive_state.persona.persona[1]:
                future_reward = 0.0
            else:
                future_reward, is_terminal, depth = self.rollout(trail_number, new_interactive_state, action, observation,
                                                                 depth + 1, seed, iteration_number + 1)
        total = reward + self.discount_factor * future_reward
        return total, observation.is_terminal, depth

    def _halting_action_reward(self, action, observation):
        reward = 0.0
        if action.value == -2:
            reward = self.reward_function(observation)
        return reward

    def _compute_terminal_tree_reward(self, persona, nested_belief):
        average_nested_persona = np.sum(nested_belief[:, 0] * nested_belief[:, 1])
        if self.action_exploration_policy.agent_type == "worker":
            split_pot = (persona - average_nested_persona) / 2
            final_offer = persona - split_pot
        else:
            split_pot = (average_nested_persona - persona) / 2
            final_offer = split_pot + persona
        reward = self.reward_function(final_offer)
        return reward

    # def plot_max_value_trajectory(self, root_node: HistoryNode):
    #     planning_tree = self.extract_max_q_value_trajectory(root_node)
    #     g = nx.DiGraph()
    #     g.add_nodes_from(list(planning_tree.keys()))
    #     g.add_edges_from(zip(planning_tree.keys(), [x[0] for x in planning_tree.values()]))
    #     nx.draw(g, with_labels=True)
    #     plt.draw()
    #     plt.show()
    #     return planning_tree
    #
    def extract_max_q_value_trajectory(self, root_node: HistoryNode, planning_tree=None, belief_tree=None):
        if planning_tree is None or belief_tree is None:
            tree = [["root", None, root_node.id, root_node.parent.action.value, root_node.observation.value, 0.0]]
            beliefs = [["root", None, root_node.id, root_node.compute_persona_distribution(),
                        root_node.compute_nested_belief_distribution()]]
        else:
            tree = planning_tree
            beliefs = belief_tree
        max_q_value_action = np.argmax(root_node.children_qvalues[:, 1])
        optimal_child = root_node.children[str(root_node.children_values[max_q_value_action])]
        tree.append(["action", root_node.id, optimal_child.id, optimal_child.parent.observation.value,
                     optimal_child.action.value, 1.0, optimal_child.q_value])
        beliefs.append(["action", root_node.id, optimal_child.id, optimal_child.parent.observation.value,
                        optimal_child.action.value, 1.0, optimal_child.compute_persona_distribution(),
                        root_node.compute_nested_belief_distribution()])
        tree, beliefs = self.extract_max_value_trajectory(optimal_child, tree, beliefs)
        return tree, beliefs

    def extract_max_value_trajectory(self, root_node: ActionNode, planning_tree, beliefs_tree):
        for potential_observation in root_node.children:
            child = root_node.children[potential_observation]
            node = ["observation",
                    root_node.id, child.id,
                    child.parent.action.value,
                    child.observation.value,
                    child.probability,
                    child.node_value()]
            beliefs = ["observation", root_node.id, child.id, child.parent.action.value,
                       child.observation.value, root_node.compute_persona_distribution(),
                       root_node.compute_nested_belief_distribution()]
            planning_tree.append(node)
            beliefs_tree.append(beliefs)
            planning_tree, beliefs_tree = self.extract_max_q_value_trajectory(child, planning_tree, beliefs_tree)
        return planning_tree, beliefs_tree




