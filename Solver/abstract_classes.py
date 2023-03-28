from abc import ABC, abstractmethod
from typing import Optional


class Action:

    def __init__(self, value: Optional[float], terminal: bool = False):
        """

        :param value: float, the q(value) of an action
        :param terminal: bool, indicates if the action terminates the environment or not
        """
        self.value = value
        self.is_terminal = terminal

    def __str__(self):
        return str(self.value)


class State:

    def __init__(self, name: str, terminal: bool):
        """

        :type name: str, the name of the state
        :type terminal: bool, indicate if the state is terminal
        """
        self.name = name
        self.terminal = terminal


class InteractiveState:

    def __init__(self, state: Optional[State], persona, opponent_belief):
        """

        :param state: State, indicating the state of the game
        :param persona:
        :param opponent_belief:
        """
        self.state = state
        self.persona = persona
        self.opponent_belief = opponent_belief

    @property
    def get_state(self):
        return self.state

    @property
    def get_persona(self):
        return self.persona

    @property
    def get_belief(self):
        return self.opponent_belief

    @property
    def get_nested_belief(self):
        return self.opponent_belief

    @property
    def sample_from_opponent_belief(self):
        if self.opponent_belief is None:
            return None
        return self.opponent_belief.sample()

    def update_nested_belief(self, belief):
        self.opponent_belief = belief


class History:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.rewards = []

    def reset(self, action_length, observation_length):
        self.actions = self.actions[0:action_length]
        self.observations = self.observations[0:observation_length]
        self.rewards = self.rewards[0:action_length]

    def get_last_observation(self):
        if len(self.observations) <= 1:
            last_observation = Action(None, False)
        else:
            last_observation = self.observations[-2]
        return last_observation

    def update_history(self, action: Action, observation: Action, reward: Optional[float] = None):
        self.update_actions(action)
        self.update_observations(observation)
        self.update_rewards(reward)

    def update_actions(self, action):
        self.actions.append(action)

    def update_observations(self, observation):
        self.observations.append(observation)

    def update_rewards(self, reward):
        self.rewards.append(reward)


class BeliefDistribution(ABC):
    """
    Samples root particles from the current history
    """

    def __init__(self, prior_belief, opponent_model, history: History):
        self.opponent_model = opponent_model
        self.prior_belief = prior_belief
        self.belief_distribution = self.prior_belief
        self.history = history

    def reset(self):
        self.belief_distribution = self.prior_belief

    def reset_prior(self):
        self.belief_distribution = self.prior_belief

    def get_current_belief(self):
        return self.belief_distribution

    @abstractmethod
    def update_distribution(self, action, observation, first_move):
        pass

    @abstractmethod
    def sample(self, rng_key, n_samples):
        pass


class EnvironmentModel:

    def __init__(self, opponent_model=None, belief_distribution=None):
        self.opponent_model = opponent_model
        self.belief_distribution = belief_distribution
        self.reward_function = None

    @abstractmethod
    def reset_persona(self, persona, action_length, observation_length, nested_beliefs):
        pass

    @abstractmethod
    def step(self, interactive_state: InteractiveState, action: Action, observation: Action, seed: int,
             iteration_number: int):
        pass

    @abstractmethod
    def update_persona(self, observation: Action, action: Action):
        pass
