class Action:

    def __init__(self, value: float, terminal: bool = False):
        """

        :param value: float, the q(value) of an action
        :param terminal: bool, indicates if the action terminates the environment or not
        """
        self.value = value
        self.is_terminal = terminal

    def __str__(self):
        return str(self.value) if self.value else None

class State:

    def __init__(self, name:str, terminal:bool):
        """

        :type name: str, the name of the state
        :type terminal: bool, indicate if the state is terminal
        """
        self.name = name
        self.terminal = terminal


class InteractiveState:

    def __init__(self, state:State, persona, opponent_belief):
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
    def get_nested_belief(self):
        return self.opponent_belief

    @property
    def sample_from_opponent_belief(self):
        if self.opponent_belief is None:
            return None
        return self.opponent_belief.sample()

    def update_nested_belief(self, belief):
        self.opponent_belief = belief
