from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
import random
import numpy as np

class TemporalDifference(ABC):

    @abstractmethod
    def method(self, nA: int, 
               next_state: int, 
               epsilon: float,
               Q: defaultdict) -> float:
        pass
    
class Sarsa(TemporalDifference):
    def method(self, nA: int, 
               next_state: int, 
               epsilon: float,
               Q: defaultdict) -> float:
        next_action: int = Agent.choose_epsilon_greedy_action(
            next_state, nA, epsilon, Q)
        return Q[next_state][next_action]
    
class ExpectedSarsa(TemporalDifference):
    def method(self, nA: int, 
               next_state: int, 
               epsilon: float,
               Q: defaultdict) -> float:
        max_action: int = np.argmax(Q[next_state])
        policy = np.ones(nA) * epsilon / nA
        policy[max_action] = 1 - epsilon + (epsilon / nA)
        return np.dot(Q[next_state], policy)
    
class QLearning(TemporalDifference):
    def method(self, nA: int, 
               next_state: int, 
               epsilon: float,
               Q: defaultdict) -> float:
        max_action: int = np.argmax(Q[next_state])
        return Q[next_state][max_action]

class Agent:

    def __init__(self, 
                 nA: int = 6,
                 temporal_difference: TemporalDifference = QLearning(),
                 alpha: float = 0.05, 
                 gamma: float = 0.9, 
                 epsilon_start: float = 1.0,
                 epsilon_decay:float=0.99999,
                 epsilon_min:float = 0.05) -> None:
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA: int = nA
        self._temporal_difference: TemporalDifference = temporal_difference
        self.Q: defaultdict = defaultdict(lambda: np.zeros(self.nA))
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon_start
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min
        self.count: int = 0
            
    @property
    def temporal_difference(self) -> TemporalDifference:
        return self._temporal_difference

    @temporal_difference.setter
    def temporal_difference(self, temporal_difference: TemporalDifference) -> None:
        self._temporal_difference = temporal_difference
    
    @staticmethod
    def choose_epsilon_greedy_action(state: int,
                                     nA: int,
                                     epsilon: float,
                                     Q: defaultdict) -> int:
        if random.random() > epsilon:
            return np.argmax(Q[state])
        else:
            return np.random.choice(
                np.arange(nA),
                p=np.full((nA), 1/nA))
    
    def select_action(self, state: int) -> int:
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return Agent.choose_epsilon_greedy_action(
            state, self.nA, self.epsilon, self.Q)

    def step(self, state, action, reward, next_state, done) -> None:
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        current_estimate: float = self.Q[state][action]
        if not done:
            next_estimate = self._temporal_difference.method(
                self.nA, next_state, self.epsilon, self.Q)
            target: float = reward + (self.gamma * next_estimate)
            self.Q[state][action] += self.alpha * (target - current_estimate)
            state = next_state
        else:
            target: float = reward + (self.gamma * 0.0)
            self.Q[state][action] += self.alpha * (target - current_estimate)
