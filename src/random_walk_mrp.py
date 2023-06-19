import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


class RandomWalkMRP:
    def __init__(self, transition_matrix, rewards_vector):
        # Define the transition matrix
        self.transition_matrix = transition_matrix
        assert (
            self.transition_matrix.shape[0] == self.transition_matrix.shape[0]
        ), f"Transition matrix must be square, shape provided {self.transition_matrix.shape=}"
        # Define the reward vector
        self.rewards_vector = rewards_vector
        self.terminal_states = [0, self.transition_matrix.shape[0] - 1]

    def transition(self, state):
        """Transition to a new state given the current state."""
        probabilities = self.transition_matrix[state]
        next_state = np.random.choice(len(probabilities), p=probabilities)
        return next_state

    def sample_episodes(self, num_episodes, initial_state=3):
        """Draw samples from the MRP."""
        episodes = {}
        for i in range(num_episodes):
            states = []
            rewards = []
            state = initial_state
            while state not in self.terminal_states:
                state = self.transition(state)
                reward = self.rewards_vector[state]
                states.append(state)
                rewards.append(reward)
            episodes[i] = {"states": states, "rewards": rewards}
        return episodes


def mc_first_visit(episodes, gamma=1.0):
    """
    First-visit MC Value Estimation.

    :param episodes: A dictionary of dicitonaries.
    like so:
    {
        0: {
            'states': [4, 3, 2, 3, 2, 3, 4, 3, 4, 3, 2, 1, 0],
            'rewards': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        1: {
            'states': [2, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6],
            'rewards': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            }
    }
    :param gamma: The discount factor.
    """
    # Initialize the sum of the returns, the count of returns, and the value estimate
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)

    # For each episode
    for _, v in episodes.items():
        states, rewards = v["states"], v["rewards"]
        discounts = [gamma**i for i in range(len(rewards))]
        for i, state in enumerate(states):
            # Compute the return following the first visit to state
            G_t = sum([a * b for a, b in zip(discounts[i:], rewards[i:])])
            # Update the sum of the returns, the count of returns, and the value estimate
            returns_sum[state] += G_t
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]

    return V


def mc_constant_alpha(episodes, alpha=0.1, gamma=1.0):
    """
    Constant-alpha MC Value Estimation.

    :param episodes: A dictionary of dicitonaries.
    like so:
    {
        0: {
            'states': [4, 3, 2, 3, 2, 3, 4, 3, 4, 3, 2, 1, 0],
            'rewards': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        1: {
            'states': [2, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6],
            'rewards': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            }
    }
    :param alpha: The step-size parameter.
    :param gamma: The discount factor.
    """
    # Initialize the value estimate
    V = defaultdict(float)

    # For each episode
    for _, v in episodes.items():
        states, rewards = v["states"], v["rewards"]
        discounts = [gamma**i for i in range(len(rewards))]
        for i, state in enumerate(states):
            # Compute the return following the current state
            G_t = sum([a * b for a, b in zip(discounts[i:], rewards[i:])])
            # Update the value estimate
            V[state] += alpha * (G_t - V[state])
    return V


if __name__ == "__main__":
    # Define number of MRP episodes to sample
    N_EPISODES = 1000
    MC_CONSTANT_ALPHA = 0.005

    # Define the transition matrix
    transition_matrix = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0.5, 0, 0.5, 0, 0, 0, 0],
            [0, 0.5, 0, 0.5, 0, 0, 0],
            [0, 0, 0.5, 0, 0.5, 0, 0],
            [0, 0, 0, 0.5, 0, 0.5, 0],
            [0, 0, 0, 0, 0.5, 0, 0.5],
            [0, 0, 0, 0, 0, 0, 1],
        ]
    )

    # Define the reward vector
    rewards = np.array([0, 0, 0, 0, 0, 0, 1])

    # Create an MRP
    mrp = RandomWalkMRP(transition_matrix, rewards)

    # Sample episodes from MRP
    episodes = mrp.sample_episodes(N_EPISODES, 3)

    # Compute the value estimates based on first visit MC
    Vfv = mc_first_visit(episodes)
    df_mcfv = (
        pd.DataFrame()
        .from_dict(Vfv, orient="index")
        .sort_index()
        .rename(columns={0: "mc_first_visit"})
    )

    # Compute the value estimates based on constant value MC
    Vca = mc_constant_alpha(episodes, alpha=MC_CONSTANT_ALPHA)
    df_mcca = (
        pd.DataFrame()
        .from_dict(Vca, orient="index")
        .sort_index()
        .rename(columns={0: "mc_constant_alpha"})
    )

    # Compute the expected value estimates
    df_exp = pd.DataFrame(
        [
            (1 / (transition_matrix.shape[0] - 1)) * x
            for x in range(transition_matrix.shape[0])
        ],
        columns=["expected"],
    )

    df = df_exp.join(df_mcfv).join(df_mcca)

    # Plot the DataFrame with lines and dots
    df.plot(marker="o", linestyle="-", grid=True)
    plt.show()
