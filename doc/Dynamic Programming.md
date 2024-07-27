## Dynamic Programming

**Policy vs Value Iteration**

Policy iteration and value iteration are two common algorithms used in the context of Markov Decision Processes (MDPs) to solve for the optimal policy. Both algorithms involve iterative processes and leverage the Bellman equation, but they differ in their approach. Policy iteration alternates between policy evaluation and improvement steps, while value iteration directly computes the optimal value function and policy.

1. Policy Iteration:

   - Policy iteration is an iterative algorithm that alternates between policy evaluation and policy improvement steps until convergence.
   - Policy evaluation aims to estimate the value function for a given policy. It iteratively updates the value estimates for each state by applying the Bellman expectation equation, which relates the value of a state to the expected immediate reward and the value of the next state.
   - Policy improvement involves updating the policy based on the current value function. It selects actions that are greedy with respect to the value function, meaning it chooses the action that maximizes the expected cumulative rewards from the current state.
   - The algorithm continues the policy evaluation and policy improvement steps until the policy converges to the optimal policy, meaning no further changes in the policy occur.

2. Value Iteration:
   - Value iteration is an iterative algorithm that directly computes the optimal value function and policy in a single step, without explicitly separating policy evaluation and policy improvement.
   - It starts with an initial value function estimate and iteratively updates the value estimates for each state using the Bellman optimality equation. The Bellman optimality equation relates the value of a state to the maximum expected cumulative rewards achievable from that state.
   - In each iteration, the algorithm performs a lookahead to select the action that maximizes the value function update at each state. This allows for direct convergence to the optimal policy.
   - The algorithm continues iterating until the value function converges to the optimal value function, which implies that the optimal policy has also been obtained.

Differences:

- Policy iteration performs separate steps for policy evaluation and policy improvement, while value iteration combines both steps in a single iteration.
- Policy iteration generally takes more iterations to converge compared to value iteration since it alternates between evaluation and improvement steps.
- Value iteration can be computationally more efficient than policy iteration in certain cases due to its one-step lookahead and direct convergence to the optimal policy.

- [Code Policy Iteration](https://www.youtube.com/watch?v=RlugupBiC6w)
- [Code Value Iteration](https://www.youtube.com/watch?v=hUqeGLkx_zs)
