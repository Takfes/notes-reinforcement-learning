## Markov Descion Processes

- **Overview**
- Markov Process
- Markov Reward Process = Markov Process + Rewards
- Markov Decision Process = Markov Reward Process + Decision
- Markov Property : Future is independent of the past given the future
- **Sampling** a Markov Chain = sampling **state transitions sequences** produced from the underlying chain

---

- **Markov Reward Process (MRP)** adding Rewards and Discount Factor to Markov Chain
- $R$ is the reward we are getting from a single state
- Ultimately we want to optimize the cumulative sum of such rewards $R$
- $G$ is the total discounted reward

---

- **Markov Decision Process (MDP)**
- adding decision making on top of MRP and we get MDPs.
- Note the difference between Bellman Expectation and Optimality Equations.
