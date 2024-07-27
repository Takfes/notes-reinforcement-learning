### Monte Carlo

The Monte Carlo (MC) approach in the context of Reinforcement Learning (RL) refers to methods that use sampled sequences of states, actions, and rewards (i.e., episodes) to estimate the value of a policy. This approach relies on the law of large numbers to estimate the expected returns from states or state-action pairs.

**Steps to Calculate State Values using Monte Carlo**

1. **Generate Episodes** : Interact with the environment using the policy π to generate episodes. Each episode consists of a sequence of state-action-reward (SAR) tuples and ends in a terminal state.

Example trajectory : (s0,a0,r0),(s1,a1,r1)...(sT,aT,rT)

2. **Calculate Returns** : For each episode, calculate the return 𝐺𝑡 _from each time step 𝑡 to the end of the episode_. The return is the sum of the discounted rewards from that time step onwards:

$$ G*{t} = ∑*{k=0}^{T-t} γ^{k}r\_{t+k}$$

3. **Estimate Value Function** : Update the value estimates for each state based on the returns observed. The value of a state is the average of the returns observed when visiting that state.

### Temporal Difference

The TD approach estimates value functions using bootstrapping, meaning it updates estimates based on other learned estimates rather than waiting for the final outcome (as in Monte Carlo methods). The key idea is to update the value of a state based on the value of the next state and the reward received, providing a more immediate update.

TD(0) Algorithm
The simplest form of the TD method is TD(0), which updates the value of a state V(s) based on the observed reward and the estimated value of the next state.

The update rule for TD(0) is

$$ V*{(S*{t})} ← V*{(S*{t})} + \alpha [r_{t+1} + \gamma V_{(S_{t+1})} - V_{(S_{t})}] $$

where assuming :

- TD Target : $ r*{t+1} + \gamma V*{(S\_{t+1})} $
- TD Error : TD Target - $ V*{(S*{t})} $

the above update rule turns out to be :

$ V*{(S*{t})} ← V*{(S*{t})} + \alpha $ \* TD Error

### Differences Between Monte Carlo and Temporal Difference Approaches

**Update Frequency**

Monte Carlo: Updates the value estimates only at the end of an episode. This requires waiting until the episode is complete to calculate the total return 𝐺𝑡​ for each state visited.

Temporal Difference: Updates the value estimates after each time step, making it more efficient and allowing for updates during the episode.

**Variance and Bias**

Monte Carlo: Estimates are unbiased because they are based on actual returns observed. However, they can have high variance due to the complete episode's variability.

Temporal Difference: Estimates are biased because they use the current value estimates (bootstrapping). However, they typically have lower variance compared to Monte Carlo methods.
