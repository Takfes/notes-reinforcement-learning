# Derivation of Policy Gradient Theorem

This document derives the policy gradient theorem from the objective function for reinforcement learning. We start with the objective function $ J(\theta) $, apply the gradient, use the likelihood ratio trick, and simplify to reach the final form.

### Resources

- [Policy Gradient - Youtube Walkthrough](https://www.youtube.com/watch?v=-tkZf8rOEhU&list=PLCip3d1iHEMWlWV9fGh4eDTUs_otqosZU&index=10)
- [Policy Gradient - Udemy A2C Walkthrough](https://titancement.udemy.com/course/cutting-edge-artificial-intelligence/learn/lecture/14355778#overview)

## Objective Function

The objective function $ J(\theta) $ is given by:

$$ J(\theta) = \sum\_{\tau} P(\tau; \theta) R(\tau) $$

where:

- $ \theta $ are the policy parameters.
- $ \tau $ represents a trajectory.
- $ P(\tau; \theta) $ is the probability of the trajectory $ \tau $ given the policy parameters $ \theta $.
- $ R(\tau) $ is the return of the trajectory $ \tau $.

## Probability of a Trajectory

The probability of a trajectory $ P(\tau; \theta) $ is given by:

$$ P(\tau; \theta) = \left[ \prod_{t=0}^{T} P(s_{t+1} \mid s_t, a_t) \pi_\theta(a_t \mid s_t) \right] $$

where:

- $ P(s\_{t+1} \mid s_t, a_t) $ represents the environment dynamics (state transition probabilities).
- $ \pi\_\theta(a_t \mid s_t) $ represents the policy (probability of taking action $ a_t $ given state $ s_t $ and parameters $ \theta $).

## Gradient of the Objective Function

Starting from the objective function $ J(\theta) $

$$ J(\theta) = \sum\_{\tau} P(\tau; \theta) R(\tau) $$

To find the gradient of $ J(\theta) $ with respect to $ \theta $:

$$ \nabla*\theta J(\theta) = \nabla*\theta \sum\_{\tau} P(\tau; \theta) R(\tau) $$

Since the sum is linear, the gradient can be moved inside the sum:

$$ {\nabla}_{\theta} J(\theta) = \sum\*{\tau} \nabla_\theta [P(\tau; \theta) R(\tau)] $$

## Applying the Product Rule

Using the product rule to take the gradient:

$$ \nabla*\theta [P(\tau; \theta) R(\tau)] = R(\tau) \nabla*\theta P(\tau; \theta) $$

## Using the Log Trick

We can use the log trick to simplify the gradient of the probability:

$$ \nabla*\theta P(\tau; \theta) = P(\tau; \theta) \nabla*\theta \log P(\tau; \theta) $$

Since $ P(\tau; \theta) $ is a product of the policy probabilities:

$$ \log P(\tau; \theta) = \log \left( \prod*{t=0}^{T} \pi*\theta(a*t \mid s_t) \right) = \sum*{t=0}^{T} \log \pi\_\theta(a_t \mid s_t) $$

Therefore:

$$ \nabla*\theta \log P(\tau; \theta) = \nabla*\theta \sum*{t=0}^{T} \log \pi*\theta(a*t \mid s_t) = \sum*{t=0}^{T} \nabla\_\theta \log \pi\*\theta(a_t \mid s_t) $$

## Combining the Results

Now we have:

$$ \nabla*\theta P(\tau; \theta) = P(\tau; \theta) \sum*{t=0}^{T} \nabla*\theta \log \pi*\theta(a_t \mid s_t) $$

Substituting back into the gradient of $ J(\theta) $:

$$ \nabla*\theta J(\theta) = \sum*{\tau} P(\tau; \theta) R(\tau) \sum*{t=0}^{T} \nabla*\theta \log \pi\_\theta(a_t \mid s_t) $$

This can be rewritten as:

$$ \nabla*\theta J(\theta) = \sum*{\tau} P(\tau; \theta) \sum*{t=0}^{T} R(\tau) \nabla*\theta \log \pi\_\theta(a_t \mid s_t) $$

## Recognizing the Expectation Form

Recognize that the sum over trajectories weighted by their probability is the expectation:

$$ \nabla*\theta J(\theta) = \mathbb{E}*{\pi*\theta} \left[ \sum*{t=0}^{T} R(\tau) \nabla*\theta \log \pi*\theta(a_t \mid s_t) \right] $$

## Simplifying

For simplicity, we often use the return $ R(\tau) $ as the total return from the trajectory:

$$ \nabla*\theta J(\theta) = \mathbb{E}*{\pi*\theta} \left[ \nabla*\theta \log \pi\_\theta(a_t \mid s_t) R(\tau) \right] $$

This is the final form shown in the policy gradient theorem.

## Summary

1. **Start with the objective function** $ J(\theta) = \sum\_{\tau} P(\tau; \theta) R(\tau) $.
2. **Take the gradient** $ \nabla\_\theta J(\theta) $ and apply the product rule.
3. **Use the log trick** to simplify the gradient of the probability.
4. **Recognize the expectation form** to derive the policy gradient theorem:

   $$ \nabla*\theta J(\theta) = \mathbb{E}*{\pi*\theta} \left[ \nabla*\theta \log \pi\_\theta(a_t \mid s_t) R(\tau) \right] $$

This derivation helps bridge the gap between the probabilistic formulation of the problem and the practical computation of policy gradients for reinforcement learning.
