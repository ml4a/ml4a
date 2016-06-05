---
layout: guide
title: Reinforcement Learning
---

Reinforcement learning involves training agents - programs, robots, etc - such that they learn how to behave in a particular environment. Reinforcement learning agents learn by interacting with their environment and seeing what happens as a result of their actions.

The basic structure of a reinforcement learning problem is:

- we model the world as various states the agent can be in.
- the agent can take actions that move between these states.
- each state has an associated reward (which may be negative, i.e. a "punishment").
- the agent explores these states and learns which sequence of actions tend to lead to more rewards.
- the agent uses what it has learned to behave in a way that maximizes reward (to the best of its knowledge).

There are many different ways a reinforcement learning agent can be trained, but a common one is called _Q-learning_. The agent learns a function called $Q(s,a)$ which takes a state $s$ and an action $a$ and returns a value for it. The higher the value, the more the agent believes $a$ is a good action to take in state $s$.

The way the agent actually behaves (i.e. decides what to do) is governed by what is called a _policy_. With Q-learning a common policy is a _greedy policy_ which just means take the highest valued action given by $Q(s,a)$.

There are two problems in reinforcement learning that you should be familiar with:

- the _credit assignment_ problem: sometimes a reward may be due to an action taken a long time ago - how do we properly assign credit to that action? Q-learning handles this by propagating rewards through time, so an early action that later leads to a reward will have some of that reward assigned to it. This will be clearer when we dig into the code.
- the _exploration vs exploitation_ problem: does the agent stick with certain rewards at the expense of possibly missing out on greater but unknown rewards (exploit)? Does the agent explore more states to find these possibly greater rewards, but at the risk of lower or negative ones (explore)? A simple approach, which we'll use here, is to set some value, called epsilon ($\epsilon$), which can be from 0 to 1. With this $\epsilon$ probability the agent will take a random action instead of the best one. This variation of the greed policy is called the $\epsilon$-greedy policy.

In this guide we'll put together a very simple Q-learning agent that navigates a grid world.

For simplicity, we are going to consider a _fully-observed_ scenario; that is, when an agent takes an action, they see all the results of it (this is contrasted to _partially-observable_ scenarios, where some results remain unknown, perhaps until later or surfacing in different ways). Our scenario will also be _deterministic_ in that an action, from a given state, always leads to the same outcome.

## The environment

A reinforcement learning agent needs an environment to explore and interact with, so let's create that first.

This will just be a simple set of discrete coordinates. So the states in our scenario will just be `(x,y)` coordinate positions.

We'll design it so that we can pass in a grid (i.e. a list of lists) that is filled with reward values. If a value of `None` is specified, it's considered a wall, i.e. the agent cannot move there.

The environment can also be used to figure out what valid actions are given a state. For example, if the agent is in the upper-left corner of the map, the only valid actions are `right`, `down`, and `stay`.

```python
class Environment():
    def __init__(self, grid):
        self.grid = grid
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])
        self.positions = self._positions()

    def actions(self, pos):
        """possible actions for a state (position)"""
        r, c = pos
        actions = ['stay']
        if r > 0 and self.grid[r-1][c] is not None:
            actions.append('up')
        if r < self.n_rows - 1 and self.grid[r+1][c] is not None:
            actions.append('down')
        if c > 0 and self.grid[r][c-1] is not None:
            actions.append('left')
        if c < self.n_cols - 1 and self.grid[r][c+1] is not None:
            actions.append('right')
        return actions

    def value(self, pos):
        """retrieve the reward value for a position"""
        r, c = pos
        return self.grid[r][c]

    def _positions(self):
        """all valid positions"""
        positions = []
        for r, row in enumerate(self.grid):
            for c, _ in enumerate(row):
                if self.grid[r][c] is not None:
                    positions.append((r,c))
        return positions
```

The way we can use this is like so:

```python
env = Environment([
    [ -10,  0,    0, 50,    0, None, None],
    [   0, 10,  100,  0, -100,   20, None],
    [   0,  0, None, 10, None,  -10, None],
    [None,  0,    5, 10, None,  500,    0]
])
```

## The agent

Now we'll move on to designing the agent.

A few notes here:

- We have a `learning_rate` parameter here (it takes a value from 0 to 1). This is useful for stochastic (non-detemrinistic) environments; it allows you to control how much new information overwrites existing information about the environment. In stochastic environments, sometimes random weird things happen and you don't want it influencing the agent too much. In deterministic environment (like ours), this isn't a problem, so you want to set it to 1 so the agent learns as quickly as possible.
- We have a `decay` parameter which we use to decrease our `explore` value each time step. Thus over time the agent explores less and less and sticks to rewards its familiar with.
- The `discount` value (from 0 to 1) specifies how much future rewards are discounted by. For instance, with `discount=0.5`, a reward at the next time step is only worth half as much as it would be now. A reward at the time step after that would be discounted twice-over, i.e. it would be worth only `0.5 * 0.5` of it's actual value.
- Instead of implementing Q as a function, we're using a lookup table. This works fine for our purposes here, but just know that it can also be a learned function (this is where deep-Q learning comes in, which I'll cover in another guide).

The most important piece here is the `_learn` method. There you can see how these individual parts come together to update the value of an `(state, action)` pair. Note that when we propagate values from future states we are optimistic and take the maximum of those values. This is appropriate because we are using a greedy policy - we'll always choose the action that takes us to the best state, so we'll always be getting the next maximum (known) reward value.


```python
class QLearner():
    def __init__(self, state, environment, rewards, discount=0.5, explore=1, learning_rate=1, decay=0.005):
        """
        - state: the agent's starting state
        - rewards: a reward function, taking a state as input, or a mapping of states to a reward value
        - discount: how much the agent values future rewards over immediate rewards
        - explore: with what probability the agent "explores", i.e. chooses a random action
        - decay: how much to decay the explore rate with each step
        - learning_rate: how quickly the agent learns. For deterministic environments (like ours), this should be left at 1
        """
        self.discount = discount
        self.explore = explore
        self.decay = decay
        self.learning_rate = learning_rate
        self.R = rewards.get if isinstance(rewards, dict) else rewards

        # our state is just our position
        self.state = state
        self.env = environment

        # initialize Q
        self.Q = {}

    def actions(self, state):
        return self.env.actions(state)

    def _take_action(self, state, action):
        r, c = state
        if action == 'up':
            r -= 1
        elif action == 'down':
            r += 1
        elif action == 'right':
            c += 1
        elif action == 'left':
            c -= 1

        # return new state
        return (r,c)

    def step(self, action=None):
        """take an action"""
        # check possible actions given state
        actions = self.actions(self.state)

        # if this is the first time in this state,
        # initialize possible actions
        if self.state not in self.Q:
            self.Q[self.state] = {a: 0 for a in actions}

        if action is None:
            if random.random() < self.explore:
                action = random.choice(actions)
            else:
                action = self._best_action(self.state)
        elif action not in actions:
            raise ValueError('unrecognized action!')

        # remember this state and action
        # so we can later remember
        # "from this state, taking this action is this valuable"
        prev_state = self.state

        # decay explore
        self.explore = max(0, self.explore - self.decay)

        # update state
        self.state = self._take_action(self.state, action)

        # update the previous state/action based on what we've learned
        self._learn(prev_state, action, self.state)
        return action

    def _best_action(self, state):
        """choose the best action given a state"""
        actions_rewards = list(self.Q[state].items())
        return max(actions_rewards, key=lambda x: x[1])[0]

    def _learn(self, prev_state, action, new_state):
        """update Q-value for the last taken action"""
        if new_state not in self.Q:
            self.Q[new_state] = {a: 0 for a in self.actions(new_state)}
        self.Q[prev_state][action] = self.Q[prev_state][action] + self.learning_rate * (self.R(new_state) + self.discount * max(self.Q[new_state].values()) - self.Q[prev_state][action])
```

With the agent defined, we can try running it in our environment:

```python
import time
import random

# start at a random position
pos = random.choice(env.positions)

# simple reward function
def reward(state):
    return env.value(state)

# try discount=0.1 and discount=0.9
agent = QLearner(pos, env, reward, discount=0.9, learning_rate=0.8, decay=0.5/steps)

delay = 0.5
steps = 500
for i in range(steps):
    agent.step()

    # print out progress
    print('step: {:03d}/{:03d}, explore: {:.2f}, discount: {}'.format(i+1, steps, agent.explore, agent.discount))

    # print out the agent's Q table
    for pos, vals in agent.Q.items():
        print('{} -> {}'.format(pos, vals))

    # delay so we can see how these values are updated over time
    time.sleep(delay)
```

One thing to try is changing the discount value. In the environment I setup above, there is a reward of 500 that is stuck on a path where one state has a reward of -100. If the agent has a low discount value, i.e. `discount=0.1`, they will avoid that -100 reward state even though there is a large reward on the other side. The reward is discounted by so much that it is not worth it to the agent.

On the other hand, if you set the discount higher, i.e. `discount=0.9`, then it will value future rewards more and decide to trudge through the -100 reward state to reach the 500 reward one.