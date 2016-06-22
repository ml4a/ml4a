---
layout: guide
title: "Reinforcement Learning: Deep Q-Networks"
---

If you aren't familiar with reinforcement learning, check out the previous guide on reinforcement learning for an introduction.

In the previous guide we implemented the Q function as a lookup table. That worked well enough for that scenario because it had a fairly small state space. However, consider something like [DeepMind's Atari player](http://www.wired.co.uk/article/google-deepmind-atari). A state in that task is a unique configuration of pixels. All those Atari games are color, so each pixel has three values (R,G,B), and there are quite a few pixels. So there is a massive state space for all possible configurations of pixels, and we simply can't implement a lookup table encompassing all of these states - it would take up too much memory.

Instead, we can learn a Q _function_ that approximately maps a set of pixel values and an action to some value. We could implement this Q function as a neural network and have it learn how to predict rewards for each action given an input state. This is the general idea behind _deep Q-learning_ (i.e. deep Q networks, or DQNs).

Here we'll put together a simple DQN agent that learns how to play a simple game of catch. The agent controls a paddle at the bottom of the screen that it can move left, right, or not at all (so there are three possible action). An object falls from the top of the screen, and the agent wins if it catches it (a reward of +1). Otherwise, it loses (a reward of -1).

We'll implement the game in black-and-white so that the pixels in the game can be represented as 1 or 0.

Using DQNs are quite like using neural networks in ways you may be more familiar with. Here we'll take a vector that represents the screen, feed it through the network, and the network will output a distribution of values over possible actions. You can think of it as a classification problem: given this input state, label it with the best action to take.

TODO atari image

This scenario is simple enough that we don't need convolutional neural networks, but we could easily extend it in that way if we wanted (just replace our vanilla neural network with a convolutional one).

To start I'll present the code for the catch game itself. It's not important that you understand this code - the part we care about is the agent itself.

Note that this needs to be run in the terminal in order to visualize the game.

```python
import numpy as np
from blessings import Terminal

class Game():
    def __init__(self, shape=(10,10)):
        self.shape = shape
        self.height, self.width = shape
        self.last_row = self.height - 1
        self.paddle_padding = 1
        self.n_actions = 3 # left, stay, right
        self.term = Terminal()
        self.reset()

    def reset(self):
        # reset grid
        self.grid = np.zeros(self.shape)

        # can only move left or right (or stay)
        # so position is only its horizontal position (col)
        self.pos = np.random.randint(self.paddle_padding, self.width - 1 - self.paddle_padding)
        self.set_paddle(1)

        # item to catch
        self.target = (0, np.random.randint(self.width - 1))
        self.set_position(self.target, 1)

    def move(self, action):
        # clear previous paddle position
        self.set_paddle(0)

        # action is either -1, 0, 1,
        # but comes in as 0, 1, 2, so subtract 1
        action -= 1
        self.pos = min(max(self.pos + action, self.paddle_padding), self.width - 1 - self.paddle_padding)

        # set new paddle position
        self.set_paddle(1)

    def set_paddle(self, val):
        for i in range(1 + self.paddle_padding*2):
            pos = self.pos - self.paddle_padding + i
            self.set_position((self.last_row, pos), val)

    @property
    def state(self):
        return self.grid.reshape((1,-1)).copy()

    def set_position(self, pos, val):
        r, c = pos
        self.grid[r,c] = val

    def update(self):
        r, c = self.target

        self.set_position(self.target, 0)
        self.set_paddle(1) # in case the target is on the paddle
        self.target = (r+1, c)

        # off the map, it's gone
        if r == self.last_row:
            # reward of 1 if collided with paddle, else -1
            if abs(c - self.pos) <= self.paddle_padding:
                return 1
            else:
                return -1

        self.set_position(self.target, 1)

        return 0

    def render(self):
        print(self.term.clear())
        for r, row in enumerate(self.grid):
            for c, on in enumerate(row):
                if on:
                    color = 235
                else:
                    color = 229

                print(self.term.move(r, c) + self.term.on_color(color) + ' ' + self.term.normal)

        # move cursor to end
        print(self.term.move(self.height, 0))
```

Ok, on to the agent itself. I'll present the code in full here, then explain parts in more detail.

```python
import random
from keras.models import Sequential
from keras.layers.core import Dense
from collections import deque

class Agent():
    def __init__(self, env, explore=0.1, discount=0.9, hidden_size=100, memory_limit=5000):
        self.env = env
        model = Sequential()
        model.add(Dense(hidden_size, input_shape=(env.height * env.width,), activation='relu'))
        model.add(Dense(hidden_size, activation='relu'))
        model.add(Dense(env.n_actions))
        model.compile(loss='mse', optimizer='sgd')
        self.Q = model

        # experience replay:
        # remember states to "reflect" on later
        self.memory = deque([], maxlen=memory_limit)

        self.explore = explore
        self.discount = discount

    def choose_action(self):
        if np.random.rand() <= self.explore:
            return np.random.randint(0, self.env.n_actions)
        state = self.env.state
        q = self.Q.predict(state)
        return np.argmax(q[0])

    def remember(self, state, action, next_state, reward):
        # the deque object will automatically keep a fixed length
        self.memory.append((state, action, next_state, reward))

    def _prep_batch(self, batch_size):
        if batch_size > self.memory.maxlen:
            Warning('batch size should not be larger than max memory size. Setting batch size to memory size')
            batch_size = self.memory.maxlen

        batch_size = min(batch_size, len(self.memory))

        inputs = []
        targets = []

        # prep the batch
        # inputs are states, outputs are values over actions
        batch = random.sample(list(self.memory), batch_size)
        random.shuffle(batch)
        for state, action, next_state, reward in batch:
            inputs.append(state)
            target = self.Q.predict(state)[0]

            # debug, "this should never happen"
            assert not np.array_equal(state, next_state)

            # non-zero reward indicates terminal state
            if reward:
                target[action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                Q_sa = np.max(self.Q.predict(next_state)[0])
                target[action] = reward + self.discount * Q_sa
            targets.append(target)

        # to numpy matrices
        return np.vstack(inputs), np.vstack(targets)

    def replay(self, batch_size):
        inputs, targets = self._prep_batch(batch_size)
        loss = self.Q.train_on_batch(inputs, targets)
        return loss

    def save(self, fname):
        self.Q.save_weights(fname)

    def load(self, fname):
        self.Q.load_weights(fname)
        print(self.Q.get_weights())
```

You'll see that this is quite similar to the previous Q-learning agent we implemented. There are explore and discount values, for example. But the Q function is now a neural network.

The biggest difference are these `remember` and `replay` methods.

A challenge with DQNs is that they can be unstable - in particular, they exhibit a problem known as _catastrophic forgetting_ in which later experiences overwrite earlier ones. When this happens, the agent is unable to take full advantage of everything it's learned, only what it's learned most recently.

A method to deal with this is called _experience replay_. We just store experienced states and their resulting rewards (as "memories"), then between actions we sample a batch of these memories (this is what the `_prep_batch` method does) and use them to train the neural network (i.e. "replay" these remembered experiences). This will become clearer in the code below, where we actually train the agent.

```python
import os
import sys
from time import sleep
game = Game()
agent = Agent(game)

print('training...')
epochs = 10000
batch_size = 256

# keep track of past record_len results
record_len = 100
record = deque([], record_len)

for i in range(epochs):
    game.reset()
    reward = 0
    loss = 0
    # rewards only given at end of game
    while reward == 0:
        prev_state = game.state
        action = agent.choose_action()
        game.move(action)
        reward = game.update()
        new_state = game.state

        # debug, "this should never happen"
        assert not np.array_equal(new_state, prev_state)

        agent.remember(prev_state, action, new_state, reward)
        loss += agent.replay(batch_size)

    sys.stdout.flush()
    sys.stdout.write('epoch: {:04d}/{} | loss: {:.3f} | win rate: {:.3f}\r'.format(i+1, epochs, loss, sum(record)/len(record) if record else 0))

    record.append(reward if reward == 1 else 0)

agent.save(fname)
```

Here we train the agent for 10000 epochs (that is, 10000 games). We also keep a trailing record of its wins to see if its win rate is improving.

A game goes on until the reward is non-zero, which means the agent has either lost (reward of -1) or won (reward of +1). Note that between each action the agent "remembers" the states and reward it just saw, as well as the action it took. Then it "replays" past experiences to update its neural network.

Once the agent is trained, we can play a round and see if it wins.

```python
# play a round
game.reset()
game.render()
reward = 0
while reward == 0:
    action = agent.choose_action()
    game.move(action)
    reward = game.update()
    game.render()
    sleep(0.1)
print('winner!' if reward == 1 else 'loser!')
```

After 10,000 epochs, the agent I trained won about 90% of the time. Not bad from the 30% or so it started at!