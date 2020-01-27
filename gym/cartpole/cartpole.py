import gym
import numpy as np

from model import CartPoleModel

env = gym.make('CartPole-v1')
model = CartPoleModel()
model.load()


def run_simulation(model, learn=True, epsilon=0, verbose=0):
    history = []
    state = env.reset()
    #state = np.reshape(state, [1, 4])
    time = 0
    while True:
        env.render()
        action = model.get_action(state, epsilon=epsilon, verbose=verbose)

        state_next, reward, done, info = env.step(action)
        
        if done and time < 490:
            reward = -999
        else:
            reward -= state_next[0] ** 2

        #state_next = np.reshape(state_next, [1, 4])
        if verbose: print(reward, info)

        model.remember((state, action, reward, state_next, done))
        if learn: model.learn(batch_size=128, discount=.995)

        if done:
            print("Episode finished after {} timesteps".format(time+1))
            break
        else:
            state = state_next
            time += 1

print(env.observation_space.shape)

epsilon = .6
epsilon_decay = .99
epsilon_min = .01

episode_idx = 0
interrupted = False
learn = True
while True:
    try:
        if learn:
            print('episode =', episode_idx, 'epsilon =', epsilon)
            run_simulation(model, epsilon=epsilon, verbose=1)
            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)
            episode_idx += 1
        else:
            run_simulation(model, epsilon=0, learn=False, verbose=0)
        interrupted = False
    except KeyboardInterrupt:
        if interrupted: break
        interrupted = True
        learn = not learn
        model.save()

env.close()

