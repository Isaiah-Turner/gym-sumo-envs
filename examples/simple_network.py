"""
Q Learning:
https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe

Simple:
https://arxiv.org/pdf/1611.01142.pdf
https://medium.com/coinmonks/build-your-first-ai-game-bot-using-openai-gym-keras-tensorflow-in-python-50a4d4296687
"""
import keras
import gym
import gym_traffic
from gym.wrappers import Monitor
import gym
import time
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense

def collect_data():
    initial_games = 10000
    score_requirement = {'simple': 4500}
    training_data = []
    accepted_scores = []
    categories = [len(light.actions) for light in env.env.lights]
    t = [light.actions for light in env.env.lights]
    print(t)
    print(categories)
    for game_index in tqdm(range(initial_games)):
        score = 0
        game_memory = []
        previous_observation = []
        for step_index in range(100):
            action = env.env.action_space.sample()
            #print(action)
            observation, reward, done, info = env.env.step(action)

            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])

            previous_observation = observation
            score += reward
            if done:
                break
        tqdm.write("Score: %i " % score)
        if score >= score_requirement['simple']:
            accepted_scores.append(score)
            for data in game_memory:
                one_hot = []
                for light_index, selected_action in enumerate(data[1]):
                    one_hot.append([0 if i != selected_action else 1 for i in range(categories[light_index])])
                #print(one_hot)
                training_data.append([data[0], one_hot])

        env.reset()

    print(accepted_scores)

    return training_data


monitor = False
#env = gym.make('Traffic-Simple-gui-v0')
env = gym.make('Traffic-Simple-cli-v0')
#env = gym.make('Traffic-DCMed-gui-v0')
#env = gym.make('Traffic-2way-gui-v0')
#env = gym.make('Traffic-litteRiver-gui-v0')
#env = gym.make('Traffic-yIntersection-gui-v0')
#env = gym.make('Traffic-Simple-gui-v0')

training_data = collect_data()
model = Sequential()
model.add(Dense(64, input_dim=2, activation='sigmoid'))
model.add(Dense(sum(env.action_space.nvec), activation='linear'))
if monitor:
    env = Monitor(env, "output/traffic/simple/random", force=True)
for i_episode in tqdm(range(500)):
    observation = env.reset()
    for t in tqdm(range(1000)):

        cars = env.get_cars_in_lanes()
        current_state = env.get_light_actions()
        action = env.env.action_space.sample()  # two envs are needed. The first is a time limited wrapper, the second is the actual env.
        #time.sleep(1)
        observation, reward, done, info = env.step(action)
        print("Observation: ", end="")
        print(observation[0], end="    ")
        print(observation[1])
        print("Reward: ", end="")
        print(reward)
        print("Done: ", end="")
        print(done)
        print("Info: ", end="")
        print(info)
        print("-------------------------------------------------")
        #print "Reward: {}".format(reward)
        if done:
            #print("Episode finished after {} timesteps".format(t+1))
            break

