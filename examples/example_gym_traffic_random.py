import gym
import gym_traffic
from gym.wrappers import Monitor
import gym
import time
from tqdm import tqdm
monitor = False
#env = gym.make('Traffic-Simple-gui-v0')
#env = gym.make('Traffic-Simple-cli-v0')
env = gym.make('Traffic-DCMed-gui-v0')
#env = gym.make('Traffic-2way-gui-v0')
#env = gym.make('Traffic-litteRiver-gui-v0')
#env = gym.make('Traffic-yIntersection-gui-v0')
#env = gym.make('Traffic-Simple-gui-v0')
if monitor:
    env = Monitor(env, "output/traffic/simple/random", force=True)
for i_episode in tqdm(range(500)):
    observation = env.reset()
    for t in tqdm(range(1000)):
        #env.render()
        #print(observation)
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
