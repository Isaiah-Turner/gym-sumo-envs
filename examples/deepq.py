import gym
import gym_traffic

from baselines import deepq

def main():
    env = gym.make("Traffic-yIntersection-cli-v0")
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        print_freq=10,
    )
    print("Saving model to yIntersection.pkl")
    act.save("yIntersection.pkl")

if __name__ == '__main__':
    main()
