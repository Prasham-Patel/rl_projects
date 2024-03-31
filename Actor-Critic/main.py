import gym
import numpy as np
from actor_critic import Agent
import matplotlib.pyplot as plt

def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    
    # plt.ylabel('Score')       
    # plt.xlabel('Game')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = Agent(alpha=1e-5, num_actions=2)
    num_games = 2000
    
    filename = "cartpole.png"
    figure_file = 'plots/' + filename
    score_history = []
    load_checkpoint = False
    best_score = 0
    
    if load_checkpoint:
        agent.load_model()
    
    for i in range(num_games):
        print("game num ", i)
        observation = env.reset()[0]
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            (observation_next, reward, done, _, info) = env.step(action)
            score += reward
            if not load_checkpoint:
                # print(observation)
                agent.learn(observation, reward, observation_next, done)
                observation = observation_next
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            if avg_score > best_score:
                best_score = avg_score
                if not load_checkpoint:
                    agent.save_model()
    
    
    x = [i+1 for i in range(num_games)]
    plotLearning(score_history, filename, window=5)
