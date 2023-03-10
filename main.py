import gym
from procgen import ProcgenEnv, ProcgenGym3Env
import numpy as np
from svd_shard.agent import Agent
from svd_shard.utils import plot_learning_curve
import json 

def main(tpu=False):
    # env = gym.make('CartPole-v1')

    # create env to get features from, like action space
    env = ProcgenEnv(num_envs=1, env_name="coinrun", distribution_mode="easy", use_backgrounds=True) 
    # envs = ProcgenGym3Env(num=1, env_name="coinrun")
    # env = gym.make()

    gpu = False
    N = 20 # learn every N steps
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    input_space = env.observation_space['rgb'].shape

    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_space=input_space, tpu=tpu)

    # agent.actor.summary() #cannot do this because model is callable and not compiled

    n_games = 100

    # figure_file = 'plots/cartpole.png'
    figure_file = 'plots/coinrun_v0.1.png'


    best_score = 0
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    actions = [] 
    rewards = [] 

    for i in range(n_games):
        # to change shape (1, 64, 64, 3) -> (64, 64, 3) #TODO should probs change to tf.squeeze
        observation = env.reset()['rgb'][0] 
        # print(type(observation))
        # print(len(observation))
        # print(observation)
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            # print(action)
            actions.append(float(action))
             #NOTE: first _ i think captures truncated? But do not know
            # print("äction", action)
            # print(type(action), action.shape)
            #NOTE: info not used
            observation_, reward, done, info= env.step(action)
            observation_ = observation_['rgb'][0] 
            rewards.append(float(reward))
            # result = env.step(action)
            # print(f"\n\n Env step: {result}")
            # print(len(result))
            
            n_steps += 1
            score += reward
            agent.store_transition(observation, action,
                                   prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_

        score_history.append(score)
        if len(score_history) >= 10:
            avg_score = np.mean(score_history[-10:])
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

    to_save = {'actions': actions, 'rewards': rewards}
    # Use json.dumps to convert the dictionary to a JSON string
    json_data = json.dumps(to_save)

    # Write the JSON string to a file
    with open("data.json", "w") as f:
        f.write(json_data)


if __name__ == "__main__":
    main()