"""

Author: Peiyong

The source file is from Morvan

main call.

Using:
Tensorflow: r1.5 -gpu
"""


from j5_maze_env import Maze
from j5_RL_brain import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        #print('-'*10)
        #print(observation)
        #print('-'*10)

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            #print('-'*10)
            #print(action)
            #print('-'*10)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            #print('-'*10)
            #print(reward)
            #print('-'*10)


            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()

    print(env.n_actions)
    print(env.n_features)


    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()


    # END
