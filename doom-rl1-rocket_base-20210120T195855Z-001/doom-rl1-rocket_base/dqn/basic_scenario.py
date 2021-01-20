import random

import skimage as skimage
from skimage import transform, color, exposure
from skimage.viewer import ImageViewer
import random
from random import choice
import numpy as np
from collections import deque
import time

import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D, Input, AveragePooling2D, Lambda, Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop
from keras import backend as K

import itertools as it
from time import sleep
import tensorflow as tf
import click
import tensorflow as tf
from vizdoom import *

from dqn import Network

# Network parameters
INPUT_SIZE = np.array([84, 84, 1])
LEARNING_RATE = 0.0002

# Training parameters
TOTAL_EPISODES = 500
MAX_STEPS = 100
BATCH_SIZE = 64
MEMORY_SIZE = 1000000

# Exploring parameters
EPSILON_START = 1.0
EPSILON_STOP = 0.01

# Reinforcement problem parameters
GAMMA = 0.99  # Discounted reward

#Select which environment
config = "../scenarios/rocket_basic.cfg"
scene = "../scenarios/rocket_basic.wad"


class DoubleDQNAgent:

    def __init__(self, state_size, action_size):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 32
        self.observe = 5000
        self.explore = 50000 
        self.frame_per_action = 4
        self.update_target_freq = 3000 
        self.timestep_per_train = 100 # Number of timesteps between training interval

        # create replay memory using deque
        self.memory = deque()
        self.max_memory = 50000 # number of previous transitions to remember

        # create main model and target model
        self.model = None
        self.target_model = None

        # Performance Statistics
        self.stats_window_size= 50 # window size for computing rolling statistics
        self.mavg_score = [] # Moving Average of Survival Time
        self.var_score = [] # Variance of Survival Time
        self.mavg_ammo_left = [] # Moving Average of Ammo used
        self.mavg_kill_counts = [] # Moving Average of Kill Counts

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:
            q = self.model.predict(state)
            action_idx = np.argmax(q)
        return action_idx

    def shape_reward(self, r_t, misc, prev_misc, t):
        
        # # Check any kill count
        # if (misc[0] > prev_misc[0]):
        #     r_t = r_t + 1

        # if (misc[1] < prev_misc[1]): # Use ammo
        #     r_t = r_t - 0.1

        if (misc[0] < prev_misc[0]): # Loss HEALTH
            r_t = r_t - 0.1

        return r_t

    # Save trajectory sample <s,a,r,s'> to the replay memory
    def replay_memory(self, s_t, action_idx, r_t, s_t1, is_terminated, t):
        self.memory.append((s_t, action_idx, r_t, s_t1, is_terminated))
        if self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        if len(self.memory) > self.max_memory:
            self.memory.popleft()

        # Update the target model to be same with model
        if t % self.update_target_freq == 0:
            self.update_target_model()

    # Pick samples randomly from replay memory (with batch_size)
    def train_minibatch_replay(self):
        """
        Train on a single minibatch
        """
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros(((batch_size,) + self.state_size)) # Shape 64, img_rows, img_cols, 4
        update_target = np.zeros(((batch_size,) + self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i,:,:,:] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i,:,:,:] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input) # Shape 64, Num_Actions

        target_val = self.model.predict(update_target)
        target_val_ = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_val[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val_[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        loss = self.model.train_on_batch(update_input, target)

        return np.max(target[-1]), loss

    # Pick samples randomly from replay memory (with batch_size)
    def train_replay(self):

        num_samples = min(self.batch_size * self.timestep_per_train, len(self.memory))
        replay_samples = random.sample(self.memory, num_samples)

        update_input = np.zeros(((num_samples,) + self.state_size)) 
        update_target = np.zeros(((num_samples,) + self.state_size))
        action, reward, done = [], [], []

        for i in range(num_samples):
            update_input[i,:,:,:] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            update_target[i,:,:,:] = replay_samples[i][3]
            done.append(replay_samples[i][4])

        target = self.model.predict(update_input) 
        target_val = self.model.predict(update_target)
        target_val_ = self.target_model.predict(update_target)

        for i in range(num_samples):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_val[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val_[i][a])

        loss = self.model.fit(update_input, target, batch_size=self.batch_size, nb_epoch=1, verbose=0)

        return np.max(target[-1]), loss.history['loss']

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)

class Memory:
    """Replay buffer object"""

    def __init__(self, max_size=MEMORY_SIZE):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]


def preprocess_frame(frame):
    """Takes an RGB frame from the game and returns a normalized grayscale and resized image"""
    # Crop the screen (remove the roof because it contains no information)
    cropped_frame = frame[30:-10, 30:-30]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, INPUT_SIZE)
    return preprocessed_frame

def preprocessImg(img, size):

    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(img,size)
    img = skimage.color.rgb2gray(img)

    return img


def create_environment(visible=False):
    """Creates a basic game environment"""
    game = DoomGame()
    # Load the correct configuration
    game.load_config(config)
    # Load the scenario
    game.set_doom_scenario_path(scene)
    # Set the desired game configuration
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)  # Bullet holes and blood on the walls
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)  # Smoke and blood
    game.set_render_messages(False)  # In-game messages
    game.set_render_corpses(False)
    game.set_window_visible(visible)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.init()
    # Possible actions for this environment
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    return game, possible_actions


def create_model():
    """Creates Deep Q Network for the agent"""
    conv_dqn = tf.keras.Sequential()
    conv_dqn.add(
        tf.keras.layers.Conv2D(
            input_shape=INPUT_SIZE,
            activation="relu",
            filters=32,
            kernel_size=[8, 8],
            strides=[4, 4],
            padding="VALID",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    conv_dqn.add(
        tf.keras.layers.Conv2D(
            filters=64,
            activation="relu",
            kernel_size=[4, 4],
            strides=[2, 2],
            padding="VALID",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    conv_dqn.add(
        tf.keras.layers.Conv2D(
            filters=128,
            activation="relu",
            kernel_size=[4, 4],
            strides=[2, 2],
            padding="VALID",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    conv_dqn.add(tf.keras.layers.Flatten())
    conv_dqn.add(tf.keras.layers.Dense(units=512, activation="relu"))
    conv_dqn.add(tf.keras.layers.Dense(units=3))
    conv_dqn.compile(loss="MSE", optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    return conv_dqn


def predict_action(state, actions, model, epsilon_start, epsilon_stop, episode):
    """Linear decaying e-greedy action selection"""
    explore_th = np.random.rand()
    explore_probability = (
        epsilon_start + (epsilon_stop - epsilon_start) / TOTAL_EPISODES * episode
    )
    if explore_probability > explore_th:
        action = random.choice(actions)
    else:
        expanded = state.reshape((1, *state.shape))
        Qs = model.predict(expanded)
        choice = np.argmax(Qs)
        action = actions[int(choice)]
    return action


def preload_buffer():
    """Loads the replay buffer with random experience to start training"""
    game, possible_actions = create_environment()
    memory = Memory()
    game.new_episode()
    for _ in range(BATCH_SIZE):
        state = game.get_state().screen_buffer
        state = preprocess_frame(state)
        action = random.choice(possible_actions)
        reward = game.make_action(action)
        done = False
        if game.is_episode_finished():
            done = True
            next_state = np.zeros(INPUT_SIZE)
            game.new_episode()
        else:
            next_state = game.get_state().screen_buffer
            next_state = preprocess_frame(next_state)
        memory.add((state, action, reward, next_state, done))
    game.close()
    return memory


@click.group()
def main():
    pass


@main.command()
def test_environment():
    """Creates a test environment to check if everything is ok"""
    game, actions = create_environment(visible=True)
    game.new_episode()
    while not game.is_episode_finished():
        action = random.choice(actions)
        game.make_action(action)
    print("Total reward:", game.get_total_reward())


@main.command()
@click.option("--model-path", default=None)
def train_model(model_path):
    """Train a Deep Q Learning Agent to play the basic scenario of VizDoom"""
    game = DoomGame()

    game.load_config(config)
    # Load the scenario
    game.set_doom_scenario_path(scene)

    game.set_sound_enabled(True)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.init()

    game.new_episode()
    game_state = game.get_state()
    misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
    prev_misc = misc

    action_size = game.get_available_buttons_size()

    img_rows , img_cols = 64, 64
    # Convert image into Black and white
    img_channels = 4 # We stack 4 frames

    state_size = (img_rows, img_cols, img_channels)

    agent = DoubleDQNAgent(state_size, action_size)

    agent.model = Network.dqn(state_size, action_size, agent.learning_rate)
    agent.target_model = Network.dqn(state_size, action_size, agent.learning_rate)

    x_t = game_state.screen_buffer # 480 x 640
    x_t = preprocessImg(x_t, size=(img_rows, img_cols))
    s_t = np.stack(([x_t]*4), axis=2) # It becomes 64x64x4
    s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4

    is_terminated = game.is_episode_finished()

    # Start training
    epsilon = agent.initial_epsilon
    GAME = 0
    t = 0
    max_life = 0 # Maximum episode life (Proxy for agent performance)
    life = 0

    # Buffer to compute rolling statistics 
    life_buffer, ammo_buffer, kills_buffer = [], [], [] 

    while not game.is_episode_finished():

        loss = 0
        Q_max = 0
        r_t = 0
        a_t = np.zeros([action_size])

        # Epsilon Greedy
        action_idx  = agent.get_action(s_t)
        a_t[action_idx] = 1

        a_t = a_t.astype(int)
        game.set_action(a_t.tolist())
        skiprate = agent.frame_per_action
        game.advance_action(skiprate)

        game_state = game.get_state()  # Observe again after we take the action
        is_terminated = game.is_episode_finished()

        r_t = game.get_last_reward()  #each frame we get reward of 0.1, so 4 frames will be 0.4

        if (is_terminated):
            if (life > max_life):
                max_life = life
            GAME += 1
            life_buffer.append(life)
            # ammo_buffer.append(misc[1])
            # kills_buffer.append(misc[0])
            print ("Episode Finish ", misc)
            game.new_episode()
            game_state = game.get_state()
            misc = game_state.game_variables
            x_t1 = game_state.screen_buffer

        x_t1 = game_state.screen_buffer
        misc = game_state.game_variables

        x_t1 = preprocessImg(x_t1, size=(img_rows, img_cols))
        x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        r_t = agent.shape_reward(r_t, misc, prev_misc, t)

        if (is_terminated):
            life = 0
        else:
            life += 1

        # Update the cache
        prev_misc = misc

        # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
        agent.replay_memory(s_t, action_idx, r_t, s_t1, is_terminated, t)

        # Do the training
        if t > agent.observe and t % agent.timestep_per_train == 0:
            Q_max, loss = agent.train_replay()
            
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            print("Now we save model")
            agent.model.save_weights("ddqn.h5", overwrite=True)

        # print info
        state = ""
        if t <= agent.observe:
            state = "observe"
        elif t > agent.observe and t <= agent.observe + agent.explore:
            state = "explore"
        else:
            state = "train"

        if (is_terminated):
            print("TIME", t, "/ GAME", GAME, "/ STATE", state, \
                  "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, "/ REWARD", r_t, \
                  "/ Q_MAX %e" % np.max(Q_max), "/ LIFE", max_life, "/ LOSS", loss)

            






    # if not model_path:
    #     conv_dqn = create_model()
    # else:
    #     print("Training existent model")
    #     conv_dqn = tf.keras.models.load_model(model_path)
    # # Pre-loading the replay buffer
    # memory = preload_buffer()
    # game, actions = create_environment()
    # for episode in range(TOTAL_EPISODES):
    #     print(f"EPSIODE {episode}")
    #     step = 0
    #     done = False
    #     game.new_episode()
    #     while step < MAX_STEPS and not done:
    #         # Playing part
    #         state = game.get_state().screen_buffer
    #         state = preprocess_frame(state)
    #         action = predict_action(
    #             state, actions, conv_dqn, EPSILON_START, EPSILON_STOP, episode
    #         )
    #         reward = game.make_action(action)
    #         done = game.is_episode_finished()
    #         if done:
    #             next_state = np.zeros(INPUT_SIZE)
    #             memory.add((state, action, reward, next_state, done))
    #             print(f"Episode: {episode}", f"Total reward: {game.get_total_reward()}")
    #         else:
    #             next_state = game.get_state().screen_buffer
    #             next_state = preprocess_frame(next_state)
    #             memory.add((state, action, reward, next_state, done))
    #         step += 1
    #         # Learning part
    #         batch = memory.sample(BATCH_SIZE)
    #         learning_states = np.array([sample[0] for sample in batch])
    #         learning_actions = np.array([sample[1] for sample in batch])
    #         learning_rewards = np.array([sample[2] for sample in batch])
    #         learning_next_states = np.array([sample[3] for sample in batch])
    #         learning_dones = np.array([sample[4] for sample in batch])
    #         Qs_next_state = conv_dqn.predict(learning_next_states)
    #         # Trick for computing the loss only for the desired action
    #         target_Qs = conv_dqn.predict(learning_states)
    #         for i in range(len(batch)):
    #             learning_action = learning_actions[i]
    #             learning_action = np.argwhere(learning_action)[0][0]
    #             terminal = learning_dones[i]
    #             if terminal:
    #                 target = learning_rewards[i]
    #             else:
    #                 target = learning_rewards[i] + GAMMA * np.max(Qs_next_state[i])
    #             # Trick for computing the loss only for the desired action
    #             target_Qs[i, learning_action] = target
    #         conv_dqn.fit(learning_states, target_Qs, epochs=1)
    # game.close()
    # conv_dqn.save("/content/drive/My Drive/basic_dqn.h5")


@main.command()
@click.option("--model-path", default="basic_dqn.h5")
def test_model(model_path):
    """Test a trained agent in VizDoom's basic scenario"""
    conv_dqn = tf.keras.models.load_model(model_path)
    game, actions = create_environment(visible=True)
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state().screen_buffer
        state = preprocess_frame(state)
        action = predict_action(state, actions, conv_dqn, 0, 0, 0)
        print(action)
        reward = game.make_action(action)
        print(reward)
    print(f"Score: {game.get_total_reward()}")
    game.close()


if __name__ == "__main__":
    main()
