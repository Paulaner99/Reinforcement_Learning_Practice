import cv2
import gym
import gym.spaces
import numpy as np 
import collections
import matplotlib.pyplot as plt

class FireResetEnv(gym.Wrapper):
    """
    Many Atary environments have a FIRE button to press to start the game.

    This Wrapper makes sure to press that button to start a new game.
    """
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        """
        Reset the environment and press that button to start the game.

        Returns:
            - the first observation of the game
        """
        self.env.reset()
        obs, _, done, _ = self.env.step(1)

        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)

        if done:
            self.env.reset()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    """
    This Wrapper perform the same action for several consecutive frames.

    In addition, it reduces the flickering effects of the environment.
    """
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """
        Perform the same action for several consecutive frames.

        Returns:
            - maximum of every pixel in the last two frames
            - the reward
            - termination signal
            - additional info
        """
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
            
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """
        Clear the observation buffer and restart the environment.

        Return:
            - the first observation of the game
        """
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84,84,1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        """
        Reduces the resolution to 84x84 pixels and converts the frame to grayscale.

        Returns:
            - processed frame
        """
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32) 
        else:
            assert False, "Unknown resolution."

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(img, (84,110), interpolation=cv2.INTER_AREA)

        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

class BufferWrapper(gym.ObservationWrapper):
    """
    This Wrapper creates a stack of subsequent frames along the first dimension and returns them as an observation.
    """
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(low=old_space.low.repeat(n_steps, axis=0),
                                                high=old_space.high.repeat(n_steps, axis=0),
                                                dtype=dtype)

    def reset(self):
        """
        Resets the observation by initializing the buffer with black frames.

        Returns:
           - observation
        """
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        """
        Stacks multiple frames to create an observation.

        Returns:
            - observation
        """
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

class ImageToPyTorch(gym.ObservationWrapper):
    """
    This Wrapper changes the shape of the observation from HWC to CHW that is the format required by PyTorch.
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Convert ints to floats and scales every pixel value from [0...255] to [0.0...1.0].
    """
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return env
