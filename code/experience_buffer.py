import numpy as np
import random


class ExperienceBuffer(object):
    """
    For experience replay
    based on implementation from https://github.com/awjuliani/DeepRL-Agents/blob/master/Deep-Recurrent-Q-Network.ipynb
    Extended with sample_in_order.
    """
    def __init__(self, buffer_sz=100):
        self.buffer = []
        self.buffer_sz = buffer_sz # max size of buffer
        self.cur_episode_index = 0
        self.max_episode_length = 0


    def add_episode(self, episode):
        if len(self.buffer) + 1 > self.buffer_sz:
            del self.buffer[0]
        self.buffer.append(episode)
        self.max_episode_length = max(self.max_episode_length, len(episode))
        assert (len(self.buffer) <= self.buffer_sz), "buffer too big"


    def sample(self, batch_sz, trace_length=-1):
        sampled_episodes = random.sample(self.buffer, batch_sz)
        return self._get_traces_from_episodes(sampled_episodes, trace_length)


    def sample_in_order(self, batch_sz, trace_length=-1):
        """
        This function can be used to go through all episodes in experience buffer in order, e.g. if you want to make sure
        that all episodes have been seen.
        Keeps track of the index of the next episode. If samples go over end of the buffer, it wraps around.
        :param batch_sz: number of traces to return
        :param trace_length: max length of each trace
        :return:
        """
        assert (batch_sz <= len(self.buffer)), "batch size is larger than experience buffer. "
        sampled_episodes = []
        if self.cur_episode_index + batch_sz <= len(self.buffer):
            sampled_episodes = self.buffer[self.cur_episode_index:self.cur_episode_index + batch_sz]
            self.cur_episode_index += batch_sz
        else:
            overhang = self.cur_episode_index + batch_sz - len(self.buffer) # number of samples to wrap around
            sampled_episodes = self.buffer[self.cur_episode_index:] + self.buffer[0:overhang]
            self.cur_episode_index = overhang
        return self._get_traces_from_episodes(sampled_episodes, trace_length)


    def _get_traces_from_episodes(self, sampled_episodes, trace_length=-1):
        """

        :param sampled_episodes:
        :param trace_length:  if -1, then return full episodes.
        :return:
        """
        sampled_traces = []
        if trace_length != -1:
            for episode in sampled_episodes:
            # choose a random point within the episode to start the trace
                point = np.random.randint(0, len(episode) + 1 - trace_length)
                sampled_traces.append(episode[point:point + trace_length])
        else:
            sampled_traces = sampled_episodes
        sampled_traces = np.array(sampled_traces)
        return sampled_traces