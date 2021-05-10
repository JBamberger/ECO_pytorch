from abc import ABC, abstractmethod

import torch


class SampleMemory(ABC):

    @abstractmethod
    def update_memory(self, sample):
        raise NotImplementedError()

    @abstractmethod
    def get_samples(self):
        raise NotImplementedError()


class WeightedSampleMemory(SampleMemory):

    def __init__(self, train_xf, sample_memory_size):
        """

        :param train_xf: List of training samples, shape: [1, 32, 237, 119, 2]
        """

        # Initialize first-frame training samples
        assert len(train_xf) == 1
        train_xf = train_xf[0]

        self.sample_memory_size = sample_memory_size

        num_init_samples = train_xf.size(0)

        init_sample_weights = train_xf.new_ones(1) / train_xf.shape[0]

        init_training_samples = train_xf.permute(2, 3, 0, 1, 4)

        # Sample counters and weights
        self.num_stored_samples = num_init_samples
        self.previous_replace_ind = None

        self.sample_weights = train_xf.new_zeros(self.sample_memory_size)
        self.sample_weights[:num_init_samples] = init_sample_weights

        self.training_samples = train_xf.new_zeros(
            train_xf.shape[2], train_xf.shape[3], self.sample_memory_size, self.compressed_dim, 2)

    def update_memory(self, sample_xf):
        # Update weights and get index to replace

        if self.num_stored_samples == 0:
            self.sample_weights[:] = 0
            self.sample_weights[0] = 1
            r_ind = 0
        else:
            _, r_ind = torch.min(self.sample_weights, 0)
            r_ind = r_ind.item()

            if self.previous_replace_ind is None:
                self.sample_weights /= 1 - 0.0075
                self.sample_weights[r_ind] = 0.0075
            else:
                self.sample_weights[r_ind] = self.sample_weights[self.previous_replace_ind] / (1 - 0.0075)
        self.sample_weights /= self.sample_weights.sum()

        self.previous_replace_ind = r_ind.copy()
        self.num_stored_samples += 1
        self.training_samples[:, :, r_ind:r_ind + 1, :, :] = sample_xf.permute(2, 3, 0, 1, 4)

    def get_samples(self):
        pass
