import numpy as np

class normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range

        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)

        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)

        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
    
    def update(self, v):
        v = v.reshape(-1, self.size)
        self.local_sum += v.sum(axis=0)
        self.local_sumsq += (np.square(v)).sum(axis=0)
        self.local_count[0] += v.shape[0]
            
    def recompute_stats(self):
        self.total_sum += self.local_sum
        self.total_sumsq += self.local_sumsq
        self.total_count += self.local_count

        # reset
        self.local_count[...] = 0
        self.local_sum[...] = 0
        self.local_sumsq[...] = 0
        
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))

    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        output = np.clip((v - self.mean) / (self.std), -clip_range, clip_range)
        return output
