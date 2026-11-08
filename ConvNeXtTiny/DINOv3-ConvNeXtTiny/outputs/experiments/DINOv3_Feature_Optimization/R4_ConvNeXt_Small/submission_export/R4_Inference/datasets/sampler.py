import random

from torch.utils.data import Sampler


class StudyBatchSampler(Sampler):
    """Build variable-size Study batches under an approximate image-memory budget."""
    def __init__(self, dataset, image_budget, shuffle=False, seed=42):
        self.dataset = dataset
        self.image_budget = max(int(image_budget), 1)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _batches(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        batches, batch, image_count = [], [], 0
        for index in indices:
            count = self.dataset.image_counts[index]
            if batch and image_count + count > self.image_budget:
                batches.append(batch)
                batch, image_count = [], 0
            batch.append(index)
            image_count += count
            if image_count >= self.image_budget:
                batches.append(batch)
                batch, image_count = [], 0
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self):
        yield from self._batches()

    def __len__(self):
        return len(self._batches())
