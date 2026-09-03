from nuscenes.utils import splits
import numpy as np

train = np.array(splits.train_detect)
idxs = np.random.randint(0, len(train), 70)
train = train[idxs]
print(', '.join('"{0}"'.format(x) for x in train))
