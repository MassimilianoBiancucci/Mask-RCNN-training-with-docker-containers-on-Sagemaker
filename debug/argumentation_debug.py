import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch


aug = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(rotate=(-10, 10))
    ])

image = ia.quokka_square(size=(256, 256))
images = [np.copy(image) for _ in range(10)]

det = aug.to_deterministic()

batches = [UnnormalizedBatch(images=[img]) for img in images]

with det.pool(processes=-1, maxtasksperchild=20, seed=1) as pool:
    batches_aug = pool.map_batches(batches)

images = [b.images_aug[0] for b in batches_aug]

ia.imshow(images[0])
