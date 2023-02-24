# Copyright (C) 2023 Suraj Kothari - All Rights Reserved

import numpy as np

def load_npz_data(data_dir):
    """ Loads data from a numpy .npz dataset """
    dataset = np.load(data_dir)
    images, poses, FOCAL = dataset["images"], dataset["poses"], dataset["focal"]

    NUM_IMAGES = images.shape[0]
    HEIGHT, WIDTH = images.shape[1:3]

    train_index = int(NUM_IMAGES*0.8)
    val_index = train_index + int(NUM_IMAGES*0.1)
    test_index = val_index + int(NUM_IMAGES*0.1)

    indexes = np.arange(NUM_IMAGES)
    np.random.shuffle(indexes)  # Random shuffle indexes in-place

    # Select train, val, test from shuffled indexes
    i_split = [indexes[:train_index], indexes[train_index:val_index], indexes[val_index:test_index]]

    # Render poses are a sample of original poses
    NUM_RENDER_POSES = 50
    STEP = int(NUM_IMAGES / NUM_RENDER_POSES)  # Round up to next integer
    render_poses = poses[::STEP,...]  # Select every <STEP> poses

    return images, poses, render_poses, [HEIGHT, WIDTH, FOCAL], i_split
