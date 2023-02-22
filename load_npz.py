# Copyright (C) 2023 Suraj Kothari - All Rights Reserved

import numpy as np

def load_npz_data(data_dir):
    """ Loads data from a numpy .npz dataset """
    dataset = np.load(data_dir)
    images, poses, FOCAL = dataset["images"], dataset["poses"], dataset["focal"]

    NUM_IMAGES = images.shape[0]
    HEIGHT, WIDTH = images.shape[1:3]

    num_train = int(NUM_IMAGES*0.8)
    num_val = num_train + int(NUM_IMAGES*0.1)
    num_test = num_val + int(NUM_IMAGES*0.1)

    indexs = np.arange(NUM_IMAGES)
    np.random.shuffle(indexs)  # Random shuffle indexs in-place

    # Select train, val, test from shuffled indexs
    i_split = [indexs[:num_train], indexs[num_train:num_val], indexs[num_val:num_test]]

    # Render poses are a sample of original poses
    NUM_RENDER_POSES = 50
    STEP = int(NUM_IMAGES / NUM_RENDER_POSES)  # Round up to next integer
    render_poses = poses[::STEP,...]  # Select every <STEP> poses

    return images, poses, render_poses, [HEIGHT, WIDTH, FOCAL], i_split
