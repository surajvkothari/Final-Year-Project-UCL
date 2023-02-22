import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

# Translation on z-axis
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

# Rotation on x-axis
rot_x = lambda x : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(x),-np.sin(x),0],
    [0,np.sin(x), np.cos(x),0],
    [0,0,0,1]]).float()

# Rotation on y-axis
rot_y = lambda y : torch.Tensor([
    [np.cos(y),0,-np.sin(y),0],
    [0,1,0,0],
    [np.sin(y),0, np.cos(y),0],
    [0,0,0,1]]).float()

# Rotation on z-axis
rot_z = lambda z : torch.Tensor([
    [np.cos(z),-np.sin(z),0, 0],
    [np.sin(z),np.cos(z),0,  0],
    [0,0,1, 0],
    [0,0,0,1]]).float()


def pose_spherical(x, y, z, radius):
    c2w = trans_t(radius)
    c2w = rot_x(x/180.*np.pi) @ c2w
    c2w = rot_y(y/180.*np.pi) @ c2w
    c2w = rot_z(z/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    # Render poses for lego tractor (360-degree rotation around z-axis)
    render_poses = torch.stack([pose_spherical(x=angle, y=-30.0, z=0, radius=4.0) for angle in np.linspace(0,360,40+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    return imgs, poses, render_poses, [H, W, focal], i_split
