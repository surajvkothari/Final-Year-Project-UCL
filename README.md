# Final Year Project

This repository contains the project source code for the UCL Final Year Project (2023) titled "On Neural Radiance Fields (NeRFs) for Analysing Dark Cave Tunnels". 

It extends upon the PyTorch NeRF repository by [yenchenlin](https://github.com/yenchenlin/nerf-pytorch) by adding features relevant to this project. These features include:
* Loading a dataset stored in a custom NumPy zip
* Explore Mode
* Plotting ray opacities
* Disparity map
* Fixed-viewpoint specular reflection videos

## User Manual
This user manual will provide a guide to using the extended features of this project.

### Install Dependencies
```
pip install -r requirements.txt
```

You will also need access to a GPU and PyTorch enabled with CUDA.

### Getting the data
Our data is stored in NumPy zips (.npz) and is not available in this repository. Instead, you will need to download the datasets from this [Google Drive link](https://drive.google.com/drive/folders/1lGJcPAoUxMEKT189W4GutoEF1wCMePGY?usp=share_link).

After downloading, unzip the datasets folder and place its content (the .npz files) inside the **data** folder.

### Using a configuration
We have trained a NeRF model of the OIVIO dataset to 85,000 iterations. The same model has been applied to different resolutions of the data. When using any command, please replace <config_filename> with any of the following options:
* OIVIO_tunnel_10x_downsampled.txt
* OIVIO_tunnel_5x_downsampled.txt
* OIVIO_tunnel_2x_downsampled.txt
* OIVIO_tunnel_full_resolution.txt

**Note:** We recommend to use the "OIVIO_tunnel_10x_downsampled.txt" option for most hardware. The other options result in a better resolution output, however may result in an Out-Of-Memory error.

<details>
<summary><h3>Explore Mode</h3></summary>

Add the --explore flag,
```
python run_nerf.py --config configs/<config_filename> --explore
```

#### Navigating in Explore Mode
After running the explore mode command, you will be shown an image of view from the start of the tunnel. Use keyboard input to move or rotate around the NeRF model's reconstruction.

* `W`, `S` - Moves forwards and backwards
* `A`, `D` - Rotates the camera left and right

</details>

<details>
<summary><h3>Plotting Ray Opacities</h3></summary>

After running explore mode, move to a desired viewpoint, and the use the `SPACE` bar key which will display a plot of the opacities for the ray going through the center of the image.

**Note:** After the plot is displayed, keyboard input to move around the scene will stop working. Just close the plot window and the explore mode window. Then the explore mode window will re-display and will start accepting keyboard input again.
</details>

<details>
<summary><h3>Generating a Disparity Map</h3></summary>

After running explore mode, move to a desired viewpoint, and the use the `M` key which will display a disparity map of that view.

**Note:** After the disparity map is displayed, keyboard input to move around the scene will stop working. Just close the map window and the explore mode window. Then the explore mode window will re-display and will start accepting keyboard input again.
</details>

<details>
<summary><h3>Creating a Fixed-Viewpoint Specular Reflection Video</h3></summary>

Replace <pose_index> with the index position of the camera pose to view the specular reflection from. If unsure, use 0 to view from the start of the tunnel.
```
python run_nerf.py --config configs/<config_filename> --fixed_pose_index <pose_index>
```

**Note:** The video will be stored as an MP4 file in the config folder inside the log folder, `logs/<config_filename>`.

</details>
