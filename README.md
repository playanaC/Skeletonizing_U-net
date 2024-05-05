# Skeletonizing *Caenorhabditis elegans* based on U-Net architectures trained with a multi-worm low-resolution synthetic dataset
Skeletonization algorithms are used as basic methods to solve tracking problems, pose estimation, or predict animal group behavior. Traditional skeletonization techniques, based on image processing algorithms, are very sensitive to the shapes of the connected components in the initial segmented image, especially when these are low-resolution images. Currently, neural networks are an alternative providing more robust results in the presence of image-based noise. However, training a deep neural network requires a very large and balanced dataset, which is sometimes too expensive or impossible to obtain. This work proposes a new training method based on a custom-generated dataset with a synthetic image simulator. This training method was applied to different U-Net neural networks architectures to solve the problem of skeletonization using low-resolution images of multiple *Caenorhabditis elegans* contained in Petri dishes measuring 55 mm in diameter. These U-Net models had only been trained and validated with a synthetic image; however, they were successfully tested with a dataset of real images. All the U-Net models presented a good generalization of the real dataset, endorsing the proposed learning method, and also gave good skeletonization results in the presence of image-based noise. The best U-Net model presented a significant improvement of 3.32% with respect to previous work using traditional image processing techniques.

# Simulator image pipeline
![GitHub Logo](https://github.com/playanaC/Skeletonizing_Unet/blob/main/Simulator_image%20pipeline.png)

# Proposed neural network model
![image](https://github.com/playanaC/Skeletonizing_Unet/blob/main/neural_network_model.png)


# Datasets pipeline
![GitHub Logo](https://github.com/playanaC/Skeletonizing_U-net/blob/main/Datasets%20pipeline.PNG)


# Dataset:
- The real dataset with all experiments can be downloaded from [dataset_skeletons](https://active-vision.ai2.upv.es/wp-content/uploads/2021/02/dataset_skeletons.zip).



# Image adquisition system:
- Images were captured by an [open hardware system](https://github.com/JCPuchalt/SiViS).


# References:
- Puchalt, J. C., Sánchez-Salmerón, A.-J., Martorell Guerola, P. & Genovás Martínez, S. "Active backlight for automating visual monitoring: An analysis of a lighting control technique for *Caenorhabditis elegans* cultured on standard Petri plates". PLOS ONE 14.4 (2019) [doi paper](https://doi.org/10.1371/journal.pone.0215548)

- Puchalt, J.C., Sánchez-Salmerón, A.-J., Ivorra, E. "Improving lifespan automation for *Caenorhabditis elegans* by using image processing and a post-processing adaptive data filter". Scientific Reports (2020) [doi paper](https://doi.org/10.1038/s41598-020-65619-4).

- Layana Castro Pablo E., Puchalt, J.C., Sánchez-Salmerón, A.-J. "Improving skeleton algorithm for helping *Caenorhabditis elegans* trackers". Scientific Reports (2020) [doi paper](https://doi.org/10.1038/s41598-020-79430-8).

- Layana Castro Pablo E., Puchalt, J.C., García Garví, A., Sánchez-Salmerón, A.-J. "*Caenorhabditis elegans* Multi-Tracker Based on a Modified Skeleton Algorithm". Sensors (2021) [doi paper](https://doi.org/10.3390/s21165622).

- Puchalt, J.C., Sánchez-Salmerón, A.-J., Ivorra, E., Llopis, S., Martínez, R., Martorell, P. "Small flexible automated system for monitoring *Caenorhabditis elegans* lifespan based on active vision and image processing techniques.". Scientific Reports (2021) [doi paper](https://doi.org/10.1038/s41598-021-91898-6).

# Citation:
```
@article{layana2023S,
  title={Skeletonizing Caenorhabditis elegans Based on U-Net Architectures Trained with a Multi-worm Low-Resolution Synthetic Dataset},
  author={Layana Castro, Pablo E and García Garví, Antonio and Navarro Moya, Francisco and Sánchez-Salmerón, Antonio-José},
  journal={International Journal of Computer Vision},
  volume={131},
  number={9},
  pages={2408--2424},
  year={2023},
  publisher={Springer}
}
```
