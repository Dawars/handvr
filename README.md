## HandVR

<img src="img/hand_anim.gif?raw=true" alt="Hand shape principle components" width="200" />

HandVR aims to reduce discomfort in VR by showing personalized hand models based on the players' physical characteristics. 

Read more: https://medium.com/kitchen-budapest/personalized-hand-models-for-vr-bdf6d6f8fad3


## Table of Content
- [Hand pose space](#hand-pose-space)
- [Joint datset](#joint-dataset)
- [Tools](#tools)
    - [Manifold rendering](#manifold-rendering)
    - [Ply renderer](#ply-renderer)
    - [Web demo](#web-demo)
- [Licence](#licence)
    


### Hand pose space

This part aims to reduce the dimensionality of hand poses to 2 with Autoencoders for better interpretability and user control. 

Here is such a manifold along with the positions of the training samples in the latent space:
<img src="img/cropped_hand_manifold.png?raw=true" alt="Manifold" width="400" />
<img src="img/cropped_hand_latent.png?raw=true" alt="Latent Space" width="400" />

Implemented AutoEncoders include:
* Fully Connected AE (vanilla)
* Convolutional AE (exploits adjecent joint hierarchy)
* Variational AE
* VAE GAN

Reconstruction losses of joint angles:

<img src="img/tb_losses.png?raw=true" alt="Hand scans" />


### Joint dataset

This dataset was created by running the [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) joint detector on the [11k](https://sites.google.com/view/11khands) hand dataset and checking the results by hand.

![Hand with joint locations](img/hand.jpg?raw=true "Hand with joint locations")

### Tools
#### Manifold rendering

During training the auto encoders it is crutial to see the progress of the model. Therefore I created a tool that can render it as fast as possible.

* Using raw OpenGl (ModernGL)
* Face normals calculated in Compute Shader
* Skinning is done in one batch


The manifolds are added to Tensorboard during training for visualization

<img src="img/tb_manifolds.png?raw=true" alt="Tensorboard manifolds" />

#### Ply renderer

The MANO models are given in the PLY format, both the low poly regisrtations and the high poly scans.

<img src="img/mano_high_poly.png?raw=true" alt="Hand scans" width="400" />

![Hand registrations](img/mano_registrations.png?raw=true "Hand registrations")

#### Web demo

Interactive web demo for visualizing the **shape** parameters

<img src="https://dawars.me/mano/images/banner.png" alt="Mano viewer" />

### Licence

The MANO model is licenced under: http://mano.is.tue.mpg.de/license
