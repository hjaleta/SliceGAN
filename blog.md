# Group 81 blog post
# Generating three-dimensional structures from a two-dimensional slice with generative adversarial network-based dimensionality expansion

J. IJpma, R. Jense, H. Lindstedt and A. Sharma

## Introduction

Material science is a vast topic of research today, and it’s not very surprising - Almost every product you can think of is made out of some material! One family of materials are the so called composites. Composites are materials that on a microscopical level consist of 2 or more different “pure” materials. One of the more famous composites is carbon fiber materials. Here, fibers of carbon are embedded into a plastic polymer, similar to how concrete can be reinforced with steel bars. The fibers give the material strength, whereas the polymer provides the structure. Carbon fiber materials are used in a vast range of applications, like airplanes and bikes. When studying the properties of these materials, simulations are often used. However, generating the desired 3D structures can be computationally costly. This is where the paper we worked with in this project comes into the picture.

## Original Paper
The paper Generating three-dimensional structures from a two-dimensional slice with generative adversarial network-based dimensionality expansion was published in Nature Machine Intelligence in April 2021. In this paper, Steven Kench and Samuel J. Cooper introduced SliceGAN, a GAN that generates 3D micro structures. In general, obtaining 3D training data can be hard. To build a full 3D volume can take a long time with for example spectroscopy. Furthermore, when training a GAN with 3D data the memory consumption can quickly grow large. It is therefore often more convenient to work with 2D images. 

The main purpose of SliceGAN is to resolve this dimensional incompatibility between training data and generated data. This is done by using a generator that produces 3D volumes, but discriminators which classify 2D images. It does this by slicing out images from the generated volume perpendicular to the x-, y- or z-axis. These slices are then passed to three Discriminators. Each of the Discriminators tries to distinguish real and generated slices corresponding to one of the different directions in the material. In case the microstructure is isotropic (meaning it looks the same from all three directions), one Discriminator suffices. A Wasserstein loss is used during training. In the paper, many different materials are tested, like battery cathodes and ceramics. In this project however, we focus on the aforementioned carbon fibers.

## Hyperparameter Tuning
![Figure 1](figures/Graphs_disc_loss_real_fake_hp_tuning.png?raw=true)

## References

[1]  S. Kench and S. J. Cooper, Generating three-dimensional structures from a
two-dimensional slice with generative adversarial
network-based dimensionality expansion

