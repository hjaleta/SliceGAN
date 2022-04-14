# Group 81 blog post
# Generating three-dimensional structures from a two-dimensional slice with generative adversarial network-based dimensionality expansion

J. IJpma, R. Jense, H. Lindstedt and A. Sharma

## Introduction

Material science is a vast topic of research today, and it’s not very surprising - Almost every product you can think of is made out of some material! One family of materials are the so called composites. Composites are materials that on a microscopical level consist of 2 or more different “pure” materials. One of the more famous composites is carbon fiber materials. Here, fibers of carbon are embedded into a plastic polymer, similar to how concrete can be reinforced with steel bars. The fibers give the material strength, whereas the polymer provides the structure. Carbon fiber materials are used in a vast range of applications, like airplanes and bikes. When studying the properties of these materials, simulations are often used. However, generating the desired 3D structures can be computationally costly. This is where the paper we worked with in this project comes into the picture.

![Figure 1](figures/Carbon_fibre_usage.png?raw=true)

## Original Paper
The paper Generating three-dimensional structures from a two-dimensional slice with generative adversarial network-based dimensionality expansion was published in Nature Machine Intelligence in April 2021. In this paper, Steven Kench and Samuel J. Cooper introduced SliceGAN, a GAN that generates 3D micro-structures from 2D training slices. In general, obtaining 3D training data can be hard. To build a full 3D volume can take a long time with for example spectroscopy. Furthermore, when training a GAN with 3D data the memory consumption can quickly grow large. It is therefore often more convenient to work with 2D images. 

The main purpose of SliceGAN is to resolve this dimensional incompatibility between training data and generated data. This is done by using a generator that produces 3D volumes, but discriminators which classify 2D images. It does this by slicing out images from the generated volume perpendicular to the x-, y- or z-axis. These slices are then passed to three Discriminators. Each of the Discriminators tries to distinguish real and generated slices corresponding to one of the different directions in the material. In case the microstructure is isotropic (meaning it looks the same from all three directions), one Discriminator suffices. A Wasserstein loss is used during training. In the paper, many different materials are tested, like battery cathodes and ceramics. In this project however, we focus on the aforementioned carbon fibers.

## Source Data
The source data is obtained from the work of Emerson et al. [2]. 
We encountered a lot of roadblocks while trying to obtain a data sample to train our network on. This was mostly due to a general scarcity of 3D data available online, especially for our case. We crop the original dataset perpendicular to the x-axis which displays the circular cross-section of these fibers such that the entire slice contains the fiber cross-sections in order to save memory during the implementation of our version of SliceGAN.

![Figure 2](figures/data.png?raw=true)

## Preprocessing
The first step to training any kind of model is applying transformations to the raw data set such that it can be accepted as input by the model. For SliceGAN this process involves trimming the raw data, processing it into a binary format and applying some cleanup on small artifacts within the image.

### Slicing Images. 
More often than not a model is not trained on a full image, but rather on an area that carries more relevance for its use case. This, together with scaling the image, leads to a massive performance boost as the complexity of the eventual input is significantly reduced. The raw input is transformed into a set of slices.

### Image Labeling. 
The two-dimensional slice carries no contextual information yet. The slice is transformed to its respective binary file type in a One Hot Encoding. It is nearly ready to be used for generating structures. 

### Artifact Cleaning. 
The image is pushed through a small image processing pipeline. First, a morphological closing is applied to close gaps between the borders of our structure slice. Second, some noise on the background is removed as this should not contribute to the output. Lastly, some small holes in the objects are closed up to ensure the volume is solid.

Every raw image has now been processed into a square and denoised image and is suitable for training. The final training data, viewed from different angles, can be seen below.
![Figure 3](figures/slices.png?raw=true)

## CircleNet

An interesting field within Deep Learning is Physically Informed Neural Networks, or PINN. These networks aim to model some physical behavior by incorporating a special term in their loss function, with the aim of better capturing some underlying laws of the problem we are facing. 
        
In our case, with the help of Professor Dr. Bariș Çağlar (from the faculty of Aerospace Engineering at TU Delft), we recognised that one physical property that is desired in this spatial generation of carbon fibers is their circularity. We note that an issue with SliceGan producing carbon structures is that if we observe the generated structure along the x-direction, the circularity of generated cross-sections is less than that of our (/distorted as compared to our) original data.

Therefore, we decide to use a custom loss function which enforces a loss pertaining to the circularity of the fibers.
        
For implementing this feature in our network, we add a custom loss term in the generator training, aptly dubbed- Circularity Loss. This loss term ensures that the number of circles with circularity > circularity_threshold for each sub-image, is of a similar order when comparing our real and generated data. 
Now, how is this loss generated?
Since, this loss needs to “flow” back (undergo some differentiation during backpropagation), we decide to train a neural network (hereon referred to as CircleNet or CNet) which simply aims to calculate the number of circles (within the desired area value and circularity value ranges). 
        
To ensure that CircleNet is tailored exclusively for our situation, we train it on the real carbon data (which we aim to recreate via the generator) by computing ground truth labels with the help of SimpleBlobDetector- an opencv library which can detect blobs in an image using various filters such as area, convexity, circularity et cetera. These ground truth labels are then compared with what our CNet products during training. The loss (for training our CircleNet) is calculated by taking a mean squared error of the number of circles predicted (by the CircleNet) vs the number of circles detected (by the SimpleBlobDetector). Towards the end of our training phase of the CircleNet, we save the weights for re-use in the GAN later (also to avoid the tragedy and catastrophe of having to train everything again just because the Google colab kernel decided to time-out).

This trained CircleNet (the architecture of which has been modeled based over our Discriminator networks including ) is then loaded and used to provide a loss term during the GAN training to the Generator network. This loss term (which is referred to as the custom loss function above) is calculated by taking a mean squared error of the number of circles predicted on the real data (by the CircleNet) vs the number of circles predicted on the fake data (again by the CircleNet).
Notice that unlike the loss used in Circle Training, the “Circularity Loss” used in GAN training uses the prediction of the CircleNet on the real data as the ground truth label to be compared against the prediction made on the generated (fake) data. This is formulated this way by design to ensure that a gradient may be calculated using the grad_fn functionality available in the PyTorch backend and backpropagation may take place effectively. 


One issue we faced while implementing this is that the real data our network is trained on also contains a lot of artifacts and distortions that affects our learning capability for  modular nodes with high circularity. The biggest problem here is that the subsections of these structures contain a lot of small noise artifacts and a lot of our circular fiber cross-sections are connected to each other due to some inherent error in the conversion from grayscale to binary. We apply water-shedding to mitigate these issues and compare our results of the CircleNet before and after water-shedding to compare the efficacy of our pre-processing measure. 

## A Closer Look on the Forward Pass
One thing discussed in the original paper is the noise seed z that we pass to the generator. In the paper, the released code, and all our training instances, a 16 * 4 * 4 * 4 array with random noise was used. The first dimension involves different channels, and the last 3 are spatial. Multiple transpose convolutions are used to go from a spatial extension of 4*4*4 to 64*64*64. These operations are, as normal convolutions, dependent on stride, kernel size and padding, s, k & p. (For more exact details on this, please refer to the original paper.)
When choosing s < k we impose kernel overlap. This means that the “influential fields” emanating from the different seeds overlap with each other. One can think of it as the opposite of a receptive field. This property is important for the continuity properties of generated fibers. If we instead had k = s, all seeds would have an independent contribution to the volume. One can think of it all seeds generating a 16^3 cube, which we then glue together. Now, when there is overlap, we find better continuity properties.
Another cool feature that comes with this way of learning continuity, is that we can make bigger forward passes than the model was trained on. When giving a noise seed of different size, for example 16 *9^3 we generate a volume of size 224^3! This can be compared with the original size 64^3 which the network was trained to generate. Both can be seen below. Note that the 64^3 is much more zoomed in.
![Figure 4](figures/forward_pass_slices.png?raw=true)
![Figure 5](figures/forward_pass_slices2.png?raw=true)

## Hyperparameter Tuning

SliceGAN has many hyperparameters, from the amount of layers, the type of loss function used to the learning rate for either the Generator or one of the Discriminators. In this blogpost we focus on three of them: the beta1 and beta2 parameters for the adam optimizer and the noise distributions which are used as seed for the Generator.  
The reason for looking further into the beta1 and beta2 values for the Adam optimizer is that the implementation from the original paper uses 0 as value for beta1. This means that the network is not using the bias in the  first moment estimate. This is remarkable since the default is to use .9 for beta1 and .99 for beta2 as recommended by the paper that introduced Adam. [reference]
We trained the GAN with beta1 values [0, 0.2, 0.5, 0.8, 0.9] keeping beta2 fixed at 0.9 (as was used in the paper) and for beta2 we used the values [0.1, 0.3, 0.5, 0.9] keeping beta1 fixed as 0.

Figure x depicts the Discriminator loss for the real and generated samples for each of the beta values. The graph shows the average of every thirty samples for clarity, since the original losses are too noisy to make a clear comparison.

Figure y shows the Wasserstein loss of the network for the different beta1 and beta2 values. The graphs suggest that especially for beta2 lower values might be better, seeing as they result in a lower loss. However, since the network was only trained for 10 epochs it might be that the higher values of beta2 result in better performance after longer training runs. 

![Figure 6](figures/Graphs_disc_loss_real_fake_hp_tuning.png?raw=true)
![Figure 6](figures/beta12_wass_Loss_Graph.png?raw=true)
## References

[1]  Kench, S., Cooper, S.J. Generating three-dimensional structures from a two-dimensional slice with generative adversarial network-based dimensionality expansion. Nat Mach Intell 3, 299–305 (2021), https://doi.org/10.1038/s42256-021-00322-1

[2] Monica J. Emerson, Vedrana A. Dahl, Knut Conradsen, Lars P. Mikkelsen, Anders B. Dahl,
A multimodal data-set of a unidirectional glass fibre reinforced polymer composite,
Volume 18, 2018, Pages 1388-1393, https://doi.org/10.1016/j.dib.2018.04.039.

[3] Diederik P. Kingma and Jimmy Lei Ba,  “ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION”, Arxiv 2014, https://arxiv.org/pdf/1412.6980.pdf, https://doi.org/10.48550/ARXIV.1412.6980

