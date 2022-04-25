# SliceGAN Revisited - Carbon Fibers

This project revolved around reproducing [this paper](https://doi.org/10.1038/s42256-021-00322-1) presented in Nature Machine Intelligence. The original code repository can be found [here](https://github.com/stke9/SliceGAN).

The main idea is to generate microstructures with a GAN (*Generative Adverserial Network*). The original paper tried the GAN on different data, whereas we only looked at carbon fibers. The project involved some data pre-processing, and we also axtended the algorithm a bit. One extension was the inclusion of a PINN (*Physically Informed Neural Networks*) term in the loss function. We also examined the possibility of sampling the noise seeds from different random distributions. Furthermore some hyperparameter testing was done. More details can be read in the wiki of the repository, or in the poster pdf file.

