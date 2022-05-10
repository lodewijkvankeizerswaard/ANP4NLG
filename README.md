# Attentive Neural Processes for Natural Laguage Generation

## 

TODO - rename r_dim to rs_dim

## Questions for TA
- We maybe want to compare our implementation to the MaskGAN model. In case you are not aware of this paper, in summary they state that the discrete nature of language makes it infeasable to propagate the gradients from the discriminator to the generator, and thus train the generator using reinforcement learning. Although, the MaskGAN and our ANP would in principle be comparable, I was thinking that we can also use a standard GAN (without reinforcement learning), and use some discrete approximation covered in the Advance Approximate Inference for Deep Latent Variable Models module. This would mean that there is a lot more work (and experimenting) to do, but I think it will be a more interesting comparison. Do you agree, and do you think it is feasable?