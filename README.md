# Attentive Neural Processes for Natural Laguage Generation

## Getting started

1. Set up conda environment by running `conda env create --file environment.yaml`
2. Download dataset: `./prepare-wikitext-103.sh`
3. Preprocess dataset by converting it to binary format: `./preprocess.sh`
4. Train Neural Process language model: `./train.sh`

## 

TODO - rename r_dim to rs_dim

## Attentive Neural Process

## MaskGAN
The MaskGAN consists of a reinforcement learning based generative network, and a regular discriminator network. The generator is trained to fill in blanks (masked language), and does this by first encoding the full masked sequence using a LSTM. Then the decoder uses the final output state of the encoder, to predict tokens for the blank tokens.

## Questions for us
- What do we want our output distribution to look like? The GAN we (maybe) want to compare to, learns a categorical distribution over de vocab, but in earlier conversations we were talking about continous prob. densities over the word embedding space.

- What dataset / task will we be traning on?
- What dataset / task will we be evaluating on?

## Questions for TA
- We maybe want to compare our implementation to the MaskGAN model. In case you are not aware of this paper, in summary they state that the discrete nature of language makes it infeasable to propagate the gradients from the discriminator to the generator, and thus train the generator using reinforcement learning. Although, the MaskGAN and our ANP would in principle be comparable, I was thinking that we can also use a standard GAN (without reinforcement learning), and use some discrete approximation covered in the Advance Approximate Inference for Deep Latent Variable Models module. This would mean that there is a lot more work (and experimenting) to do, but I think it will be a more interesting comparison. Do you agree, and do you think it is feasable?