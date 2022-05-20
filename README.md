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
- What are we going to make hyperparameters? Attentive vs non attentive, number of layers, maybe randomness seed?? 

## Questions for TA May 12th Meeting 1
- We maybe want to compare our implementation to the MaskGAN model. In case you are not aware of this paper, in summary they state that the discrete nature of language makes it infeasable to propagate the gradients from the discriminator to the generator, and thus train the generator using reinforcement learning. Although, the MaskGAN and our ANP would in principle be comparable, I was thinking that we can also use a standard GAN (without reinforcement learning), and use some discrete approximation covered in the Advance Approximate Inference for Deep Latent Variable Models module. This would mean that there is a lot more work (and experimenting) to do, but I think it will be a more interesting comparison. Do you agree, and do you think it is feasable?

## Updates and Questions for TA May 20th Meething 2
- Compare ANP to NP and Transformers (GPT). 
- (TIJMEN) Cross attention is having some difficulties. The queries and keys do not have the same shape which is however necessary. 
- (VELI) We still have to come up with a sample method for our ANP/NP architecture. 
- (VELI) Metrics: given the time frame we feel like we can at least achieve implementing the perplexity metrics from the paper to use as metric. 
- Exploration methods. (only attentive vs non attentive) want to alter what? hyperparameters? 
- We are going to work with a smaller dataset than the original one. We were using wikitext 103 and are going to switch to a "huggingface" preprocessed version of wikipedia articles dataset? 
- Thing we still need to do: half of the points are context and half of the points become target. It works, but we cannot do this, because if we sample from this, then we need to create our own context and target sets. MAKE A SAMPLE FUNCTION?? 
- Do we need to make a anonymous GitHub repository when we upload the report? 
- ------------------------------------------------------------------------------------------------------------------- 
- We finished the draft version for the Introductio. As of now we are still working on the Related work and Approach section. We are aiming to have it finished this week. 
- We managed to implement (original) Neural Processes, loss goes down yahy! Now we are adding attention as well. As of now training takes 1 epoch takes 13 minutes with a very basic NLP.

## Takeaways of meeting 2
- Review Submission on Monday 30th
- Final submission on Monday June 6th 
- 2 days to review Tuesday and wednesday: deadline on Wednesday June 1st. 
- Poster presentation on the Thursday. NOT GRADED.
- Updated the TAs:
- cross attention typically: project queries keys and values to same dimension: either with a matrix or MLP. 
- Discussion on MLM vs SC. Could be but than the MLM part needs to have an indication of when to stop. 
- Atypical to model ... embeddings instead of discrete space. STICK WITH CATEGORICAL distribution. 
- related work: focus on architecture of NP and ANP and then go on with what has been done with masked language modelling: VAEs, ...
- NP statistical paradigm approach.. However, transformers are not entirely relevant. 
- ANP wuould be better due to differences between domains of wikipedia articles. 
- Compare convergence speed of ANP and NP.... 
