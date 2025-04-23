# disRNN
Welcome to my repo containing a Flax implementation for the paper [Cognitive Model Discovery via Disentangled RNNs](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c194ced51c857ec2c1928b02250e0ac8-Abstract-Conference.html) by Miller et. al. 2024. 

My motivation for building this is essentially to understand the model in granular detail and to learn Flax at the same time. I hope to use the implementation in the near future for a project on human behavioral data. 

Note the following
- The repo is still work in progress
- I rely on the new [Flax library](https://flax.readthedocs.io/en/latest/), *not* Flax linen
- The Flax documentation is still very sparse, so at times, the implementation may not be fully native to Flax; will update as new examples and documentation come along
- I have only worked on the Q-learning dataset

One thing I am exploring currently is how to make the model less computationally expensive. It was originally trained on Google's TPU, for which training still lasted hours, so training on my institutions limited GPUs is quite inefficient.

You can use `train_haiku.py` to run the model that was trained by the original authors, though this file is only a minimum viable implementation that I used for comparision. Use `example.py` in the original repo for the true original implementation.

# Installation
Upon cloning this repo, make sure to set the `--recursive-submodules` flag, as I am referencing the [original disentangled_rnns implementation](https://github.com/google-deepmind/disentangled_rnns/tree/main) as a submodule.
If you already cloned it, use `git submodule update --init --recursive`

I rely on the same environment installation as the original repo, though remember to `pip install -U flax`.


