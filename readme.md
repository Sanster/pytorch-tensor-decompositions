# PyTorch Tensor Decompositions

This is an implementation of Tucker and CP decomposition of convolutional layers.
A blog post about this can be found [here](https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning).

It depends on [TensorLy](https://github.com/tensorly/tensorly) for performing tensor decompositions.

# Usage
1. Train a VGG11 model:
```bash
python3 main.py 
--net vgg11 \
--save_dir models/vgg11
```

2. Fine tune with decomposed model: 
```bash
python3 main.py 
--ckpt /path/to/trained_model.pth \
--save_dir models/vgg11_decomposed \
--decompose  \
--decompose_method tucker \
--learning_rate 0.0001
```

TODO: make fine tune decompose work

# References

- CP Decomposition for convolutional layers is described here: https://arxiv.org/abs/1412.6553
- Tucker Decomposition for convolutional layers is described here: https://arxiv.org/abs/1511.06530
- VBMF for rank selection is described here: http://www.jmlr.org/papers/volume14/nakajima13a/nakajima13a.pdf
- VBMF code was taken from here: https://github.com/CasvandenBogaard/VBMF
- Tensorly: https://github.com/tensorly/tensorly
