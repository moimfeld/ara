## How to compile

### with vectorized kernel
```
ENV_DEFINES="-D<model> -DUSE_VEXT" make bin/custom_model
# model is one of these {TINY_FC_MODEL, FC_MODEL, CONV_MODEL, CONV_PAD_MODEL, CONV_POOL_MODEL}

example:
ENV_DEFINES="-DCONV_MODEL -DUSE_VEXT" make bin/custom_model
```

### with measurements
```
ENV_DEFINES="-D<model> -DUSE_VEXT -DMEASURE" make bin/custom_model
# model is one of these {TINY_FC_MODEL, FC_MODEL, CONV_MODEL, CONV_PAD_MODEL, CONV_POOL_MODEL}

example:
ENV_DEFINES="-DCONV_MODEL -DUSE_VEXT" make bin/custom_model
```