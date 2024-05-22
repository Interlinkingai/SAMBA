# SAMBA Model Documentation

This repository contains modules for the SAMBA model, which maps electrophysiological data to hemodynamic responses using neural network architectures.

## Module Overview

### Electrophysiological to Hemodynamic Mapping
The `SambaEleToHemo.py` script in `./model/samba_ele_to_hemo` initializes the process of HRF learning, temporal encoding (downsampling), and spatial decoding (upsampling). It leverages various modules located in `./nn/`.

### Neural Network Modules

#### Temporal Encoding
The `temporal_encoder.py` script contains the following classes:
- **PerParcelHrfLearning**: Models Hemodynamic Response Functions (HRFs) for each brain parcel by convolving neural activity with respective HRFs.
- **Differentiable_HRF**: Utilizes a double gamma function to learn HRFs, incorporating neural network parameters like response/undershoot delays, dispersions, and scales.
- **WaveletAttentionNet**: Implements wavelet-based temporal downsampling with an attention mechanism, decomposes signals into frequency bands, integrates information with attention scores, and reconstructs signals for accurate hemodynamic response modeling.

#### Spatial Decoding
The `spatial_decoder.py` script includes the following classes:
- **GraphAttentionV2Layer**: Implements a multi-head graph attention mechanism to process node features and compute attention scores efficiently.
- **GMWA**: Initializes the Graph Multi-Headed Wavelet Attention module for context-aware spatial upsampling and autoregression to predict hemodynamic responses from electrophysiological data.
- **GMWANet**: Configures the GMWA module with custom parameters to enhance spatial upsampling and autoregressive decoding, aiming for improved prediction accuracy.
- **Autoregressive**: Integrates recurrent layers (e.g., LSTM) with dropout and fully connected layers to process sequences and generate outputs.
- **MLP**: A simple multilayer perceptron that uses input, hidden, and output layers with ReLU activation for basic neural processing tasks.
