"""
This code is part of the supplementary materials for the NeurIPS 2024 submission:
'SAMBA: Latent Representation Learning for Multimodal Brain Activity Translation' 

"""


import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_wavelets import DWT1DForward


class PerParcelHrfLearning(nn.Module): 
    def __init__(self, args):
        super(PerParcelHrfLearning, self).__init__() 
        """
        Initialize a module for per-parcel Hemodynamic Response Function (HRF) learning.
        This module constructs a list of differentiable HRFs tailored to individually learn 
        response and undershooting parameters for each parcel. It then convolves each parcel's 
        neural activity with the inferred HRF.
        
        Args:
            args: The arguments or configurations passed for HRF initialization.
            hrf_stride (int): The stride value for convolution operation.
            n_parcels (int): The total number of parcels to model.
        """
        self.hrf_stride = args.hrf_stride
        self.hrfs = [Differentiable_HRF(args) for _ in range(args.ele_to_hemo_n_source_parcels)]
             
    def forward(self, x_ele):
        """
        Perform forward propagation through the PerParcelHrfs module by applying learned
        HRFs to the input data. This results in a convolved output representing the 
        hemodynamic response for each parcel.

        Args:
            x_ele (torch.float32): Input data with dimensions  

        Returns:
            zm_hrf (torch.float32): The convolved output of the HRFs for each parcel  
        """
        zm_hrf = None
        for p in range(x_ele.shape[1]):
            hrf_p = self.hrfs[p].forward().unsqueeze(0).unsqueeze(0)
            zm_p_hrf = F.conv1d(x_ele[:, p, :].unsqueeze(1), hrf_p, padding=499, stride=self.hrf_stride)
            zm_hrf = zm_p_hrf if p == 0 else torch.cat((zm_hrf, zm_p_hrf), dim=0)
        
        return zm_hrf 
 
 
class Differentiable_HRF(nn.Module):
    def __init__(self, args):
        super(Differentiable_HRF, self).__init__()
        """
        Initialize the Differentiable_HRF module designed to learn hemodynamic response functions (HRFs).  

        Args:
            args: A configuration object containing initialization parameters  

        Attributes:
            response_delay, undershoot_delay, response_dispersion, undershoot_dispersion,
            response_scale, undershoot_scale (nn.Parameter): Torch tensors initialized from the args
            and set as trainable parameters.
            net (torch.nn.Sequential): A multi-layer perceptron for inferring updates to HRF parameters.
        """
        # Initialize parameters from the configuration object
        self.hrf_length = args.hrf_length
        self.device = args.device
        self.hrf_temporal_resolution = args.hrf_temporal_resolution
        self.response_delay_init = args.hrf_response_delay_init
        self.undershoot_delay_init = args.hrf_undershoot_delay_init
        self.response_dispersion_init = args.hrf_response_dispersion_init
        self.undershoot_dispersion_init = args.hrf_undershoot_dispersion_init
        self.response_scale_init = args.hrf_response_scale_init
        self.undershoot_scale_init = args.hrf_undershoot_scale_init
        self.dispersion_deviation = args.dispersion_deviation
        self.scale_deviation = args.scale_deviation

        # Define trainable parameters
        self.response_delay = nn.Parameter(torch.tensor(self.response_delay_init).float(), requires_grad=True).to(args.device)
        self.undershoot_delay = nn.Parameter(torch.tensor(self.undershoot_delay_init).float(), requires_grad=True).to(args.device)
        self.response_dispersion = nn.Parameter(torch.tensor(self.response_dispersion_init).float(), requires_grad=True).to(args.device)
        self.undershoot_dispersion = nn.Parameter(torch.tensor(self.undershoot_dispersion_init).float(), requires_grad=True).to(args.device)
        self.response_scale = nn.Parameter(torch.tensor(self.response_scale_init).float(), requires_grad=True).to(args.device)
        self.undershoot_scale = nn.Parameter(torch.tensor(self.undershoot_scale_init).float(), requires_grad=True).to(args.device)

        # Setup the neural network for parameter inference
        hrf_mlp_neurons = [128, 512, 128]
        self.net = torch.nn.Sequential(
            nn.Linear(args.hrf_n_parameters, hrf_mlp_neurons[0]),
            nn.GELU(),
            nn.Linear(hrf_mlp_neurons[0], hrf_mlp_neurons[1]),
            nn.GELU(),
            nn.Linear(hrf_mlp_neurons[1], hrf_mlp_neurons[2]),
            nn.GELU(),
            nn.Linear(hrf_mlp_neurons[2], args.hrf_n_parameters),
        ).to(args.device)
   
    def _double_gamma_hrf(
        self,
        response_delay,
        undershoot_delay,
        response_dispersion,
        undershoot_dispersion,
        response_scale,
        undershoot_scale,
        temporal_resolution,
        ):
        """Create the double gamma HRF with the timecourse evoked activity.
        Default values are based on Glover, 1999 and Walvaert, Durnez,
        Moerkerke, Verdoolaege and Rosseel, 2011
        
        only _double_gamma_hrf is adapted from: 
        https://github.com/brainiak/brainiak/blob/master/brainiak/utils/fmrisim.py

        Parameters
        ----------

        response_delay : float
            How many seconds until the peak of the HRF

        undershoot_delay : float
            How many seconds until the trough of the HRF

        response_dispersion : float
            How wide is the rising peak dispersion

        undershoot_dispersion : float
            How wide is the undershoot dispersion

        response_scale : float
            How big is the response relative to the peak

        undershoot_scale :float
            How big is the undershoot relative to the trough

        scale_function : bool
            Do you want to scale the function to a range of 1

        temporal_resolution : float
            How many elements per second are you modeling for the stimfunction
        Returns
        ----------

        hrf : multi dimensional array
            A double gamma HRF to be used for convolution.

        """ 
        hrf_len = int(self.hrf_length * temporal_resolution)
        hrf_counter = torch.arange(hrf_len).float().to(self.device) 
        
        response_peak = response_delay * response_dispersion
        undershoot_peak = undershoot_delay * undershoot_dispersion

        # Specify the elements of the HRF for both the response and undershoot
        resp_pow = torch.pow((hrf_counter / temporal_resolution) / response_peak, response_delay)
        resp_exp = torch.exp(-((hrf_counter / temporal_resolution) - response_peak) / response_dispersion)

        response_model = response_scale * resp_pow * resp_exp

        undershoot_pow = torch.pow((hrf_counter / temporal_resolution) / undershoot_peak, undershoot_delay)
        undershoot_exp = torch.exp(-((hrf_counter / temporal_resolution) - undershoot_peak) / undershoot_dispersion)

        undershoot_model = undershoot_scale * undershoot_pow * undershoot_exp

        # For each time point, find the value of the HRF
        hrf = response_model - undershoot_model 
        return hrf  
    
    def forward(self):
        """
        Forward pass to learn six parameters of the Hemodynamic Response Function (HRF)
        using a Multi-Layer Perceptron (MLP). This method involves several steps:
        
        1. Concatenation of differentiable parameters: response delay, undershoot delay,
        response dispersion, undershoot dispersion, response scale, and undershoot scale.
        2. Forward propagation through the MLP to infer updated parameter values.
        3. Application of the double gamma function to separate the inferred parameters into
        their respective components for the next levels of HRF modeling.
        
        Returns:
            torch.Tensor: The inferred HRF modeled with the double gamma function, adjusted
            for the specific temporal resolution of the study.
        """
        # Concatenate initial parameter values to prepare for MLP input.
        x_hrf = torch.cat((
            self.response_delay.unsqueeze(0),
            self.undershoot_delay.unsqueeze(0),
            self.response_dispersion.unsqueeze(0),
            self.undershoot_dispersion.unsqueeze(0),
            self.response_scale.unsqueeze(0),
            self.undershoot_scale.unsqueeze(0)
        ), dim=0).unsqueeze(0)

        # Forward pass through the MLP to infer updated parameter values.
        y_hrf = self.net(x_hrf).squeeze(0)

        # Apply transformations to infer the final HRF parameters  
        response_delay = F.tanh(y_hrf[0:1]) * (2/6) * self.response_delay_init + self.response_delay_init
        undershoot_delay = F.tanh(y_hrf[1:2]) * (2/12) * self.undershoot_delay_init + self.undershoot_delay_init
        response_dispersion = F.tanh(y_hrf[2:3]) * self.dispersion_deviation * self.response_dispersion_init + self.response_dispersion_init
        undershoot_dispersion = F.tanh(y_hrf[3:4]) * self.dispersion_deviation * self.undershoot_dispersion_init + self.undershoot_dispersion_init
        response_scale = F.tanh(y_hrf[4:5]) * self.scale_deviation * self.response_scale_init + self.response_scale_init
        undershoot_scale = F.tanh(y_hrf[5:6]) * self.scale_deviation * self.undershoot_scale_init + self.undershoot_scale_init

        # Compute the HRF using the double gamma function with the inferred parameters.
        hrf_out = self._double_gamma_hrf(
            response_delay=response_delay,
            undershoot_delay=undershoot_delay,
            response_dispersion=response_dispersion,
            undershoot_dispersion=undershoot_dispersion,
            response_scale=response_scale,
            undershoot_scale=undershoot_scale,
            temporal_resolution=self.hrf_temporal_resolution
        )
        
        return hrf_out
    
    
class WaveletAttentionNet(nn.Module):
    def __init__(self, args):
        super(WaveletAttentionNet, self).__init__()
        """
        Initialize the WaveletAttentionNet module which applies wavelet-based encoding with
        a specialized attention mechanism over each frequency band, using the SAMBA
        attention framework. This module transforms neural signal representations
        for improved modeling of hemodynamic responses.

        Args:
            args: A configuration object containing initialization parameters, including
                  dimensions for the wavelet transform, inverse time dimension, and
                  computational device settings.

        Attributes:
            dwt (DWT1DForward): A discrete wavelet transform layer for signal decomposition.
            g1, g2, g3, g4 (nn.Linear): Linear transformation layers for different wavelet bands.
            attention_scores (nn.Linear): Computes attention scores for weighted sum of bands.
            inversewave (ThreeLayerMLP): A multilayer perceptron for signal reconstruction.
        """
        # Device and dimension configuration
        wavelet_dim = args.ele_to_hemo_wavelet_dim
        inverse_time_dim = args.ele_to_hemo_inverse_time_dim
        device = args.device

        # Wavelet transform and band-specific linear transformations
        self.dwt = DWT1DForward(wave='db5', J=3).to(device)
        dims = [1004, 506, 257, 257]  # Dimensions for single GPU setup
        self.g1 = nn.Linear(dims[0], wavelet_dim, bias=True).to(device)
        self.g2 = nn.Linear(dims[1], wavelet_dim, bias=True).to(device)
        self.g3 = nn.Linear(dims[2], wavelet_dim, bias=True).to(device)
        self.g4 = nn.Linear(dims[3], wavelet_dim, bias=True).to(device)

        # Attention mechanism to integrate information across different bands
        self.attention_scores = nn.Linear(in_features=wavelet_dim, out_features=1).to(device)

        # MLP for reversing the wavelet transform 
        hidden_size = 1024
        self.inversewave = ThreeLayerMLP(wavelet_dim, hidden_size, inverse_time_dim).to(device)

    def forward(self, x_meg_hrf):
        """
        Forward pass through the WaveletAttentionNet model to compute joint embeddings of MEG data
        post-hemodynamic response filtering, applying an LSTM autoencoder using a teacher-forced
        algorithm.

        Args:
            x_meg_hrf (torch.float32): The embedding of MEG data after HRF processing.
            x_fmri (torch.float32): Original fMRI data used during training with the teacher-forcing
            ratio; `x_fmri` is used only during training.

        Returns:
            tuple: Contains the output of the LSTM autoencoder and the attention weights.
        """
        # Wavelet transform and individual band processing
        xl, xh_list = self.dwt(x_meg_hrf)
        z1 = self.g1(xh_list[0])
        z2 = self.g2(xh_list[1])
        z3 = self.g3(xh_list[2])
        z4 = self.g4(xl)
        v_list = [z1, z2, z3, z4]

        # Compute attention scores and apply attention to the wavelet bands
        attention_scores = torch.cat([self.attention_scores(v) for v in v_list], dim=1)
        alphas = F.softmax(attention_scores.squeeze(-1), dim=1)

        # Weighted sum of the bands based on attention scores
        u_p_meg = sum(v.squeeze(1) * alphas[:, idx].unsqueeze(-1) for idx, v in enumerate(v_list))

        # Rearrange and apply inverse wavelet MLP
        u_p_meg = rearrange(u_p_meg, '(b d) w -> b (d) w', d=200)
        u_p_meg = self.inversewave(u_p_meg)
        u_p_meg = rearrange(u_p_meg, 'b (d c) w -> (b d) c w', d=200, c=1)

        return u_p_meg, alphas
    
class ThreeLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ThreeLayerMLP, self).__init__() 
        
        self.layer1 = nn.Linear(input_size, hidden_size) 
        self.layer2 = nn.Linear(hidden_size, hidden_size) 
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x): 
        x = F.relu(self.layer1(x)) 
        x = F.relu(self.layer2(x)) 
        x = self.layer3(x)
        return x
    
    