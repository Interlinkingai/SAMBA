"""
This code is part of the supplementary materials for the NeurIPS 2024 submission:
'SAMBA: Latent Representation Learning for Multimodal Brain Activity Translation' 

"""


import torch.nn as nn
import torch.nn.functional as F 
import torch
import random 
import numpy as np 
from einops import rearrange
from labml_helpers.module import Module
  
  
class GraphAttentionV2Layer(nn.Module):
    def __init__(self, in_features, out_features, n_heads):
        """
        Implementaion of a multi-head graph attention mechanism. 
        
        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node
            n_heads (int): Number of attention heads to use for parallel attention processes
        """
        super().__init__()
        self.n_heads = n_heads
        self.n_hidden = out_features
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, zh, adj_mat):
        """
        Propagates the input through the graph attention layer to compute the output features and attention scores.

        Args:
            zh (torch.tensor): Node features tensor,  
            adj_mat (torch.tensor): Adjacency matrix of the graph

        Returns:
            torch.tensor: Aggregated node features after applying attention 
            torch.tensor: Attention scores for each node pair
        """
        
        g_l = self.linear_l(zh).view(zh.shape[0], self.n_heads, self.n_hidden)
        g_r = self.linear_r(zh).view(zh.shape[0], self.n_heads, self.n_hidden)

        # Create repeated arrays
        g_l_repeat = g_l.repeat(zh.shape[0], 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(zh.shape[0], dim=0)

        # Sum features from transformations
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(zh.shape[0], zh.shape[0], self.n_heads, self.n_hidden)

        # Compute attention scores 
        e = self.attn(self.activation(g_sum)).squeeze(-1)
        att_score = e.masked_fill(adj_mat == 0, float('-inf'))

        # Apply softmax to normalize the attention scores
        att_score_sm = self.softmax(att_score)
        attn_res = torch.einsum('ijh,jhf->ihf', att_score_sm, g_r)

        return attn_res.mean(dim=1), att_score_sm.squeeze(-1)
    
    
class GMWA(nn.Module):
    def __init__(self, in_features, out_features, n_heads, dim_head, n_source_nodes, 
                 n_target_nodes, n_patches, lstm_num_layers, dropout, device):
        """
        Initializes the Graph Multi-Headed Wavelet Attention (GMWA) module.

        Args:
        - in_features (int): Number of features.
        - out_features (int): Number of features.
        - n_heads (int): Number of attention heads.
        - dim_head (int): Dimensionality of each attention head.
        - n_source_nodes (int): Number of source nodes in the graph.
        - n_target_nodes (int): Number of target nodes in the graph.
        - n_patches (int): Number of patches for graph division.
        - lstm_num_layers (int): Number of layers in the LSTM decoder.
        - dropout (float): Dropout rate.
        - device (str): Device to run the module ('cpu' or 'gpu'). 
        """
        super(GMWA, self).__init__()

        self.n_patches = n_patches
        self.device = device
        
        # Dropout layer configuration
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Define graph attention layers
        self.gat_layer1 = GraphAttentionV2Layer(in_features, in_features, n_heads)
        self.gat_layer2 = GraphAttentionV2Layer(out_features, out_features, n_heads)

        # Decoder configuration
        dec_embedding_dim = n_target_nodes
        dec_features = n_target_nodes
        self.autoregressive_model = Autoregressive(lstm_num_layers, out_features, dec_embedding_dim, dec_features, dropout).to(device)

        # Initialize mappers for each patch
        n_super_source_nodes = n_source_nodes // self.n_patches
        n_super_target_nodes = n_target_nodes // self.n_patches
        self.mapper = nn.ModuleList([
            nn.Linear(n_super_source_nodes, n_super_target_nodes).to(device) for _ in range(self.n_patches)
        ])

        # Activation and LSTM initial states
        self.activation = nn.ELU()
        self.h0 = nn.Parameter(torch.zeros(lstm_num_layers, 1, n_target_nodes))
        self.c0 = nn.Parameter(torch.zeros(lstm_num_layers, 1, n_target_nodes))
        
    def context_aware_spatial_upsampling(self, z_ele, adjacency_matrix):
        """
        Perform context-aware spatial upsampling from electrophysiological (ele) recordings
        to the hemodynamic (hemo) domain. 

        Args:
            z_ele (torch.float32): The output tensor from the source graph network, 
            adjacency_matrix (torch.float32): The adjacency matrix representing the graph structure 

        Returns:
            torch.float32: Spatially upsampled tensor in the hemodynamic domain,  
            torch.float32: Attention weights  
        """
        # Reshape the electrophysiological  
        z_reshaped = rearrange(z_ele, '(d1 d2) t -> t d1 d2', d1=self.n_patches)

        # Apply learned mappings for each patch 
        spatial_temporal_attention = []
        for i, mapper in enumerate(self.mapper):
            z_mapped = mapper(z_reshaped[:, i, :]).unsqueeze(1)
            spatial_temporal_attention.append(z_mapped)

        # Concatenate of the attention weights  
        spatial_temporal_attention = torch.cat(spatial_temporal_attention, dim=1)

        # Rearrange  
        z_hemo = rearrange(spatial_temporal_attention, 't p1 p2 -> (p1 p2) t')

        return z_hemo, spatial_temporal_attention

    def forward_graph_and_upsampling(self, x_ele, adj_source, adj_target):
        """
        Perform a forward pass through the source graph attention layer and then spatially
        upsample the results for a target graph. This function processes electrophysiological
        data through a graph attention network (GAT) followed by a context-aware spatial
        upsampling.

        Args:
            x_ele (torch.float32): The input tensor of electrophysiological data.
            adj_source (torch.float32): The adjacency matrix for the source graph.
            adj_target (torch.float32): The adjacency matrix for the target graph.

        Returns:
            torch.float32: The tensor representing spatially upsampled electrophysiological data
            torch.float32: Attention weights from the source graph layer, for visualization.
            torch.float32: Spatial-temporal attention 
        """
        # Apply the first graph attention layer  
        z, attention_weights = self.gat_layer1(x_ele, adj_source.unsqueeze(-1))

        # Activate the outputs from the first GAT layer
        z_activated = self.activation(z)

        # Perform context-aware spatial upsampling  
        z_upsampled, spatial_temporal_attention = self.context_aware_spatial_upsampling(z_activated, adj_target)

        # Rearrange 
        z_final = rearrange(z_upsampled, 'd t -> t d')

        return z_final, attention_weights, spatial_temporal_attention
    
    def rnn_forward(self, x_spatial_temporal, y_original, adj_target, teacher_forcing_ratio):
        """
        Forward pass through an RNN with teacher forcing for sequence generation.  

        Args:
            x_spatial_temporal (torch.float32): The input tensor, spatially upsampled and temporally
                                                downsampled electrophysiological data.
            y_original (torch.float32): The original hemodynamic data 
            adj_target (torch.float32): The adjacency matrix
            teacher_forcing_ratio (float): The probability of using the true previous output 

        Returns:
            torch.float32: The tensor containing the sequence of outputs from the LSTM.
            torch.float32: The attention weights from the final graph attention layer, for visualization.
        """
        # Unsqueeze the input and target tensors to add a batch dimension
        x_spatial_temporal = x_spatial_temporal.unsqueeze(0)
        y_original = y_original.unsqueeze(0)

         
        batch_size, trg_len, trg_dim = y_original.shape

        # Initialize hidden and cell states
        hidden = self.h0.repeat(1, batch_size, 1)
        cell = self.c0.repeat(1, batch_size, 1)
 
        # Generate sequence
        input_tensor = y_original[:, 0, :]
        outputs = []
        for t in range(trg_len):
            input_tensor = input_tensor.reshape((batch_size, 1, trg_dim))
            output, hidden, cell = self.autoregressive_model(input_tensor, hidden, cell)
            outputs.append(output)

            # Determine the next input based on teacher forcing ratio
            if random.random() < teacher_forcing_ratio:
                input_tensor = y_original[:, t, :]
            else:
                input_tensor = x_spatial_temporal[:, t, :]

            # Apply graph attention layer
            z, attention_weights = self.gat_layer2(rearrange(input_tensor.squeeze(1), 'd t -> t d'), adj_target.unsqueeze(-1))
            input_tensor = rearrange(z, 'd t -> t d')

        outputs = torch.cat(outputs, dim=1)

        return outputs, attention_weights
    
    def forward(self, x, y, adj_source, adj_target, teacher_forcing_ratio):
        x, source_att, st_attention = self.forward_graph_and_upsampling(x, adj_source, adj_target)
        x, target_att = self.rnn_forward(x, y, adj_target, teacher_forcing_ratio)
        return x, target_att.unsqueeze(0)
    
    
class GMWANet(nn.Module):
    def __init__(self, args):
        super(GMWANet, self).__init__()
        """
        Initializes the GMWANet class with custom parameters
        This module first applies spatial upsampling and then employs autoregression 
        to decode precise hemodynamic predictions 
        from electrophysiological data.
        """
        self.args = args
        self.gmwa_layer = GMWA(
            in_features=args.ele_to_hemo_in_features,
            out_features=1,
            n_heads=args.ele_to_hemo_n_heads,
            dim_head=args.ele_to_hemo_dim_head,
            n_source_nodes=args.ele_to_hemo_n_source_parcels,
            n_target_nodes=args.ele_to_hemo_n_target_parcels,
            n_patches=args.ele_to_hemo_n_patches,
            lstm_num_layers=args.ele_to_hemo_lstm_num_layers,
            dropout=args.ele_to_hemo_dropout,
            device=args.device
        )

    def forward(self, x, y, batch_size, sub_hemo, sub_ele, teacher_forcing_ratio):
        """
        Processes input data through the network, performing spatial upsampling 
        and autoregression using LSTM to enhance prediction accuracy.
        
        Args:
            x (torch.float32): Wavelet-transformed electrophysiological data.
            y (torch.float32): Hemodynamic data.
            batch_size (int): Specifies the number of samples in a batch.
            sub_hemo (list of strings): Needed to load specific hemodynamic adjacency matrices.
            sub_ele (list of strings): Needed to load specific electrophysiological adjacency matrices.
            teacher_forcing_ratio (float): Ratio for applying teacher forcing  

        Returns:
            torch.float32: The reconstructed hemodynamic predictions.
            torch.float32: Attention values for visualization purposes.
        """
        x = rearrange(x, '(b s1) s2 t -> b (s1 s2) t', b=batch_size)
        y = rearrange(y, 'b d t -> b t d')

        for b in range(batch_size):
            hemo_adjacency_matrix = torch.tensor(np.load(self.args.hemo_adjacency_matrix_dir + sub_hemo[b] + '.npy')).float().to(self.args.device)
            ele_adjacency_matrix = torch.tensor(np.load(self.args.ele_adjacency_matrix_dir + sub_ele[b] + '.npy')).float().to(self.args.device)

            xfm_graph_b, h_att_b = self.gmwa_layer(
                x[b, :, :],
                y[b, :, :],
                ele_adjacency_matrix,
                hemo_adjacency_matrix,
                teacher_forcing_ratio
            )

            if b == 0:
                xfm_graph = xfm_graph_b
                h_att = h_att_b
            else:
                xfm_graph = torch.cat((xfm_graph, xfm_graph_b), dim=0)
                h_att = torch.cat((h_att, h_att_b), dim=0)

        xfm_graph = rearrange(xfm_graph, 'b t d -> b d t').float()
        return xfm_graph, h_att
    
class Autoregressive(nn.Module):
    def __init__(self, num_layers, seq_len, embedding_dim, n_features, dropout):
        """ 
        This module integrates recurrent layers to process sequences, 
        with dropout for regularization and fully connected layers for output generation.

        Args:
            num_layers (int): Number of layers in the LSTM.
            seq_len (int): Length of the input sequence.
            embedding_dim (int): Dimensionality of the embedding space in the LSTM.
            n_features (int): Number of features in the input and output data.
            dropout (float): Dropout rate for regularization 
        """
        super(Autoregressive, self).__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.n_features = n_features
        self.dropout = dropout

        # LSTM layer  
        self.rnn = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.embedding_dim,
            num_layers=num_layers,
            dropout=self.dropout,
            batch_first=True
        )

        # Fully connected layers 
        self.output_layer1 = nn.Linear(self.embedding_dim, 128)
        self.output_layer2 = nn.Linear(128, self.n_features)
        self.relu = nn.ReLU()

    def forward(self, x, h_0, c_0):
        """
        Forward pass through the decoder model 

        Args:
            x (torch.Tensor): Input sequence tensor  
            h_0 (torch.Tensor): Initial hidden state  
            c_0 (torch.Tensor): Initial cell state  

        Returns:
            torch.Tensor: The output sequence from the decoder 
            torch.Tensor: Final hidden states  
            torch.Tensor: Final cell states  
        """
        # Recurrent pass
        x, (hidden_n, cell_n) = self.rnn(x, (h_0, c_0))
        
        # Reshape  
        x = x.reshape((len(x), 1, self.embedding_dim))
        
        # Sequential linear transformations with ReLU activation and dropout
        x = self.relu(self.output_layer1(x))
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.output_layer2(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = x.reshape((len(x), 1, self.n_features))

        return x, hidden_n, cell_n
    
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
    