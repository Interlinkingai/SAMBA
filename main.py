import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from model.model_registry import str2model  
from args import params_fn, print_gpu_info, make_directroy, meg2List 
from utils.viz_corr import correlation_r2_plot
from data.trva_spliter import tr_va_datalaoder
from argparse import ArgumentParser
import numpy as np
import torch
import os
 
from data.schaeferparcel_kong2022_17network import SchaeferParcel_Kong2022_17Network

def root_fn(server_mode, dataset):
    """
    Generates paths based on the server mode and dataset.
    Raises ValueError if server mode is not recognized.
    """
    dataset_paths = {
        'eegfmri_translation': 'EEGfMRI/',
        'megfmri': 'MEGfMRI/'
    }

    server_roots = {
        'ws1': '/home/scratch/datasets/NEUROSCIENCE/',
        'ws2': '/home/datasets/NEUROSCIENCE/'
    }

    if dataset not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset}")
    if server_mode not in server_roots:
        raise ValueError("Specify the server_mode by 'ws1', or 'ws2'")

    datasets_root = server_roots[server_mode] + dataset_paths[dataset]
    task_dir = 'minute_dataset/' if dataset == 'megfmri' else 'minute_dataset_translation/'
    graph_dir = 'graphs/' if dataset == 'megfmri' else 'graphs_' + dataset.split('_')[0] + '/' 

    source_data_dir = datasets_root + task_dir
    graph_dir = datasets_root + graph_dir
    parcel_dir = datasets_root + 'schaefer_parcellation_labels/'
    
    return source_data_dir, graph_dir, parcel_dir

def subject_lists(dataset):
    """
    Returns subject lists based on the dataset.
    """
    subjects = {
        'megfmri': (['02', '03', '04', '05', '06', '07', '08', '09', '10', '11'],
                    ['02', '03', '04', '05', '06', '09', '10', '14', '15', '16', '17', '18', '19', '20']),
        'eegfmri_translation': (['07', '08', '10', '11', '12', '13', '14', '16', '19', '20', '21', '22'],
                                ['07', '08', '10', '11', '12', '13', '14', '16', '19', '20', '21', '22'])
    }

    if dataset not in subjects:
        raise ValueError(f"Unknown dataset: {dataset}")

    return subjects[dataset]

def print_gpu_info(args, train_dataloader):
    os.system('cls' if os.name == 'nt' else 'clear')
    if torch.cuda.is_available():
        print("---------------------------------------------------\n")
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            props = torch.cuda.get_device_properties(device)
            print(f'       Device {i}:  ')
            print(f'           Name: {props.name} ')
            print(f'           Memory: {props.total_memory / 1024 ** 3:.2f} GB')
            print("           ----------------               ") 
            if train_dataloader.single_subj:
                print(f'           {train_dataloader.ele_subs[0]} -> {train_dataloader.hemo_subs[0]}')
            else:
                print('           Across the subjects!')
            print("           ---------------------------------             ") 
            print(f'           Model: {args.output_key}\n')
    else:
        print('No GPU available.')
    print("---------------------------------------------------") 

def params_fn(server_mode, dataset):
    """
    Configures and returns command-line arguments for a neuroscience dataset processing pipeline.
    """
    n_hemo_parcels = 500
    n_ele_parcels = 200

    source_data_dir, graph_dir, parcel_dir = root_fn(server_mode, dataset)
    parser = ArgumentParser(description="Setup parameters for neuroscience data processing")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser.add_argument("--model", type=str)
    parser.add_argument("--clf_mode")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--device", default=device)
    parser.add_argument("--graph_dir", default=graph_dir)
    parser.add_argument("--parcels_dir", default=parcel_dir)
    parser.add_argument("--n_hemo_parcels", type=int, default=n_hemo_parcels)
    parser.add_argument("--n_ele_parcels", type=int, default=n_ele_parcels)
    parser.add_argument("--single_subj", type=bool)
    parser.add_argument("--hemo_adjacency_matrix_dir", default=graph_dir + f'/fmri-{n_hemo_parcels}parcels/')
    parser.add_argument("--ele_adjacency_matrix_dir", default=graph_dir + f'/{dataset.split("_")[0]}-{n_ele_parcels}parcels/')
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--n_way", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--validation_iteration", type=int)
    parser.add_argument("--save_model")
    parser.add_argument("--output_key", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--hemo_dir")
    parser.add_argument("--ele_dir")

    parser.add_argument("--ele_to_hemo_n_source_parcels", type=int)
    parser.add_argument("--ele_to_hemo_n_target_parcels", type=int)
    parser.add_argument("--ele_to_hemo_wavelet_dim", type=int)
    parser.add_argument("--ele_to_hemo_inverse_time_dim", type=int)
    parser.add_argument("--ele_to_hemo_in_features", type=int)
    parser.add_argument("--ele_to_hemo_n_heads", type=int)
    parser.add_argument("--ele_to_hemo_dim_head", type=int)
    parser.add_argument("--ele_to_hemo_n_patches", type=int)
    parser.add_argument("--ele_to_hemo_lstm_num_layers", type=int)
    parser.add_argument("--ele_to_hemo_dropout", type=float)
    parser.add_argument("--ele_to_hemo_teacher_forcing_ratio", type=float)

    parser.add_argument("--hrf_length", type=int)
    parser.add_argument("--hrf_stride", type=int)
    parser.add_argument("--hrf_n_parameters", type=int)
    parser.add_argument("--hrf_temporal_resolution", type=float)
    parser.add_argument("--hrf_response_delay_init", type=float)
    parser.add_argument("--hrf_undershoot_delay_init", type=float)
    parser.add_argument("--hrf_response_dispersion_init", type=float)
    parser.add_argument("--hrf_undershoot_dispersion_init", type=float)
    parser.add_argument("--hrf_response_scale_init", type=float)
    parser.add_argument("--hrf_undershoot_scale_init", type=float)
    parser.add_argument("--dispersion_deviation", type=float)
    parser.add_argument("--scale_deviation", type=float)

    params = parser.parse_args()

    labels, names, ctabs = SchaeferParcel_Kong2022_17Network(parcel_number=n_ele_parcels)
    params.parcels_name = [name[13:-1] for name in names]
    params.parcel_name_vertex = labels
    params.parcels_name.pop(0)
    params.lh_rh_lob_names = ['Default', 'Lang.', 'Cont', 'SalVenAttn', 'DorsAttn', 'Aud', 'SomMot', 'Visual']

    ele_sub_list, hemo_sub_list = subject_lists(dataset)
    params.ele_sub_list = ele_sub_list
    params.hemo_sub_list = hemo_sub_list
    
    params.hemo_adjacency_matrix_dir = params.graph_dir + f'/fmri-{params.n_hemo_parcels}parcels/'
    params.ele_adjacency_matrix_dir = params.graph_dir + f'/{dataset.split("_")[0]}-{params.n_ele_parcels}parcels/'

    return params


def training(args, model, train_dataloader, valid_dataloader):
    optimizer = Adam(model.parameters(), lr=args.lr)
    best_val_loss = np.inf

    for iteration, ((xm, xf, y_meta, y_batch), [minute_index, sub_f, sub_m]) in enumerate(train_dataloader):
        model.train()
        
        loss = model.loss(xm, xf, sub_m, sub_f, iteration, [minute_index, sub_f, sub_m]) if args.clf_mode else model.loss(xm, xf, sub_m, sub_f, iteration)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration == 0 or iteration % args.validation_iteration == 0:
            model.eval()
            val_loss_i = validation(args, model, valid_dataloader, iteration)
            
            if val_loss_i < best_val_loss:
                best_val_loss = val_loss_i
                save_model(args.output_dir + '/samba/best_model.pth', iteration, model, model.result_list, cpu=True)

            save_model(args.output_dir + '/samba/current_model.pth', iteration, model, model.result_list)
            model.print_results(iteration)
            meg2List(args.output_dir + '/xmodel/results.txt', model.result_list)
            model.to(args.device)

def save_model(path, iteration, model, result_list, cpu=False):
    model_data = {
        'iteration': iteration,
        'model': model.cpu() if cpu else model,
        'result_list': result_list
    }
    torch.save(model_data, path)

def main():
    datasets = ['megfmri', 'eegfmri_translation']
    args = params_fn(server_mode='misha', dataset=datasets[0])
    args.model = 'SambaEleToHemoClf2'
    args.single_subj = False

    if args.dataset == 'megfmri':
        from data.dataloader import NumpyBatchDataset as NumpyDataset
    elif args.dataset == 'eegfmri_translation':
        from data.dataloader_translation import NumpyBatchDataset as NumpyDataset

    proto_model = str2model(args.model)
    model = proto_model(args).to(args.device)
    args.output_key = time.strftime(args.model + "-%Y%m%d-%H%M%S_" + args.dataset)
    
    train_dataloader, valid_dataloader = tr_va_datalaoder(args)
    print_gpu_info(args, train_dataloader)
    
    training(args, model, train_dataloader, valid_dataloader)

if __name__ == "__main__":
    main()

 
  
  
   