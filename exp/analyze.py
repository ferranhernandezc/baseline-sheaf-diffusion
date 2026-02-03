#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import random
import time
import torch
# import git
import numpy as np
import matplotlib.pyplot as plt

# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from models.positional_encodings import append_top_k_evectors
from models.cont_models import DiagSheafDiffusion, BundleSheafDiffusion, GeneralSheafDiffusion
from models.disc_models import DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion, DiscreteGeneralSheafDiffusion, DiscreteIdentityDiffusion
from utils.heterophilic import get_dataset, get_fixed_splits
from data_analysis.heterophily import heterophily_classifier, min_and_max_gains


def run_model(args, dataset, fold):
    data = dataset[0]
    data = get_fixed_splits(data, args.dataset, fold)
    data = data.to(args.device)
    
    best_model = torch.load(f"./best_models/{dataset.name}/{args.model}/fold_{fold}_model.pt", weights_only=False)

    _, rayleigh_quotients, baseline_rayleigh_quotients = best_model(data.x, return_rayleigh_quotient=True)

    return rayleigh_quotients, baseline_rayleigh_quotients


def fancy_plot(all_coeffs1, all_coeffs2):

    # Convert to NumPy array: shape (n_lists, list_length)
    coeffs1 = np.array(all_coeffs1)
    coeffs2 = np.array(all_coeffs2)

    # Compute mean and std along the lists axis
    mean1 = coeffs1.mean(axis=0)
    std1 = coeffs1.std(axis=0)
    mean2 = coeffs2.mean(axis=0)
    std2 = coeffs2.std(axis=0)

    # X-axis: positions in the sublists
    x = np.arange(len(mean1))

    # Plot
    plt.clf()
    plt.plot(mean1, label="Rf", color="red")
    plt.fill_between(
        x,
        mean1-std1,
        mean1+std1,
        color="red",
        alpha=0.2
    )
    plt.plot(mean2, label="Ri", color="blue")
    plt.fill_between(
        x,
        mean2-std2,
        mean2+std2,
        color="blue",
        alpha=0.2
    )

    plt.xlabel("Layers")
    plt.ylabel("Average Rayleigh Quotient")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'./plots/{dataset.name}/{args.model}/rayleigh_quotients.png')

def time_model_run(args, dataset, fold):
     
    data = dataset[0]
    data = get_fixed_splits(data, args.dataset, fold)
    data = data.to(args.device)
     
    best_model = torch.load(f"./best_models/{dataset.name}/{args.model}/fold_{fold}_model.pt", weights_only=False)

    start = time.time()
    best_model(data.x)
    end = time.time()
    return (end-start)
     

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    models = {
        'DiagSheafODE': DiagSheafDiffusion,
        'BundleSheafODE': BundleSheafDiffusion,
        'GeneralSheafODE': GeneralSheafDiffusion,
        'DiagSheaf': DiscreteDiagSheafDiffusion,
        'BundleSheaf': DiscreteBundleSheafDiffusion,
        'GeneralSheaf': DiscreteGeneralSheafDiffusion,
        'IdentitySheaf': DiscreteIdentityDiffusion,
    }

    # repo = git.Repo(search_parent_directories=True)
    sha = "shgapo"

    if args.model in models:
        model_cls = models[args.model]
    else:
        raise ValueError(f'Unknown model {args.model}')

    dataset = get_dataset(args.dataset)
    if args.evectors > 0:
        dataset = append_top_k_evectors(dataset, args.evectors)

    os.makedirs("./plots", exist_ok=True)
    os.makedirs(f"./plots/{dataset.name}", exist_ok=True)
    os.makedirs(f"./plots/{dataset.name}/{args.model}", exist_ok=True)
    os.makedirs("./best_models/", exist_ok=True)
    os.makedirs(f"./best_models/{dataset.name}", exist_ok=True)
    os.makedirs(f"./best_models/{dataset.name}/{args.model}", exist_ok=True)


    # Add extra arguments
    args.sha = sha
    args.graph_size = dataset[0].x.size(0)
    args.input_dim = dataset.num_features
    args.output_dim = dataset.num_classes
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    assert args.normalised or args.deg_normalised
    if args.sheaf_decay is None:
        args.sheaf_decay = args.weight_decay

    # Set the seed for everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    all_rayleigh_quotients = []
    all_baseline_rayleigh_quotients = []
    times_per_layer = []
    for fold in range(args.folds):
            
            data = dataset[0]
            data = get_fixed_splits(data, args.dataset, fold)
            data = data.to(args.device)
            
            best_model = torch.load(f"./best_models/{dataset.name}/{args.model}/fold_{fold}_model.pt", weights_only=False)

            _, rayleigh_quotients, baseline_rayleigh_quotients = best_model(data.x, return_rayleigh_quotient=True)

            rayleigh_quotients, baseline_rayleigh_quotients = run_model(args, dataset, fold)

            times_per_layer.append(time_model_run(args, dataset, fold))
            
            rayleigh_quotients = [dir_en for dir_en in rayleigh_quotients]
            baseline_rayleigh_quotients = [dir_en for dir_en in baseline_rayleigh_quotients]
            plt.clf()
            plt.plot(rayleigh_quotients, label='Sheaf Rayleigh Quotient')
            plt.plot(baseline_rayleigh_quotients, label='Baseline Rayleigh Quotient')
            plt.legend()
            plt.savefig(f'./plots/{dataset.name}/{args.model}/fold_{fold}_rayleigh_quotients.png')
            all_rayleigh_quotients.append(rayleigh_quotients)
            all_baseline_rayleigh_quotients.append(baseline_rayleigh_quotients)


    mean_rayleighs = [np.mean(rq) for rq in all_rayleigh_quotients]
    mean_baseline_rayleighs = [np.mean(rq) for rq in all_baseline_rayleigh_quotients]
    rayleigh_std = np.std(mean_rayleighs)
    baseline_rayleigh_std = np.std(mean_baseline_rayleighs)

    fancy_plot(all_rayleigh_quotients, all_baseline_rayleigh_quotients)


    sheaf_diffs = [np.mean([rq2 - rq1 for rq1, rq2 in zip(rq[:-1], rq[1:])]) for rq in all_rayleigh_quotients]
    baseline_diffs = [np.mean([rq2 - rq1 for rq1, rq2 in zip(rq[:-1], rq[1:])]) for rq in all_baseline_rayleigh_quotients]
    model_name = args.model if args.evectors == 0 else f"{args.model}+LP{args.evectors}"
    print(f'{model_name} on {args.dataset} | SHA: {sha}')
    print(f"Time per layer: {np.mean(times_per_layer)}")
    print(f'Min gain / Max Gain: {min_and_max_gains(dataset[0].y, dataset[0].edge_index)}')
    print(f'Heterophily: {heterophily_classifier(dataset[0].y, dataset[0].edge_index)}')
    print(f'Sheaf Rayleigh Quotient: {np.mean(mean_rayleighs):.2f} +/- {rayleigh_std:.2f}')
    print(f'Baseline Rayleigh Quotient: {np.mean(mean_baseline_rayleighs):.2f} +/- {baseline_rayleigh_std:.2f}')
    print(f'Sheaf Difference between Rayleigh Quotients: {np.mean(sheaf_diffs):.2f} +/- {np.std(sheaf_diffs):.2f}')
    print(f'Baseline Difference between Rayleight Quotients: {np.mean(baseline_diffs):.2f} +/- {np.std(baseline_diffs):.2f}')


