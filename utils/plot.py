import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from openTSNE import TSNE
import seaborn as sns
import pandas as pd

color_dict = {
    0: "#1f77b4",
    1: "#ff7f0e", 
    2: '#2ca02c', 
    3: '#d62728', 
    4: '#9467bd', 
    5: '#8c564b', 
    6: '#e377c2', 
    7: '#7f7f7f', 
    8: '#bcbd22', 
    9: '#17becf'
} 

def plot_label_distribution_each_client(label_distribution_each_client, output_path): 
    label_distribution_each_client = np.array(label_distribution_each_client)
    n_clients, n_classes = label_distribution_each_client.shape
    if n_classes > 10:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(n_classes):
        ax.bar(np.arange(n_clients), label_distribution_each_client[:, i], bottom=np.sum(label_distribution_each_client[:, :i], axis=1), color=color_dict[i])

    ax.set_xlabel('Clients')   
    ax.set_ylabel('Label Distribution')

    # plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_label_distribution(label_distribution, output_path):
    n_classes = len(label_distribution)
    if n_classes > 10:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    class_colors = [v for v in color_dict.values()]

    ax.bar(np.arange(n_classes), label_distribution, color=class_colors)

    ax.set_xlabel('Classes')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()



def plot_tsne(vectors, gt_labels, output_path):
    classes= set(np.unique(gt_labels))
    gt_labels= gt_labels.reshape(-1, 1)

    embed = TSNE(n_jobs=4).fit(vectors)
    pd_embed = pd.DataFrame(embed)
    pd_embed_prototype = pd_embed[len(gt_labels):]
    pd_embed_prototype.insert(loc=2, column='class ID', value=range(len(classes)))
    pd_embed_data = pd_embed[:len(gt_labels)]
    pd_embed_data.insert(loc=2, column='label', value=gt_labels)

    sns.set_context({'figure.figsize': [15, 10]})
    sns.scatterplot(x=0, y=1, hue="label", data=pd_embed_data, legend=False, palette=color_dict)
    sns.scatterplot(x=0, y=1, hue="class ID", data=pd_embed_prototype, s=200, palette=color_dict)
    plt.axis('off')
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_singular_values(singular_values, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(singular_values) + 1), np.log(singular_values), linestyle='-')
    plt.xlabel("Singular Value Rank Index")
    plt.ylabel("Log of Singular Values")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def plot_singular_values_compare(all_sv, output_path, title):
    plt.figure(figsize=(8, 5))

    for task, val in all_sv.items():
        plt.plot(range(1, len(val) + 1), np.log(val), linestyle='-', label=task)

    plt.title(title)
    plt.xlabel("Singular Value Rank Index")
    plt.ylabel("Log of Singular Values")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_eff_rank_compare(all_eff_rank, output_path, title):
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_eff_rank)))

    plt.figure(figsize=(8, 5))

    for i, (exp, eff_rank) in enumerate(all_eff_rank.items()):
        plt.bar(i, eff_rank, color=colors[i], label=exp, alpha=0.7, width=0.5)

    plt.title(title)
    plt.ylabel("Effective Rank")
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2  # Adjust the number of columns for readability
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def plot_vmf_kde(theta, density, output_path):
    plt.figure(figsize=(8, 5))
    sns.histplot(x=theta, weights=density, kde=True)

    plt.xlabel('Angles')
    plt.ylabel('Density')
    plt.title('von Mises–Fisher KDE')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_gaussian_kde(density, output_path):
    plt.figure(figsize=(6, 6))

    plt.imshow(density, extent=(-1, 1, -1, 1))
    plt.title('Feature Distribution')
    plt.savefig(output_path)


def plot_distances_distribution(distances, output_path):
    mean_distance = np.mean(distances)
    plt.figure(figsize=(10, 6))
    bins = np.arange(0, 2.25, 0.25)
    plt.hist(distances, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_distance, color='black', linestyle='--', linewidth=1.5, label='Mean')

    plt.xlabel(r"$\ell_2$ Distances")
    plt.ylabel("Counts")
    plt.title('Positive Pair Feature Distances')

    plt.legend()
    plt.savefig(output_path)


def plot_uniformity_compare(all_sv, output_path, title, ymin, ymax):
    plt.figure(figsize=(8, 5))

    for task, val in all_sv.items():
        plt.plot(range(1, len(val) + 1), val, linestyle='-', label=task)

    plt.ylim((ymin, ymax))
    plt.title(title)
    plt.xlabel("Rounds")
    plt.ylabel(r"$\mathcal{L}_{\mathrm{uniform}} (t=2)$")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_uniformity_compare_errorbar(all_sv, output_path, title):
    plt.figure(figsize=(8, 5))

    for task, val in all_sv.items():
        # plt.plot(range(1, len(val) + 1), val, linestyle='-', label=task)
        mean = [v[0] for v in val]
        std = [v[1] for v in val]
        plt.errorbar(range(1, len(val) + 1), mean, yerr=std, fmt='o-', label=task)


    plt.ylim((-2.5, -1.5))
    plt.title(title)
    plt.xlabel("Rounds")
    plt.ylabel(r"$\mathcal{L}_{\mathrm{uniform}} (t=2)$")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def plot_uniformity_round(score, output_path, title):
    plt.figure(figsize=(10, 6))

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.11

    for i, name in enumerate(score):
        x = np.arange(len(score[name]))
        ax.bar(x + i * width, score[name], width, label=name)

    # ax.set_xlabel('Clients')   

    ax.set_xticks(np.arange(len(x)) + width)
    ax.set_xticklabels([f'Client {i}' for i in range(1, len(x)+1)], rotation=45, ha='right', fontsize=10)

    ax.set_ylabel(r"$\mathcal{L}_{\mathrm{uniform}} (t=2)$")

    plt.legend(
        loc='lower center',
        # bbox_to_anchor=(0.5, -0.15),
        bbox_to_anchor=(0.5, 1.0),
        ncol=2  # Adjust the number of columns for readability
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_cosine_similarity(cos_sim, output_path, title, bins=10):

    for name, data in cos_sim.items():
        sns.histplot(data, bins=bins, kde=False, label=name)

    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
