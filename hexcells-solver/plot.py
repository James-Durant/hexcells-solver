import os
import numpy as np
import matplotlib.pyplot as plt

COLOURS = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_accuracy_single(log_path, models, file_name, labels=None):
    fig = plt.figure(figsize=[9, 7], dpi=600)
    ax = fig.add_subplot(111)

    for i, (model, colour) in enumerate(zip(models, COLOURS)):
        accuracy_history = np.loadtxt(os.path.join(log_path, f'model_{model}.csv'), delimiter=',')
        level_nums = accuracy_history[:, 0]
        train_accuracy = accuracy_history[:, 1]
        val_accuracy = accuracy_history[:, 2]

        label = f'Model {model}' if labels is None else labels[i]

        ax.plot(level_nums, train_accuracy, color=colour, label=label, linewidth=0.8, linestyle='solid')
        ax.plot(level_nums, val_accuracy, color=colour, linewidth=0.8, linestyle='dashed')

    ax.set_xlabel('Number of Levels Solved', fontsize=11, weight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, weight='bold')
    ax.set_ylim(0.35, 1.05)
    ax.legend()

    save_path = os.path.join(log_path, '..', 'figures')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig.savefig(os.path.join(save_path, f'{file_name}.png'), dpi=300)


def plot_accuracy_split(log_path, models, file_name, labels=None):
    fig1 = plt.figure(figsize=[9, 7], dpi=600)
    ax1 = fig1.add_subplot(111)

    fig2 = plt.figure(figsize=[9, 7], dpi=600)
    ax2 = fig2.add_subplot(111)

    for i, (model, colour) in enumerate(zip(models, COLOURS)):
        accuracy_history = np.loadtxt(os.path.join(log_path, f'model_{model}.csv'), delimiter=',')
        level_nums = accuracy_history[:, 0]
        train_accuracy = accuracy_history[:, 1]
        val_accuracy = accuracy_history[:, 2]

        label = f'Model {model}' if labels is None else labels[i]

        ax1.plot(level_nums, train_accuracy, color=colour, label=label, linewidth=0.8, linestyle='solid')
        ax2.plot(level_nums, val_accuracy, color=colour, label=label, linewidth=0.8, linestyle='dashed')

    ax1.set_xlabel('Number of Levels Solved', fontsize=11, weight='bold')
    ax2.set_xlabel('Number of Levels Solved', fontsize=11, weight='bold')

    ax1.set_ylabel('Training Accuracy', fontsize=11, weight='bold')
    ax2.set_ylabel('Validation Accuracy', fontsize=11, weight='bold')

    ax1.set_ylim(0.35, 1.05)
    ax2.set_ylim(0.35, 1.05)

    ax1.legend()
    ax2.legend()

    save_path = os.path.join(log_path, '..', 'figures')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig1.savefig(os.path.join(save_path, f'{file_name}_training.png'), dpi=300)
    fig2.savefig(os.path.join(save_path, f'{file_name}_validation.png'), dpi=300)


if __name__ == '__main__':
    plot_accuracy_split('resources/models/logs', [1, 2, 3, 4, 5], 'comparison_nodes_layers')
    plot_accuracy_split('resources/models/logs', [4, 6, 7, 8, 9], 'comparison_activation_filters')
    plot_accuracy_single('resources/models/logs', [4, 10, 11], 'comparison_replay_double', labels=['Standard', 'Experience Replay', 'Double DQN'])
    plot_accuracy_single('resources/models/logs', [4, 12, 13], 'comparison_level_sizes', labels=['Small', 'Medium', 'Large'])
