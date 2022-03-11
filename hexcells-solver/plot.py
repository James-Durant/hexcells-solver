import os
import numpy as np
import matplotlib.pyplot as plt

from parse import RESOURCES_PATH


def plot_accuracy(log_path, models, file_name, split, labels=None):
    """Plot training and validation accuracy versus number of levels solved.

    Args:
        log_path (str): file path to the log containing accuracies to plot.
        models (list): models to load logs for.
        file_name (str): file name to use when saving the plot.
        split (bool): whether to use separate plots for training and validation accuracies.
        labels (list, optional): labels to use for the plot legend(s).
    """
    # Create the figures.
    fig1 = plt.figure(figsize=[9, 7], dpi=600)
    fig2 = plt.figure(figsize=[9, 7], dpi=600)
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)

    # The default matplotlib colour cycle.
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Iterate over each model.
    for i, (model, colour) in enumerate(zip(models, colours)):
        # Load the log for the model.
        accuracy_history = np.loadtxt(os.path.join(log_path, f'model_{model}.csv'), delimiter=',')

        # Get the training and validation accuracies and the number of levels solved at each step.
        level_nums = accuracy_history[:, 0]
        train_accuracy = accuracy_history[:, 1]
        val_accuracy = accuracy_history[:, 2]

        # Label for the legend.
        label = f'Model {model}' if labels is None else labels[i]

        # Plot the training and validation accuracies.
        if split:
            ax1.plot(level_nums, train_accuracy, color=colour, label=label, linewidth=0.8, linestyle='solid')
            ax2.plot(level_nums, val_accuracy, color=colour, label=label, linewidth=0.8, linestyle='dashed')
        else:
            ax1.plot(level_nums, train_accuracy, color=colour, label=label, linewidth=0.8, linestyle='solid')
            ax1.plot(level_nums, val_accuracy, color=colour, linewidth=0.8, linestyle='dashed')

    # Set the x-axis labels.
    ax1.set_xlabel('Number of Levels Solved', fontsize=11, weight='bold')
    ax2.set_xlabel('Number of Levels Solved', fontsize=11, weight='bold')

    # Set the y-axis labels.
    if split:
        ax1.set_ylabel('Training Accuracy', fontsize=11, weight='bold')
        ax2.set_ylabel('Validation Accuracy', fontsize=11, weight='bold')
    else:
        ax1.set_ylabel('Accuracy', fontsize=11, weight='bold')

    # Set the range for the y-axis.
    ax1.set_ylim(0.35, 1.05)
    ax2.set_ylim(0.35, 1.05)

    # Add legends to the plots.
    ax1.legend()
    if split:
        ax2.legend()

    # Create a new directory for figures if one does not exist.
    save_path = os.path.join(log_path, '..', 'figures')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the figure(s).
    if split:
        fig1.savefig(os.path.join(save_path, f'{file_name}_training.png'), dpi=300)
        fig2.savefig(os.path.join(save_path, f'{file_name}_validation.png'), dpi=300)
    else:
        fig1.savefig(os.path.join(save_path, f'{file_name}.png'), dpi=300)


if __name__ == '__main__':
    # If this file is being run, create the plots used in the final report.
    log_path = os.path.join(RESOURCES_PATH, 'models', 'logs')
    plot_accuracy(log_path, [1, 2, 3, 4, 5], 'comparison_nodes_layers', True)
    plot_accuracy(log_path, [4, 6, 7, 8, 9], 'comparison_activation_filters', True)
    plot_accuracy(log_path, [4, 10, 11], 'comparison_replay_double', False, ['Standard', 'Experience Replay', 'Double DQN'])
    plot_accuracy(log_path, [4, 12, 13], 'comparison_level_sizes', False, ['Small', 'Medium', 'Large'])
