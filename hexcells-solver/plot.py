import os
import numpy as np
import matplotlib.pyplot as plt

COLOURS = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_accuracy(log_path, models, file_name):
    fig = plt.figure(figsize=[9, 7], dpi=600)
    ax = fig.add_subplot(111)

    for model, colour in zip(models, COLOURS):
        accuracy_history = np.loadtxt(os.path.join(log_path, f'model_{model}.csv'), delimiter=',')
        level_nums = accuracy_history[:, 0]
        train_accuracy = accuracy_history[:, 1]
        test_accuracy = accuracy_history[:, 2]

        ax.plot(level_nums, train_accuracy, color=colour, linestyle='solid', linewidth=0.8, label=f'Model {model}')
        ax.plot(level_nums, test_accuracy, color=colour, linestyle='dashed', linewidth=0.8)

    ax.set_xlabel('Number of Levels Solved', fontsize=11, weight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, weight='bold')
    ax.legend()

    save_path = os.path.join(log_path, '..', 'figures')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig.savefig(os.path.join(save_path, f'{file_name}.png'), dpi=300)


if __name__ == '__main__':
    plot_accuracy('resources/models/logs', [1, 2, 3, 4], 'comparison_nodes_layers')
    # plot_accuracy('resources/models/logs', [5, 6, 7, 8], 'comparison_activation_filters')
    # plot_accuracy('resources/models/logs', [9, 10, 11], 'comparison_replay_double')
    # plot_accuracy('resources/models/logs', [12, 13, 14], 'comparison_level_sizes')
