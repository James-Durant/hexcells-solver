import os
import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy_epochs(log_path):
    fig = plt.figure(figsize=[9, 7], dpi=600)
    ax = fig.add_subplot(111)

    accuracy_history = np.loadtxt(log_path, delimiter=',')
    train_accuracy = accuracy_history[:, 0]
    test_accuracy = accuracy_history[:, 1]
    epoch_nums = range(1, len(train_accuracy)+1)

    ax.plot(epoch_nums, train_accuracy, label='Train')
    ax.plot(epoch_nums, test_accuracy, label='Test')

    ax.set_xlabel('Number of Levels Solved', fontsize=11, weight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, weight='bold')
    ax.set_xticks(1, len(train_accuracy)+1)

    log_path, _ = os.path.splitext(log_path)
    fig.savefig(log_path+'.png', dpi=300)


if __name__ == '__main__':
    model = 'model_1'

    file_path = f'resources/models/logs/{model}.csv'
    plot_accuracy_epochs(file_path)
