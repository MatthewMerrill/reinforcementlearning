import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots lever data for an n-armed bandit')
    parser.add_argument('outfile', type=argparse.FileType('wb'),
                        help='File to write to')
    argv = parser.parse_args()

    data = np.loadtxt(sys.stdin, ndmin=2)
    sns.violinplot(data=data)
    plt.savefig(argv.outfile)
