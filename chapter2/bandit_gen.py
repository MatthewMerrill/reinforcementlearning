import argparse
import numpy as np
import random
import sys

def generate_data(num_levers, epochs, stddev):
    lever_means = [random.gauss(0, 1) for n in range(num_levers)]
    return np.array([[random.gauss(mean, stddev) for mean in lever_means]
            for epoch in range(epochs)
            ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Generates problem case for n-armed bandit')
    parser.add_argument('-n', nargs='?', type=int, default=10,
            help='number of levers to generate (default=10)')
    parser.add_argument('-e', nargs='?', type=int,
            metavar='EPOCHS', default=1000,
            help='number of epochs to generate lever data for')
    parser.add_argument('-s', nargs='?', type=int,
            metavar='STDDEV', default=1,
            help='stddev for values within a lever')
    argv = parser.parse_args()
    np.savetxt(sys.stdout, generate_data(argv.n, argv.e, argv.s), fmt='%.6f')


