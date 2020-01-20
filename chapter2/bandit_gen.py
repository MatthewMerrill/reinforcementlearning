import argparse
import random


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

    N = argv.n
    E = argv.e

    lever_means = [random.gauss(0, 1) for n in range(N)]
    for epoch in range(E):
        epoch_data = ['%.2f' % random.gauss(mean, argv.s) for mean in lever_means]
        print(' '.join(epoch_data))

