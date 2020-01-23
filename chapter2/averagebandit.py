import random

class AverageBandit:
    def __init__(self, num_levers=10, e=0, e_decay=1):
        self.num_levers = num_levers
        self.estimates = [ColumnEstimate() for col in range(num_levers)]
        self.e = e
        self.e_decay = e_decay
        self.e_initial = e
        self.last_lever = None

    def reset(self):
        for estimate in self.estimates: estimate.reset()
        self.e = self.e_initial
        self.last_lever = None

    def best_lever_index(self):
        best_levers = [0]
        best_mean = self.estimates[0].mean()

        for lever_idx, lever in enumerate(self.estimates):
            cur_mean = lever.mean()
            if best_mean < cur_mean:
                best_levers = [lever_idx]
                best_mean = cur_mean
            elif best_mean == cur_mean:
                best_levers.append(lever_idx)

        return random.choice(best_levers)

    def get_action(self):
        if random.random() < self.e:
            ax = random.randrange(self.num_levers)
        else:
            ax = self.best_lever_index()

        self.e *= self.e_decay
        self.last_lever = ax
        return ax

    def observe(self, rx):
        self.estimates[self.last_lever].add_value(rx)


class ColumnEstimate:
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.value_sum = 0

    def add_value(self, value):
        self.values.append(value)
        self.value_sum += value

    def mean(self):
        return (self.value_sum / len(self.values)) if self.values else 0


