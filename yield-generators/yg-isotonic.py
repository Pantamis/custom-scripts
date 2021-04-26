#!/usr/bin/env python3
from future.utils import iteritems

import random

from jmbase import get_log, jmprint
from jmclient import YieldGeneratorBasic, ygmain, jm_single

# This is a maker for the purposes of generating a yield from held bitcoins
# while maximising the difficulty of spying on blockchain activity.
# This is primarily attempted by randomizing all aspects of orders
# after transactions wherever possible.

# YIELD GENERATOR SETTINGS ARE NOW IN YOUR joinmarket.cfg CONFIG FILE
# (You can also use command line flags; see --help for this script).

jlog = get_log()

# PAVA (Pool Adjascent Violators Algorithm) is the algorithm used to solve the isotonic regression
# problem. It fits a nondecreasing line to a sequence of observations in a linear order.
# The result minimizes a score which can be interpreted as the distance of the data to the isotonic 
# cone associated with the constraints given by the linear order. The isotonic YG
# uses the smallest possible mixdepth for the linear order of mixdepths (considered 
# in cyclic order) for which is the associated isotonic cone is the closest. 
# The following code implements PAVA in O(n) and was inspired from sklearn/_isotonic.pyx
# Credit: Nelle Varoquaux, Andrew Tulloch, Antony Lee, the scikit-learn developers

def PAVA_score(y):
    yhat = y.copy()
    n = len(yhat)
    target = list(range(n))
    w = [1]*len(yhat)
    i = 0
    while i < n:
        k = target[i] + 1
        if k == n:
            break
        if yhat[i] < yhat[k]:
            i = k
            continue
        sum_wy = w[i] * yhat[i]
        sum_w = w[i]
        while True:
            # We are within a decreasing subsequence.
            prev_y = yhat[k]
            sum_wy += w[k] * yhat[k]
            sum_w += w[k]
            k = target[k] + 1
            if k == n or prev_y < yhat[k]:
                # Non-singleton decreasing subsequence is finished,
                # update first entry.
                yhat[i] = sum_wy / sum_w
                w[i] = sum_w
                target[i] = k - 1
                target[k - 1] = i
                if i > 0:
                    # Backtrack if we can.  This makes the algorithm
                    # single-pass and ensures O(n) complexity.
                    i = target[i - 1]
                # Otherwise, restart from the same point.
                break
    # Reconstruct the solution.
    i = 0
    while i < n:
        k = target[i] + 1
        yhat[i + 1 : k] = [yhat[i]]*(max(k-1-i,0))
        i = k
    return sum([(a - b)**2 for a,b in zip(y,yhat)])

# Given a distribution of funds in mixdepths arranged in cyclic order,
# compute the isotonic score for all possible linear orders or cuts.
# Return the rank where the first mixdepth for the optimal linear
# order is.

def rank_best_order(y):
    scores = [0]*len(y)
    for i in range(len(y)):
        scores[i] = PAVA_score(y[i:]+y[:i])
    return min(range(len(y)), key = scores.__getitem__)

class YieldGeneratorIsotonic(YieldGeneratorBasic):

    def __init__(self, wallet_service, offerconfig):
        super().__init__(wallet_service, offerconfig)
        mix_balance = wallet_service.get_balance_by_mixdepth(verbose=False, minconfs=1)
        self.rank = None
        
    def select_input_mixdepth(self, available, offer, amount):
        """Returns the smallest mixdepth that can be chosen from, i.e. has enough 
        balance but after the rank defining the linear order of mixdepths for which 
        balances are increasing. This rank is computed with PAVA when order is created.
        """
        available = sorted(available.items(), 
                           key = lambda entry: (entry[0] - self.rank)%(self.wallet_service.mixdepth + 1))
        return available[0][0]
    
    def create_my_orders(self):
        mix_balance = self.get_available_mixdepths()
        # Update the optimal rank for linear order now to fill potential orders without delay
        # For each possible linear order of the mixdepth, we compute the isotonic score
        self.rank = rank_best_order([b for m, b in sorted(mix_balance.items())])
        jlog.info('Arrangement in linear order starting from mixdepth '+ str(self.rank))
        # We publish ONLY the maximum amount and use minsize for lower bound;
        # leave it to oid_to_order to figure out the right depth to use.
        f = '0'
        if self.ordertype in ['swreloffer', 'sw0reloffer']:
            f = self.cjfee_r
        elif self.ordertype in ['swabsoffer', 'sw0absoffer']:
            f = str(self.txfee + self.cjfee_a)
        mix_balance = dict([(m, b) for m, b in iteritems(mix_balance)
                            if b > self.minsize])
        if len(mix_balance) == 0:
            jlog.error('You do not have the minimum required amount of coins'
                       ' to be a maker: ' + str(self.minsize) + \
                       '\nTry setting txfee to zero and/or lowering the minsize.')
            return []
        max_mix = max(mix_balance, key=mix_balance.get)

        # randomizing the different values
        randomize_txfee = int(random.uniform(self.txfee * (1 - float(self.txfee_factor)),
                                             self.txfee * (1 + float(self.txfee_factor))))
        randomize_minsize = int(random.uniform(self.minsize * (1 - float(self.size_factor)),
                                               self.minsize * (1 + float(self.size_factor))))
        possible_maxsize = mix_balance[max_mix] - max(jm_single().DUST_THRESHOLD, randomize_txfee)
        randomize_maxsize = int(random.uniform(possible_maxsize * (1 - float(self.size_factor)),
                                               possible_maxsize))

        if self.ordertype in ['swabsoffer', 'sw0absoffer']:
            randomize_cjfee = int(random.uniform(float(self.cjfee_a) * (1 - float(self.cjfee_factor)),
                                                 float(self.cjfee_a) * (1 + float(self.cjfee_factor))))
            randomize_cjfee = randomize_cjfee + randomize_txfee
        else:
            randomize_cjfee = random.uniform(float(f) * (1 - float(self.cjfee_factor)),
                                             float(f) * (1 + float(self.cjfee_factor)))
            randomize_cjfee = "{0:.6f}".format(randomize_cjfee)  # round to 6 decimals

        order = {'oid': 0,
                 'ordertype': self.ordertype,
                 'minsize': randomize_minsize,
                 'maxsize': randomize_maxsize,
                 'txfee': randomize_txfee,
                 'cjfee': str(randomize_cjfee)}

        # sanity check
        assert order['minsize'] >= 0
        assert order['maxsize'] > 0
        assert order['minsize'] <= order['maxsize']
        if order['ordertype'] in ['swreloffer', 'sw0reloffer']:
            while order['txfee'] >= (float(order['cjfee']) * order['minsize']):
                order['txfee'] = int(order['txfee'] / 2)
                jlog.info('Warning: too high txfee to be profitable, halfing it to: ' + str(order['txfee']))

        return [order]


if __name__ == "__main__":
    ygmain(YieldGeneratorIsotonic, nickserv_password='')
    jmprint('done', "success")
