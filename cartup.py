"""
Classical CART frame credit to https://github.com/chrisspen/dtree
The major contributions based on dtree are user preference and more object-oriented style
"""
from __future__ import print_function

from collections import defaultdict
from decimal import Decimal
from operator import itemgetter
from pprint import pprint
import copy
import csv
import math
from math import pi, log10
import os
import random
import re
import unittest
import pandas as pd

import six
from six.moves import cPickle as pickle
from six import iteritems, iterkeys, itervalues, string_types

VERSION = (1, 0, 1)
__version__ = '.'.join(map(str, VERSION))

# Traditional entropy.
ENTROPY1 = 'entropy1'

# Modified entropy that penalizes universally unique values.
ENTROPY2 = 'entropy2'

# Modified entropy that penalizes universally unique values
# as well as features with large numbers of values.
ENTROPY3 = 'entropy3'

DISCRETE_METRICS = [
    ENTROPY1,
    ENTROPY2,
    ENTROPY3,
]

# Simple statistical variance, the measure of how far a set of numbers
# is spread out.
VARIANCE1 = 'variance1'

# Like ENTROPY2, is the variance weighted to penalize attributes with
# universally unique values.
VARIANCE2 = 'variance2'

CONTINUOUS_METRICS = [
    VARIANCE1,
    VARIANCE2,
]

DEFAULT_DISCRETE_METRIC = ENTROPY1
DEFAULT_CONTINUOUS_METRIC = VARIANCE1

# Methods for aggregating the predictions of trees in a forest.
EQUAL_MEAN = 'equal-mean'
WEIGHTED_MEAN = 'weighted-mean'
BEST = 'best'
AGGREGATION_METHODS = [
    EQUAL_MEAN,
    WEIGHTED_MEAN,
    BEST,
]

# Forest growth algorithms.
GROW_RANDOM = 'random'
GROW_AUTO_MINI_BATCH = 'auto-mini-batch'
GROW_AUTO_INCREMENTAL = 'auto-incremental'
GROW_METHODS = [
    GROW_RANDOM,
    GROW_AUTO_MINI_BATCH,
    GROW_AUTO_INCREMENTAL,
]

# Data format names.
ATTR_TYPE_NOMINAL = NOM = 'nominal'
ATTR_TYPE_DISCRETE = DIS = 'discrete'
ATTR_TYPE_CONTINUOUS = CON = 'continuous'
ATTR_MODE_CLASS = CLS = 'class'
ATTR_HEADER_PATTERN = re.compile("([^,:]+):(nominal|discrete|continuous)(?::(class))?")

def get_mean(seq):
    """
    Batch mean calculation.
    """
    return sum(seq)/float(len(seq))

def get_variance(seq):
    """
    Batch variance calculation.
    """
    m = get_mean(seq)
    return sum((v-m)**2 for v in seq)/float(len(seq))

def standard_deviation(seq):
    return math.sqrt(get_variance(seq))

def mean_absolute_error(seq, correct):
    """
    Batch mean absolute error calculation.
    """
    assert len(seq) == len(correct)
    diffs = [abs(a-b) for a, b in zip(seq, correct)]
    return sum(diffs)/float(len(diffs))

def normalize(seq):
    """
    Scales each number in the sequence so that the sum of all numbers equals 1.
    """
    s = float(sum(seq))
    return [v/s for v in seq]

def erfcc(x):
    """
    Complementary error function.
    """
    z = abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * math.exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+
        t*(.09678418+t*(-.18628806+t*(.27886807+
        t*(-1.13520398+t*(1.48851587+t*(-.82215223+
        t*.17087277)))))))))
    if (x >= 0.):
        return r
    else:
        return 2. - r

def normcdf(x, mu, sigma):
    """
    Describes the probability that a real-valued random variable X with a given
    probability distribution will be found at a value less than or equal to X
    in a normal distribution.
    
    http://en.wikipedia.org/wiki/Cumulative_distribution_function
    """
    t = x-mu
    y = 0.5*erfcc(-t/(sigma*math.sqrt(2.0)))
    if y > 1.0:
        y = 1.0
    return y

def normpdf(x, mu, sigma):
    """
    Describes the relative likelihood that a real-valued random variable X will
    take on a given value.
    
    http://en.wikipedia.org/wiki/Probability_density_function
    """
    u = (x-mu)/abs(sigma)
    y = (1/(math.sqrt(2*pi)*abs(sigma)))*math.exp(-u*u/2)
    return y

def normdist(x, mu, sigma, f=True):
    if f:
        y = normcdf(x, mu, sigma)
    else:
        y = normpdf(x, mu, sigma)
    return y

def normrange(x1, x2, mu, sigma, f=True):
    p1 = normdist(x1, mu, sigma, f)
    p2 = normdist(x2, mu, sigma, f)
    return abs(p1-p2)

def cmp(a, b): # pylint: disable=redefined-builtin
    return (a > b) - (a < b)

class DDist(object):
    """
    Incrementally tracks the probability distribution of discrete elements.
    """
    
    def __init__(self, seq=None):
        self.clear()
        if seq:
            for k in seq:
                self.counts[k] += 1
                self.total += 1
    
    def __cmp__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return cmp(
            (frozenset(self.counts.items()), self.total),
            (frozenset(other.counts.items()), other.total)
        )
    
    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (frozenset(self.counts.items()), self.total) == \
            (frozenset(other.counts.items()), other.total)
    
    def __getitem__(self, k):
        """
        Returns the probability for the given element.
        """
        cnt = 0
        if k in self.counts:
            cnt = self.counts[k]
        return cnt/float(self.total)
    
    def __hash__(self):
        return hash((frozenset(self.counts.items()), self.total))
    
    def __repr__(self):
        s = []
        for k, prob in self.probs:
            s.append("%s=%s" % (k, prob))
        return "<%s %s>" % (type(self).__name__, ', '.join(s))
    
    def add(self, k, count=1):
        """
        Increments the count for the given element.
        """
        self.counts[k] += count
        self.total += count
    
    @property
    def best(self):
        """
        Returns the element with the highest probability.
        """
        b = (-1e999999, None)
        for k, c in iteritems(self.counts):
            b = max(b, (c, k))
        return b[1]
    
    @property
    def best_prob(self):
        probs = self.probs
        if not probs:
            return
        best = -1e999999
        for _, prob in probs:
            best = max(best, prob)
        return best
    
    def clear(self):
        self.counts = defaultdict(int)
        self.total = 0
    
    def copy(self):
        return copy.deepcopy(self)
    
    @property
    def count(self):
        """
        The total number of samples forming this distribution.
        """
        return self.total
    
    def keys(self):
        return self.counts.keys()
    
    @property
    def probs(self):
        """
        Returns a list of probabilities for all elements in the form
        [(value1,prob1),(value2,prob2),...].
        """
        return [
            (k, self.counts[k]/float(self.total))
            for k in iterkeys(self.counts)
        ]
    
    def update(self, dist):
        """
        Adds the given distribution's counts to the current distribution.
        """
        assert isinstance(dist, DDist)
        for k, c in iteritems(dist.counts):
            self.counts[k] += c
        self.total += dist.total

class CDist(object):
    """
    Incrementally tracks the probability distribution of continuous numbers.
    """
    
    def __init__(self, seq=None, mean=None, var=None, stdev=None):
        self.clear()
        if mean is not None:
            self.mean_sum = mean
            self.mean_count = 1
        if var is not None:
            self.last_variance = var
            self.mean_count = 1
        if stdev is not None:
            self.last_variance = stdev**2
            self.mean_count = 1
        if seq:
            for n in seq:
                self += n
    
    def clear(self):
        self.mean_sum = 0
        self.mean_count = 0
        self.last_variance = 0
    
    def copy(self):
        return copy.deepcopy(self)
    
    def __repr__(self):
        return "<%s mean=%s variance=%s>" \
            % (type(self).__name__, self.mean, self.variance)
    
    def __iadd__(self, value):
        last_mean = self.mean
        self.mean_sum += value
        self.mean_count += 1
        if last_mean is not None:
            self.last_variance = self.last_variance \
                + (value  - last_mean)*(value - self.mean)
        return self
    
    @property
    def count(self):
        """
        The total number of samples forming this distribution.
        """
        return self.mean_count
        
    @property
    def mean(self):
        if self.mean_count:
            return self.mean_sum/float(self.mean_count)
    
    @property
    def variance(self):
        if self.mean_count:
            return self.last_variance/float(self.mean_count)
        
    @property
    def standard_deviation(self):
        var = self.variance
        if var is None:
            return
        return math.sqrt(var)
    
    def probability_lt(self, x):
        """
        Returns the probability of a random variable being less than the
        given value.
        """
        if self.mean is None:
            return
        return normdist(x=x, mu=self.mean, sigma=self.standard_deviation)
    
    def probability_in(self, a, b):
        """
        Returns the probability of a random variable falling between the given
        values.
        """
        if self.mean is None:
            return
        p1 = normdist(x=a, mu=self.mean, sigma=self.standard_deviation)
        p2 = normdist(x=b, mu=self.mean, sigma=self.standard_deviation)
        return abs(p1 - p2)
    
    def probability_gt(self, x):
        """
        Returns the probability of a random variable being greater than the
        given value.
        """
        if self.mean is None:
            return
        p = normdist(x=x, mu=self.mean, sigma=self.standard_deviation)
        return 1-p

def entropy(data, class_attr=None, method=DEFAULT_DISCRETE_METRIC):
    """
    Calculates the entropy of the attribute attr in given data set data.
    
    Parameters:
    data<dict|list> :=
        if dict, treated as value counts of the given attribute name
        if list, treated as a raw list from which the value counts will be generated
    attr<string> := the name of the class attribute
    """
    assert (class_attr is None and isinstance(data, dict)) \
        or (class_attr is not None and isinstance(data, list))
    if isinstance(data, dict):
        counts = data
    else:
        counts = defaultdict(float) # {attr:count}
        for record in data:
            # Note: A missing attribute is treated like an attribute with a value
            # of None, representing the attribute is "irrelevant".
            counts[record.get(class_attr)] += 1.0
    len_data = float(sum(cnt for _, cnt in iteritems(counts)))
    # n = max(2, len(counts))
    n = 4
    total = float(sum(counts.values()))
    # print(total)
    if total == 0.0:
        print(data)
    # print(counts)
    assert total, "There must be at least one non-zero count."
    try:
        #return -sum((count/total)*math.log(count/total,n) for count in counts)
        if method == ENTROPY1:
            return -sum((count/len_data)*math.log(count/len_data, n)
                for count in itervalues(counts) if count)
        elif method == ENTROPY2:
            return -sum((count/len_data)*math.log(count/len_data, n)
                for count in itervalues(counts) if count) - ((len(counts)-1)/float(total))
        elif method == ENTROPY3:
            return -sum((count/len_data)*math.log(count/len_data, n)
                for count in itervalues(counts) if count) - 100*((len(counts)-1)/float(total))
        else:
            raise Exception("Unknown entropy method %s." % method)
    except Exception:
        raise

def entropy_variance(data, class_attr=None,
    method=DEFAULT_CONTINUOUS_METRIC):
    """
    Calculates the variance fo a continuous class attribute, to be used as an
    entropy metric.
    """
    assert method in CONTINUOUS_METRICS, "Unknown entropy variance metric: %s" % (method,)
    assert (class_attr is None and isinstance(data, dict)) \
        or (class_attr is not None and isinstance(data, list))
    if isinstance(data, dict):
        lst = data
    else:
        lst = [record.get(class_attr) for record in data]
    return get_variance(lst)

def get_gain(data, attr, class_attr,
    method=DEFAULT_DISCRETE_METRIC,
    only_sub=0, prefer_fewer_values=False, entropy_func=None):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    
    Parameters:
    
    prefer_fewer_values := Weights the gain by the count of the attribute's
        unique values. If multiple attributes have the same gain, but one has
        slightly fewer attributes, this will cause the one with fewer
        attributes to be preferred.
    """
    # print(attr)
    if not data:
        print('get_gain1')
    entropy_func = entropy_func or entropy
    val_freq = defaultdict(float)
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        val_freq[record.get(attr)] += 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record.get(attr) == val]
        if not data_subset:
            print('get_gain2')
            # print('attr: {}, value: {}' .format(attr, val))
            # print(val_freq)
            # e = 1.0

        e = entropy_func(data_subset, class_attr, method=method)
        subset_entropy += val_prob * e
        
    if only_sub:
        return subset_entropy

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    main_entropy = entropy_func(data, class_attr, method=method)
    
    # Prefer gains on attributes with fewer values.
    if prefer_fewer_values:
#        n = len(val_freq)
#        w = (n+1)/float(n)/2
        #return (main_entropy - subset_entropy)*w
        return ((main_entropy - subset_entropy), 1./len(val_freq))
    else:
        return (main_entropy - subset_entropy)

def gain_variance(*args, **kwargs):
    """
    Calculates information gain using variance as the comparison metric.
    """
    return get_gain(entropy_func=entropy_variance, *args, **kwargs)

def majority_value(data, class_attr):
    """
    Creates a list of all values in the target attribute for each record
    in the data list object, and returns the value that appears in this list
    the most frequently.
    """
    if is_continuous(data[0][class_attr]):
        return CDist(seq=[record[class_attr] for record in data])
    else:
        return most_frequent([record[class_attr] for record in data])

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def unique(lst):
    """
    Returns a list made up of the unique values found in lst.  i.e., it
    removes the redundant values in lst.
    """
    lst = lst[:]
    unique_lst = []

    # Cycle through the list and add each value to the unique list only once.
    for item in lst:
        if unique_lst.count(item) <= 0:
            unique_lst.append(item)
            
    # Return the list with all redundant values removed.
    return unique_lst

def get_values(data, attr):
    """
    Creates a list of values in the chosen attribut for each record in data,
    prunes out all of the redundant values, and return the list.  
    """
    return unique([record[attr] for record in data])


# Splitter class for splitting the attributes and creating the tree
class Splitter():

    def __init__(self, costs=defaultdict(), attributes=[]):
        self.cost_so_far = 0
        self.features_so_far = []


        # self.dummy_orig = defaultdict()
        # for dummy in attributes:
        #     for orig_feature in costs.keys():
        #         if str(dummy).startswith(str(orig_feature)):
        #             self.dummy_orig[dummy] = orig_feature


    def choose_attribute(self, data, attributes, class_attr, fitness, method, coe=0, costs=defaultdict(), max_cost=0):
        """
        Cycles through all the attributes and returns the attribute with the
        highest information gain (or lowest entropy).
        """
        best = (-1e999999, None)
        if not data:
            print('choose_attribute')

        for attr in attributes:
            if attr == class_attr:
                continue
            # add the cost term to the variable gain to influence the choosing of best splitting attributes
            # if self.cost_so_far < max_cost and self.dummy_orig[attr] not in self.features_so_far:
                # gain = fitness(data, attr, class_attr, method=method) - coe * costs[self.dummy_orig[attr]]
            if self.cost_so_far < max_cost and attr not in self.features_so_far:
                gain = fitness(data, attr, class_attr, method=method) - coe * costs[attr]
            else:
                gain = fitness(data, attr, class_attr, method=method)
            # print(best)
            best = max([best, (gain, attr)], key=itemgetter(0))
            # best = max(best, (gain, attr))
        # print(best[1])
        return best[1]

    def is_continuous(self, v):
        return isinstance(v, (float, Decimal))

    def create_decision_tree(self, data, attributes, class_attr, fitness_func, wrapper, coe=0, costs=defaultdict(), max_cost=None, max_feature=30,
                             **kwargs):
        """
        Returns a new decision tree based on the examples given.
        """

        split_attr = kwargs.get('split_attr', None)
        split_val = kwargs.get('split_val', None)

        # if max cost has not been assigned, use the number of attributes
        if max_cost == None:
            max_cost = len(attributes) + 2

        assert class_attr not in attributes
        node = None
        data = list(data) if isinstance(data, Data) else data
        if wrapper.is_continuous_class:
            stop_value = CDist(seq=[r[class_attr] for r in data])
            # For a continuous class case, stop if all the remaining records have
            # a variance below the given threshold.
            stop = wrapper.leaf_threshold is not None \
                and stop_value.variance <= wrapper.leaf_threshold
        else:
            stop_value = DDist(seq=[r[class_attr] for r in data])
            # For a discrete class, stop if all remaining records have the same
            # classification.
            stop = len(stop_value.counts) <= 1

        if not data or len(attributes) <= 0:
            # If the dataset is empty or the attributes list is empty, return the
            # default value. The target attribute is not in the attributes list, so
            # we need not subtract 1 to account for the target attribute.
            print('create_decision_tree')
            if wrapper:
                wrapper.leaf_count += 1

            return stop_value
        elif stop:
            # If all the records in the dataset have the same classification,
            # return that classification.
            if wrapper:
                wrapper.leaf_count += 1
            return stop_value
        else:
            # Choose the next best attribute to best classify our data
            best = self.choose_attribute(
                data,
                attributes,
                class_attr,
                fitness_func,
                method=wrapper.metric,
                coe=coe,
                costs=costs,
                max_cost=max_cost)
            # print("The current attribute:", best)

            # Create a new decision tree/node with the best attribute and an empty
            # dictionary object--we'll fill that up next.
    #        tree = {best:{}}
            node = Node(tree=wrapper, attr_name=best)
            node.n += len(data)

            # Only if a feature is being used for the first time, we charge the corresponding cost
            if self.cost_so_far < max_cost and len(self.features_so_far) < max_feature:
                best_orig = best# self.dummy_orig[best]
                if best_orig not in self.features_so_far:
                            self.cost_so_far += costs[best_orig]
                            self.features_so_far.append(best_orig)
            else:
                print("using impurity criterion")
            print("Cost so far: ", self.cost_so_far)
            print("Features used so far: ", self.features_so_far)
            to_save = pd.DataFrame(self.features_so_far)
            to_save.to_csv('feature_order_' + str(coe) + '_15.csv')
                # to_save.to_csv('feature_order_rate_' + str(log10(coe)) + '.csv')


            # Create a new decision tree/sub-node for each of the values in the
            # best attribute field
            # for numerical attributes, the split should be for the lower and upper parts of the dataset
            # criterion starts from this for loop
            for val in get_values(data, best):

                if self.cost_so_far < max_cost and len(self.features_so_far) < max_feature:
                # Create a subtree for the current value under the "best" field
                    subtree = self.create_decision_tree(
                        [r for r in data if r[best] == val],
                        [attr for attr in attributes if attr != best],
                        class_attr,
                        fitness_func,
                        split_attr=best,
                        split_val=val,
                        wrapper=wrapper,
                        coe=coe,
                        costs=costs,
                        max_cost=max_cost,
                        max_feature=max_feature)
                else:
                    subtree = self.create_decision_tree(
                        [r for r in data if r[best] == val],
                        [attr for attr in attributes if attr != best and attr in self.features_so_far],
                        class_attr,
                        fitness_func,
                        split_attr=best,
                        split_val=val,
                        wrapper=wrapper,
                        coe=coe,
                        costs=costs,
                        max_cost=max_cost,
                        max_feature=max_feature)

                # Add the new subtree to the empty dictionary object in our new
                # tree/node we just created.
                if isinstance(subtree, Node):
                    node._branches[val] = subtree
                elif isinstance(subtree, (CDist, DDist)):
                    # print(COST_SO_FAR)
                    node.set_leaf_dist(attr_value=val, dist=subtree)
                else:
                    raise Exception("Unknown subtree type: %s" % (type(subtree),))

        return node

class Data(object):
    """
    Parses, validates and iterates over tabular data in a file
    or an generic iterator.
    
    This does not store the actual data rows. It only stores the row schema.
    """
    
    def __init__(self, inp, order=None, types=None, modes=None):
        
        self.header_types = types or {} # {attr_name:type}
        self.header_modes = modes or {} # {attr_name:mode}
        print(self.header_modes)
        if isinstance(order, string_types):
            order = order.split(',')
        self.header_order = order or [] # [attr_name,...]
        
        # Validate header type.
        if isinstance(self.header_types, (tuple, list)):
            assert self.header_order, 'If header type names were not ' + \
                'given, an explicit order must be specified.'
            assert len(self.header_types) == len(self.header_order), \
                'Header order length must match header type length.'
            self.header_types = dict(zip(self.header_order, self.header_types))
        
        self.filename = None
        self.data = None
        if isinstance(inp, string_types):
            filename = inp
            assert os.path.isfile(filename), \
                "File \"%s\" does not exist." % filename
            self.filename = filename
        else:
            assert self.header_types, "No attribute types specified."
            assert self.header_modes, "No attribute modes specified."
            #assert self.header_order, "No attribute order specified."
            self.data = inp
        
        self._class_attr_name = None
        if self.header_modes:
            for k, v in iteritems(self.header_modes):
                if v != CLS:
                    continue
                self._class_attr_name = k
                break
            assert self._class_attr_name, "No class attribute specified."

    #     self.preprocess_continuous()
    #
    #
    # def preprocess_continuous(self):
    #     for var_name in self.header_order[:-1]:
    #         if self.header_types[var_name] == CON:

    
    def copy_no_data(self):
        """
        Returns a copy of the object without any data.
        """
        return type(self)(
            [],
            order=list(self.header_modes),
            types=self.header_types.copy(),
            modes=self.header_modes.copy())
    
    def __len__(self):
        if self.filename:
            return max(0, open(self.filename).read().strip().count('\n'))
        elif hasattr(self.data, '__len__'):
            return len(self.data)

    def __bool__(self):
        return bool(len(self))
    __nonzero__ = __bool__

    @property
    def class_attribute_name(self):
        return self._class_attr_name

    @property
    def attribute_names(self):
        self._read_header()
        return [
            n for n in iterkeys(self.header_types)
            if n != self._class_attr_name
        ]

    def get_attribute_type(self, name):
        if not self.header_types:
            self._read_header()
        return self.header_types[name]

    @property
    def is_continuous_class(self):
        self._read_header()
        return self.get_attribute_type(self._class_attr_name) \
            == ATTR_TYPE_CONTINUOUS

    def is_valid(self, name, value):
        """
        Returns true if the given value matches the type for the given name
        according to the schema.
        Returns false otherwise.
        """
        if name not in self.header_types:
            return False
        t = self.header_types[name]
        if t == ATTR_TYPE_DISCRETE:
            return isinstance(value, int)
        elif t == ATTR_TYPE_CONTINUOUS:
            return isinstance(value, (float, Decimal))
        return True

    def _read_header(self):
        """
        When a CSV file is given, extracts header information the file.
        Otherwise, this header data must be explicitly given when the object
        is instantiated.
        """
        if not self.filename or self.header_types:
            return
        rows = csv.reader(open(self.filename))
        #header = rows.next()
        header = next(rows)
        self.header_types = {} # {attr_name:type}
        self._class_attr_name = None
        self.header_order = [] # [attr_name,...]
        for el in header:
            matches = ATTR_HEADER_PATTERN.findall(el)
            assert matches, "Invalid header element: %s" % (el,)
            el_name, el_type, el_mode = matches[0]
            el_name = el_name.strip()
            self.header_order.append(el_name)
            self.header_types[el_name] = el_type
            if el_mode == ATTR_MODE_CLASS:
                assert self._class_attr_name is None, \
                    "Multiple class attributes are not supported."
                self._class_attr_name = el_name
            else:
                assert self.header_types[el_name] != ATTR_TYPE_CONTINUOUS, \
                    "Non-class continuous attributes are not supported."
        assert self._class_attr_name, "A class attribute must be specified."

    def validate_row(self, row):
        """
        Ensure each element in the row matches the schema.
        """
        clean_row = {}
        if isinstance(row, (tuple, list)):
            assert self.header_order, "No attribute order specified."
            assert len(row) == len(self.header_order), \
                "Row length does not match header length."
            itr = zip(self.header_order, row)
        else:
            assert isinstance(row, dict)
            itr = iteritems(row)
        for el_name, el_value in itr:
            if self.header_types[el_name] == ATTR_TYPE_DISCRETE:
                clean_row[el_name] = int(el_value)
            elif self.header_types[el_name] == ATTR_TYPE_CONTINUOUS:
                clean_row[el_name] = float(el_value)
            else:
                clean_row[el_name] = el_value
        return clean_row

    def _get_iterator(self):
        if self.filename:
            self._read_header()
            itr = csv.reader(open(self.filename))
            next(itr) # Skip header.
            return itr
        return self.data

    def __iter__(self):
        for row in self._get_iterator():
            if not row:
                continue
            yield self.validate_row(row)
            
    def split(self, ratio=0.5, leave_one_out=False):
        """
        Returns two Data instances, containing the data randomly split between
        the two according to the given ratio.
        
        The first instance will contain the ratio of data specified.
        The second instance will contain the remaining ratio of data.
        
        If leave_one_out is True, the ratio will be ignored and the first
        instance will contain exactly one record for each class label, and
        the second instance will contain all remaining data.
        """
        a_labels = set()
        a = self.copy_no_data()
        b = self.copy_no_data()
        for row in self:
            if leave_one_out and not self.is_continuous_class:
                label = row[self.class_attribute_name]
                if label not in a_labels:
                    a_labels.add(label)
                    a.data.append(row)
                else:
                    b.data.append(row)
            elif not a:
                a.data.append(row)
            elif not b:
                b.data.append(row)
            elif random.random() <= ratio:
                a.data.append(row)
            else:
                b.data.append(row)
        return a, b

USE_NEAREST = 'use_nearest'
USE_MOST = 'use_most'
MISSING_VALUE_POLICIES = set([
    USE_NEAREST, USE_MOST
])

def _get_dd_int():
    return defaultdict(int)

def _get_dd_dd_int():
    return defaultdict(_get_dd_int)

def _get_dd_cdist():
    return defaultdict(CDist)

class NodeNotReadyToPredict(Exception):
    pass

class Node(object):
    """
    Represents a specific split or branch in the tree.
    """
    
    def __init__(self, tree, attr_name=None):
        
        # The number of samples this node has been trained on.
        self.n = 0
        
        # A reference to the container tree instance.
        self._tree = tree
        
        # The splitting attribute at this node.
        self.attr_name = attr_name
        
        #### Discrete values.
        
        # Counts of each observed attribute value, used to calculate an
        # attribute value's probability.
        # {attr_name:{attr_value:count}}
        self._attr_value_counts = defaultdict(_get_dd_int)
        # {attr_name:total}
        self._attr_value_count_totals = defaultdict(int)
        
        # Counts of each observed class value and attribute value in
        # combination, used to calculate an attribute value's entropy.
        # {attr_name:{attr_value:{class_value:count}}}
        self._attr_class_value_counts = defaultdict(_get_dd_dd_int)
        
        #### Continuous values.
        
        # Counts of each observed class value, used to calculate a class
        # value's probability.
        # {class_value:count}
        self._class_ddist = DDist()
        
        # {attr_name:{attr_value:CDist(variance)}}
        self._attr_value_cdist = defaultdict(_get_dd_cdist)
        self._class_cdist = CDist()
        
        self._branches = {} # {v:Node}
    
    def __getitem__(self, attr_name):
        assert attr_name == self.attr_name
        branches = self._branches.copy()
        for value in self.get_values(attr_name):
            if value in branches:
                continue
            elif self.tree.data.is_continuous_class:
                branches[value] = self._attr_value_cdist[self.attr_name][value].copy()
            else:
                branches[value] = self.get_value_ddist(self.attr_name, value)
        return branches

    def _get_attribute_value_for_node(self, record):
        """
        Gets the closest value for the current node's attribute matching the
        given record.
        """
        
        # Abort if this node has not get split on an attribute. 
        if self.attr_name is None:
            return
        
        # Otherwise, lookup the attribute value for this node in the
        # given record.
        attr = self.attr_name
        attr_value = record[attr]
        attr_values = self.get_values(attr)
        if attr_value in attr_values:
            return attr_value
        else:
            # The value of the attribute in the given record does not directly
            # map to any previously known values, so apply a missing value
            # policy.
            policy = self.tree.missing_value_policy.get(attr)
            # assert policy, \
            #     ("No missing value policy specified for attribute %s.") \
            #     % (attr,)
            if policy == USE_NEAREST:
                # print("succesfully got nearest")
                # Use the value that the tree has seen that's also has the
                # smallest Euclidean distance to the actual value.
                assert self.tree.data.header_types[attr] \
                    in (ATTR_TYPE_DISCRETE, ATTR_TYPE_CONTINUOUS), \
                    "The use-nearest policy is invalid for nominal types."
                nearest = (1e999999, None)
                for _value in attr_values:
                    nearest = min(
                        nearest,
                        (abs(_value - attr_value), _value))
                _, nearest_value = nearest
                return nearest_value

            if policy == USE_MOST:
                # print("succesfully got most")
                # Use the value that the tree has seen that's also has the
                # smallest Euclidean distance to the actual value.
                assert self.tree.data.header_types[attr] \
                    in (ATTR_TYPE_NOMINAL, ATTR_TYPE_CONTINUOUS), \
                    "The use-nearest policy is invalid for nominal types."
                most = (None, 0)
                counts = defaultdict(int)
                for _value in attr_values:
                    counts[_value] += 1
                for value, count in enumerate(counts.values()):
                    if count >= most[1]:
                        most = (value, count)
                _, most_value = most
                return most_value
            else:
                raise Exception("Unknown missing value policy: %s" % (policy,))

    @property
    def attributes(self):
        return iterkeys(self._attr_value_counts)
    
    def get_values(self, attr_name):
        """
        Retrieves the unique set of values seen for the given attribute
        at this node.
        """
        ret = list(self._attr_value_cdist[attr_name].keys()) \
            + list(self._attr_value_counts[attr_name].keys()) \
            + list(self._branches.keys())
        ret = set(ret)
        return ret
    
    @property
    def is_continuous_class(self):
        return self._tree.is_continuous_class

    def get_best_splitting_attr(self):
        """
        Returns the name of the attribute with the highest gain.
        """
        best = (-1e999999, None)
        for attr in self.attributes:
            best = max(best, (self.get_gain(attr), attr))
        best_gain, best_attr = best
        return best_attr

    def get_entropy(self, attr_name=None, attr_value=None):
        """
        Calculates the entropy of a specific attribute/value combination.
        """
        is_con = self.tree.data.is_continuous_class
        if is_con:
            if attr_name is None:
                # Calculate variance of class attribute.
                var = self._class_cdist.variance
            else:
                # Calculate variance of the given attribute.
                var = self._attr_value_cdist[attr_name][attr_value].variance
            if self.tree.metric == VARIANCE1 or attr_name is None:
                return var
            elif self.tree.metric == VARIANCE2:
                unique_value_count = len(self._attr_value_counts[attr_name])
                attr_total = float(self._attr_value_count_totals[attr_name])
                return var*(unique_value_count/attr_total)
        else:
            if attr_name is None:
                # The total number of times this attr/value pair has been seen.
                total = float(self._class_ddist.total)
                # The total number of times each class value has been seen for
                # this attr/value pair.
                counts = self._class_ddist.counts
                # The total number of unique values seen for this attribute.
                unique_value_count = len(self._class_ddist.counts)
                # The total number of times this attribute has been seen.
                attr_total = total
            else:
                total = float(self._attr_value_counts[attr_name][attr_value])
                counts = self._attr_class_value_counts[attr_name][attr_value]
                unique_value_count = len(self._attr_value_counts[attr_name])
                attr_total = float(self._attr_value_count_totals[attr_name])
            assert total, "There must be at least one non-zero count."
            
            # n = max(2, len(counts))
            n = 4
            if self._tree.metric == ENTROPY1:
                # Traditional entropy.
                return -sum(
                    (count/total)*math.log(count/total, n)
                    for count in itervalues(counts)
                )
            elif self._tree.metric == ENTROPY2:
                # Modified entropy that down-weights universally unique values.
                # e.g. If the number of unique attribute values equals the total
                # count of the attribute, then it has the maximum amount of unique
                # values.
                return -sum(
                    (count/total)*math.log(count/total, n)
                    for count in itervalues(counts)
                #) - ((len(counts)-1)/float(total))
                ) + (unique_value_count/attr_total)
            elif self._tree.metric == ENTROPY3:
                # Modified entropy that down-weights universally unique values
                # as well as features with large numbers of values.
                return -sum(
                    (count/total)*math.log(count/total, n)
                    for count in itervalues(counts)
                #) - 100*((len(counts)-1)/float(total))
                ) + 100*(unique_value_count/attr_total)
        
    def get_gain(self, attr_name):
        """
        Calculates the information gain from splitting on the given attribute.
        entrance for criteria of different kinds of variables
        """
        subset_entropy = 0.0
        for value in iterkeys(self._attr_value_counts[attr_name]):
            value_prob = self.get_value_prob(attr_name, value)
            e = self.get_entropy(attr_name, value)
            subset_entropy += value_prob * e
        return (self.main_entropy - subset_entropy)

    def get_value_ddist(self, attr_name, attr_value):
        """
        Returns the class value probability distribution of the given
        attribute value.
        """
        assert not self.tree.data.is_continuous_class, \
            "Discrete distributions are only maintained for " + \
            "discrete class types."
        ddist = DDist()
        cls_counts = self._attr_class_value_counts[attr_name][attr_value]
        for cls_value, cls_count in iteritems(cls_counts):
            ddist.add(cls_value, count=cls_count)
        return ddist
    
    def get_value_prob(self, attr_name, value):
        """
        Returns the value probability of the given attribute at this node.
        """
        if attr_name not in self._attr_value_count_totals:
            return
        n = self._attr_value_counts[attr_name][value]
        d = self._attr_value_count_totals[attr_name]
        return n/float(d)

    @property
    def main_entropy(self):
        """
        Calculates the overall entropy of the class attribute.
        """
        return self.get_entropy()
    
    def predict(self, record, depth=0):
        """
        Returns the estimated value of the class attribute for the given
        record.
        """
        
        # Check if we're ready to predict.
        if not self.ready_to_predict:
            raise NodeNotReadyToPredict
        
        # Lookup attribute value.
        attr_value = self._get_attribute_value_for_node(record)
        
        # Propagate decision to leaf node.
        if self.attr_name:
            if attr_value in self._branches:
                try:
                    return self._branches[attr_value].predict(record, depth=depth+1)
                except NodeNotReadyToPredict:
                    #TODO:allow re-raise if user doesn't want an intermediate prediction?
                    pass
                
        # Otherwise make decision at current node.
        if self.attr_name:
            if self._tree.data.is_continuous_class:
                return self._attr_value_cdist[self.attr_name][attr_value].copy()
            else:
#                return self._class_ddist.copy()
                return self.get_value_ddist(self.attr_name, attr_value)
        elif self._tree.data.is_continuous_class:
            # Make decision at current node, which may be a true leaf node
            # or an incomplete branch in a tree currently being built.
            assert self._class_cdist is not None
            return self._class_cdist.copy()
        else:
            return self._class_ddist.copy()

    @property
    def ready_to_predict(self):
        return self.n > 0

    @property
    def ready_to_split(self):
        """
        Returns true if this node is ready to branch off additional nodes.
        Returns false otherwise.
        """
        # Never split if we're a leaf that predicts adequately.
        threshold = self._tree.leaf_threshold
        if self._tree.data.is_continuous_class:
            var = self._class_cdist.variance
            if var is not None and threshold is not None \
            and var <= threshold:
                return False
        else:
            best_prob = self._class_ddist.best_prob
            if best_prob is not None and threshold is not None \
            and best_prob >= threshold:
                return False
            
        return self._tree.auto_grow \
            and not self.attr_name \
            and self.n >= self._tree.splitting_n

    def set_leaf_dist(self, attr_value, dist):
        """
        Sets the probability distribution at a leaf node.
        """
        assert self.attr_name
        assert self.tree.data.is_valid(self.attr_name, attr_value), \
            "Value %s is invalid for attribute %s." \
                % (attr_value, self.attr_name)
        if self.is_continuous_class:
            assert isinstance(dist, CDist)
            assert self.attr_name
            self._attr_value_cdist[self.attr_name][attr_value] = dist.copy()
#            self.n += dist.count
        else:
            assert isinstance(dist, DDist)
            # {attr_name:{attr_value:count}}
            self._attr_value_counts[self.attr_name][attr_value] += 1
            # {attr_name:total}
            self._attr_value_count_totals[self.attr_name] += 1
            # {attr_name:{attr_value:{class_value:count}}}
            for cls_value, cls_count in iteritems(dist.counts):
                self._attr_class_value_counts[self.attr_name][attr_value] \
                    [cls_value] += cls_count
    
    def to_dict(self):
        if self.attr_name:
            # Show a value's branch, whether it's a leaf or another node.
            ret = {self.attr_name:{}} # {attr_name:{attr_value:dist or node}}
            values = self.get_values(self.attr_name)
            for attr_value in values:
                if attr_value in self._branches:
                    ret[self.attr_name][attr_value] = self._branches[attr_value].to_dict()
                elif self._tree.data.is_continuous_class:
                    ret[self.attr_name][attr_value] = \
                        self._attr_value_cdist[self.attr_name][attr_value].copy()
                else:
                    ret[self.attr_name][attr_value] = \
                        self.get_value_ddist(self.attr_name, attr_value)
            return ret
        elif self.tree.data.is_continuous_class:
            # Otherwise we're at a continuous leaf node.
            return self._class_cdist.copy()
        else:
            # Or a discrete leaf node.
            return self._class_ddist.copy()

    @property
    def tree(self):
        return self._tree

    def train(self, record):
        """
        Incrementally update the statistics at this node.
        """
        self.n += 1
        class_attr = self.tree.data.class_attribute_name
        class_value = record[class_attr]
        
        # Update class statistics.
        is_con = self.tree.data.is_continuous_class
        if is_con:
            # For a continuous class.
            self._class_cdist += class_value
        else:
            # For a discrete class.
            self._class_ddist.add(class_value)
        
        # Update attribute statistics.
        for an, av in iteritems(record):
            if an == class_attr:
                continue
            self._attr_value_counts[an][av] += 1
            self._attr_value_count_totals[an] += 1
            if is_con:
                self._attr_value_cdist[an][av] += class_value
            else:
                self._attr_class_value_counts[an][av][class_value] += 1
        
        # Decide if branch should split on an attribute.
        if self.ready_to_split:
            self.attr_name = self.get_best_splitting_attr()
            self.tree.leaf_count -= 1
            for av in self._attr_value_counts[self.attr_name]:
                self._branches[av] = Node(tree=self.tree)
                self.tree.leaf_count += 1
            
        # If we've split, then propagate the update to appropriate sub-branch.
        if self.attr_name:
            key = record[self.attr_name]
            del record[self.attr_name]
            self._branches[key].train(record)

class Tree(object):
    """
    Represents a single grown or built decision tree.
    """
    
    def __init__(self, data, **kwargs):
        assert isinstance(data, Data)
        self._data = data

        self.splitter = Splitter()
        
        # Root splitting node.
        # This can be traversed via [name1][value1][name2][value2]...
        self._tree = Node(self)
        
        # The mean absolute error.
        self.mae = CDist()
        self._mae_clean = True
        
        # Set the metric used to calculate the information gain
        # after an attribute split.
        if self.data.is_continuous_class:
            self.metric = kwargs.get('metric', DEFAULT_CONTINUOUS_METRIC)
            assert self.metric in CONTINUOUS_METRICS
        else:
            self.metric = kwargs.get('metric', DEFAULT_DISCRETE_METRIC)
            assert self.metric in DISCRETE_METRICS
            
        # Set metric to splitting nodes after a sample threshold has been met.
        self.splitting_n = kwargs.get('splitting_n', 100)
        
        # Declare the policy for handling missing values for each attribute.
        self.missing_value_policy = {}

        for attr in data.attribute_names:
            if self.data.header_types[attr] == DIS:
                self.missing_value_policy[attr] = USE_NEAREST
            if self.data.header_types[attr] == NOM:
                self.missing_value_policy[attr] = USE_MOST
        # Allow the tree to automatically grow and split after an update().
        self.auto_grow = kwargs.get('auto_grow', False)
        
        # Determine the threshold at which further splitting is unnecessary
        # if enough accuracy has been achieved.
        if self.data.is_continuous_class:
            # Zero variance is the default continuous stopping criteria.
            self.leaf_threshold = kwargs.get('leaf_threshold', 0.0)
        else:
            # A 100% probability is the default discrete stopping criteria.
            self.leaf_threshold = kwargs.get('leaf_threshold', 1.0)
            
        # The total number of leaf nodes.
        self.leaf_count = 0
        
        # The total number of samples trained on.
        self.sample_count = 0
        
        ### Used for forests.
        
        # The prediction accuracy on held-out samples.
        self.out_of_bag_accuracy = CDist()
        
        # Samples not given to the tree for training with which the
        # out-of-bag accuracy is calculated from.
        self._out_of_bag_samples = []
        
        # The mean absolute error for predictions on out-of-bag samples.
        self._out_of_bag_mae = CDist()
        self._out_of_bag_mae_clean = True
    
    def __getitem__(self, attr_name):
        return self.tree[attr_name]

    @classmethod
    def build(cls, data, coe=0, costs=defaultdict(), max_cost=None, max_feature=30, *args, **kwargs):
        """
        Constructs a classification or regression tree in a single batch by
        analyzing the given data.
        """
        assert isinstance(data, Data)
        if data.is_continuous_class:
            fitness_func = gain_variance
        else:
            fitness_func = get_gain

        t = cls(data=data, *args, **kwargs)
        t._data = data
        t.sample_count = len(data)
        t.splitter = Splitter(costs=costs, attributes=data.attribute_names)
        t._tree = t.splitter.create_decision_tree(
            data=data,
            attributes=data.attribute_names,
            class_attr=data.class_attribute_name,
            fitness_func=fitness_func,
            wrapper=t,
            coe=coe,
            costs=costs,
            max_cost=max_cost,
            max_feature=max_feature
        )
        return t
    
    @property
    def data(self):
        return self._data
    
    @property
    def is_continuous_class(self):
        return self.data.is_continuous_class
    
    @classmethod
    def load(cls, fn):
        tree = pickle.load(open(fn))
        assert isinstance(tree, cls), "Invalid pickle."
        return tree
    
    @property
    def out_of_bag_mae(self):
        """
        Returns the mean absolute error for predictions on the out-of-bag
        samples.
        """
        if not self._out_of_bag_mae_clean:
            try:
                self._out_of_bag_mae = self.test(self.out_of_bag_samples)
                self._out_of_bag_mae_clean = True
            except NodeNotReadyToPredict:
                return
        return self._out_of_bag_mae.copy()
    
    @property
    def out_of_bag_samples(self):
        """
        Returns the out-of-bag samples list, inside a wrapper to keep track
        of modifications.
        """
        #TODO:replace with more a generic pass-through wrapper?
        class O(object):
            def __init__(self, tree):
                self.tree = tree
            def __len__(self):
                return len(self.tree._out_of_bag_samples)
            def append(self, v):
                self.tree._out_of_bag_mae_clean = False
                return self.tree._out_of_bag_samples.append(v)
            def pop(self, v):
                self.tree._out_of_bag_mae_clean = False
                return self.tree._out_of_bag_samples.pop(v)
            def __iter__(self):
                for _ in self.tree._out_of_bag_samples:
                    yield _
        return O(self)

    def predict(self, record):
        record = record.copy()
        return self._tree.predict(record)
    
    def save(self, fn):
        pickle.dump(self, open(fn, 'w'))
    
    def set_missing_value_policy(self, policy, target_attr_name=None):
        """
        Sets the behavior for one or all attributes to use when traversing the
        tree using a query vector and it encounters a branch that does not
        exist.
        """
        assert policy in MISSING_VALUE_POLICIES, \
            "Unknown policy: %s" % (policy,)
        for attr_name in self.data.attribute_names:
            if target_attr_name is not None and target_attr_name != attr_name:
                continue
            self.missing_value_policy[attr_name] = policy

    def test(self, data):
        """
        Iterates over the data, classifying or regressing each element and then
        finally returns the classification accuracy or mean-absolute-error.
        """
#        assert data.header_types == self._data.header_types, \
#            "Test data schema does not match the tree's schema."
        is_cont = self._data.is_continuous_class
        agg = CDist()
        for record in data:
            actual_value = self.predict(record)
            expected_value = record[self._data.class_attribute_name]
            if is_cont:
                assert isinstance(actual_value, CDist)
                actual_value = actual_value.mean
                try:
                    agg += abs(actual_value - expected_value)
                except TypeError:
                    agg += 0
            else:
                assert isinstance(actual_value, DDist)
                agg += actual_value.best == expected_value
        return agg
    
    def to_dict(self):
        return self._tree.to_dict()
    
    @property
    def tree(self):
        return self._tree
    
    def train(self, record):
        """
        Incrementally updates the tree with the given sample record.
        """
        assert self.data.class_attribute_name in record, \
            "The class attribute must be present in the record."
        record = record.copy()
        self.sample_count += 1
        self.tree.train(record)

def _get_defaultdict_cdist():
    return defaultdict(CDist)

class Forest(object):
    
    def __init__(self, data, tree_kwargs=None, **kwargs):
        assert isinstance(data, Data)
        self._data = data
        
        # The population of trees.
        self.trees = []
        
        # Arguments that will be passed to each tree during init.
        self.tree_kwargs = tree_kwargs or {}
        
        self.grow_method = kwargs.get('grow_method', GROW_RANDOM)
        assert self.grow_method in GROW_METHODS, \
            "Growth method %s is not supported." % (self.grow_method,)
        
        # The number of trees in the forest.
        self.size = kwargs.get('size', 10)
        
        # The ratio of training samples given to each tree.
        # The rest are held to test tree accuracy.
        self.sample_ratio = kwargs.get('sample_ratio', 0.9)
        
        # The maximum number of out of bag samples to store in each tree.
        self.max_out_of_bag_samples = \
            kwargs.get('max_out_of_bag_samples', 1000)
        
        # The method for how we consolidate each tree's prediction into
        # a single prediction.
#        self.aggregation_method = kwargs.get('aggregation_method', WEIGHTED_MEAN)
#        assert self.aggregation_method in AGGREGATION_METHODS, \
#            "Aggregation method %s is not supported." \
#                % (self.aggregation_method,)
        self.weighting_method = kwargs.get('weighting_method',
            Forest.mean_oob_mae_weight)
                
        # The criteria defining how and when old trees are removed from the
        # forest and replaced by new trees.
        # This is a callable that is given a list of all the current trees
        # and returns a list of trees that should be removed.
        self.fell_method = kwargs.get('fell_method', None)
    
    def _fell_trees(self):
        """
        Removes trees from the forest according to the specified fell method.
        """
        if callable(self.fell_method):
            for tree in self.fell_method(list(self.trees)):
                self.trees.remove(tree)
    
    def _get_best_prediction(self, record, train=True):
        """
        Gets the prediction from the tree with the lowest mean absolute error.
        """
        if not self.trees:
            return
        best = (+1e999999, None)
        for tree in self.trees:
            best = min(best, (tree.mae.mean, tree))
        _, best_tree = best
        prediction, tree_mae = best_tree.predict(record, train=train)
        return prediction.mean
    
    @staticmethod
    def best_oob_mae_weight(trees):
        """
        Returns weights so that the tree with smallest out-of-bag mean absolute error
        """
        best = (+1e999999, None)
        for tree in trees:
            oob_mae = tree.out_of_bag_mae
            if oob_mae is None or oob_mae.mean is None:
                continue
            best = min(best, (oob_mae.mean, tree))
        best_mae, best_tree = best
        if best_tree is None:
            return
        return [(1.0, best_tree)]
        
    @staticmethod
    def mean_oob_mae_weight(trees):
        """
        Returns weights proportional to the out-of-bag mean absolute error for each tree.
        """
        weights = []
        active_trees = []
        for tree in trees:
            oob_mae = tree.out_of_bag_mae
            if oob_mae is None or oob_mae.mean is None:
                continue
            weights.append(oob_mae.mean)
            active_trees.append(tree)
        if not active_trees:
            return
        weights = normalize(weights)
        return zip(weights, active_trees)
    
    def _grow_trees(self):
        """
        Adds new trees to the forest according to the specified growth method.
        """
        if self.grow_method == GROW_AUTO_INCREMENTAL:
            self.tree_kwargs['auto_grow'] = True
        
        while len(self.trees) < self.size:
            self.trees.append(Tree(data=self.data, **self.tree_kwargs))
    
    @property
    def data(self):
        return self._data
        
    def predict(self, record):
        """
        Attempts to predict the value of the class attribute by aggregating
        the predictions of each tree.
        
        Parameters:
            weighting_formula := a callable that takes a list of trees and
                returns a list of weights.
        """
        
        # Get raw predictions.
        # {tree:raw prediction}
        predictions = {}
        for tree in self.trees:
            _p = tree.predict(record)
            if _p is None:
                continue
            if isinstance(_p, CDist):
                if _p.mean is None:
                    continue
            elif isinstance(_p, DDist):
                if not _p.count:
                    continue
            predictions[tree] = _p
        if not predictions:
            return

        # Normalize weights and aggregate final prediction.
        weights = self.weighting_method(predictions.keys())
        if not weights:
            return
#        assert sum(weights) == 1.0, "Sum of weights must equal 1."
        if self.data.is_continuous_class:
            # Merge continuous class predictions.
            total = sum(w*predictions[tree].mean for w, tree in weights)
        else:
            # Merge discrete class predictions.
            total = DDist()
            for weight, tree in weights:
                prediction = predictions[tree]
                for cls_value, cls_prob in prediction.probs:
                    total.add(cls_value, cls_prob*weight)
        
        return total
    
    def set_missing_value_policy(self, policy, target_attr_name=None):
        for tree in self.trees:
            tree.set_missing_value_policy(policy, target_attr_name)
    
    def test(self, data):
        """
        Iterates over the data, classifying or regressing each element and then
        finally returns the classification accuracy or mean-absolute-error.
        """
#        assert data.header_types == self._data.header_types, \
#            "Test data schema does not match the tree's schema."
        is_cont = self.data.is_continuous_class
        agg = CDist()
        for record in data:
            actual_value = self.predict(record)
            if actual_value is None:
                continue
            expected_value = record[self._data.class_attribute_name]
            if is_cont:
                assert isinstance(actual_value, CDist), \
                    "Invalid prediction type: %s" % (type(actual_value),)
                actual_value = actual_value.mean
                agg += abs(actual_value - expected_value)
            else:
                assert isinstance(actual_value, DDist), \
                    "Invalid prediction type: %s" % (type(actual_value),)
                agg += actual_value.best == expected_value
        return agg
    
    def train(self, record):
        """
        Updates the trees with the given training record.
        """
        self._fell_trees()
        self._grow_trees()
        for tree in self.trees:
            if random.random() < self.sample_ratio:
                tree.train(record)
            else:
                tree.out_of_bag_samples.append(record)
                while len(tree.out_of_bag_samples) > self.max_out_of_bag_samples:
                    tree.out_of_bag_samples.pop(0)

class Test(unittest.TestCase):

    def test_stat(self):
        print('Testing statistics classes...')
        nums = range(1, 10)
        s = CDist()
        seen = []
        for n in nums:
            seen.append(n)
            s += n
            print('mean:', s.mean)
            print('variance:', get_variance(seen))
            print('variance:', s.variance)
        self.assertAlmostEqual(s.mean, get_mean(nums), 1)
        self.assertAlmostEqual(s.variance, get_variance(nums), 2)
        self.assertEqual(s.count, 9)
#        print(s.mean
#        print(s.standard_deviation
        self.assertAlmostEqual(s.probability_lt(s.mean-s.standard_deviation*6), 0.0, 5)
        self.assertAlmostEqual(s.probability_lt(s.mean+s.standard_deviation*6), 1.0, 5)
        self.assertAlmostEqual(s.probability_gt(s.mean-s.standard_deviation*6), 1.0, 5)
        self.assertAlmostEqual(s.probability_gt(s.mean+s.standard_deviation*6), 0.0, 5)
        self.assertAlmostEqual(s.probability_in(s.mean, 50), 0.5, 5)
        
        d1 = DDist(['a', 'b', 'a', 'a', 'b'])
        d2 = DDist(['a', 'b', 'a', 'a', 'b'])
        d3 = DDist(['a', 'b', 'a', 'a', 'b', 'c'])
        self.assertEqual(d1, d2)
        self.assertNotEqual(d1, d3)
        self.assertNotEqual(d2, d3)
        self.assertEqual(d1.best, 'a')
        self.assertEqual(d1.best_prob, 3/5.)
        self.assertEqual(d2.best, 'a')
        self.assertEqual(d3.best, 'a')
        
        print('Done.')

    def test_data(self):
        print('Testing data class...')
        
        # Load data from a file.
        data = Data('rdata1')
        self.assertEqual(len(data), 16)
        data = list(Data('rdata1'))
        self.assertEqual(len(data), 16)
#        for row in data:
#            print(row

        # Load data from memory or some other arbitrary source.
        data = """a,b,c,d,cls
1,1,1,1,a
1,1,1,2,a
1,1,2,3,a
1,1,2,4,a
1,2,3,5,a
1,2,3,6,a
1,2,4,7,a
1,2,4,8,a
2,3,5,1,b
2,3,5,2,b
2,3,6,3,b
2,3,6,4,b
2,4,7,5,b
2,4,7,6,b
2,4,8,7,b
2,4,8,8,b""".strip().split('\n')
        rows = list(csv.DictReader(data))
        self.assertEqual(len(rows), 16)
        
        rows = Data(
            #csv.DictReader(data),
            #map(lambda r: r.split(','), data[1:]),
            [r.split(',') for r in data[1:]],
            order=['a', 'b', 'c', 'd', 'cls'],
            types=dict(a=DIS, b=DIS, c=DIS, d=DIS, cls=NOM),
            modes=dict(cls=CLS))
        self.assertEqual(len(rows), 16)
        self.assertEqual(len(list(rows)), 16)
        for row in rows:
            print(row)
            
        a, b = rows.split(ratio=0.1)
        self.assertEqual(len(rows), len(a)+len(b))
        print('-'*80)
        print('a:')
        for row in a:
            print(row)
        print('-'*80)
        print('b:')
        for row in b:
            print(row)
            
        print('Done.')

    def test_batch_tree(self):
        print('Testing batch tree...')
        
        # If we set no leaf threshold for a continuous class
        # then there will be the same number of leaf nodes
        # as there are number of records.
        t = Tree.build(Data('rdata2'))
        self.assertEqual(type(t), Tree)
        #pprint(t._tree, indent=4)
        print("Tree:")
        pprint(t.to_dict(), indent=4)
        self.assertEqual(set(t._tree['b'].keys()), set([1, 2, 3, 4]))
        result = t.test(Data('rdata1'))
        self.assertEqual(type(result), CDist)
        print('MAE:', result.mean)
        self.assertAlmostEqual(result.mean, 0.001368, 5)
        self.assertEqual(t.leaf_count, 16)
        
        # If we set a leaf threshold, then this will limit the number of leaf
        # nodes created, speeding up prediction, at the expense of increasing
        # the mean absolute error.
        t = Tree.build(Data('rdata2'), leaf_threshold=0.0005)
        print("Tree:")
        pprint(t.to_dict(), indent=4)
        print(t._tree['b'].keys())
        self.assertEqual(t._tree.get_values('b'), set([1, 2, 3, 4]))
        result = t.test(Data('rdata1'))
        print('MAE:', result.mean)
        self.assertAlmostEqual(result.mean, 0.00623, 5)
        self.assertEqual(t.leaf_count, 10)
        
        t = Tree.build(Data('cdata1'))
        print("Tree:")
        self.assertEqual(t['Age']['36 - 55'].attr_name, 'Marital Status')
        self.assertEqual(t['Age']['36 - 55']\
            .get_values('Marital Status'), set(['single', 'married']))
        self.assertEqual(set(t['Age'].keys()), set(['< 18', '18 - 35', '36 - 55', '> 55']))
        self.assertEqual(t['Age']['18 - 35'].best, 'won\'t buy')
        self.assertEqual(t['Age']['36 - 55']['Marital Status']['single'].best, 'will buy')
#        return
        d = t.to_dict()
        pprint(d, indent=4)
#        return
        result = t.test(Data('cdata1'))
        print('Accuracy:', result.mean)
        self.assertAlmostEqual(result.mean, 1.0, 5)
        
        t = Tree.build(Data('cdata2'))
        pprint(t.to_dict(), indent=4)
        result = t.test(Data('cdata2'))
        print('Accuracy:', result.mean)
        self.assertAlmostEqual(result.mean, 1.0, 5)
        result = t.test(Data('cdata3'))
        print('Accuracy:', result.mean)
        self.assertAlmostEqual(result.mean, 0.75, 5)
        
        # Send it a corpus that's purposefully difficult to predict.
        t = Tree.build(Data('cdata4'))
        pprint(t.to_dict(), indent=4)
        result = t.test(Data('cdata4'))
        print('Accuracy:', result.mean)
        self.assertAlmostEqual(result.mean, 0.5, 5)
        
        # Send it a case it's never seen.
        with self.assertRaises(AssertionError):
            # By default, it should throw an exception because it hasn't been
            # given a policy for resolving unseen attribute value.
            t.predict(dict(a=1, b=2, c=3, d=4))
        # But if we tell it to use the nearest value, then it should pass.
        t.set_missing_value_policy(USE_NEAREST)
        result = t.predict(dict(a=1, b=2, c=3, d=4))
        print(result)
        print('Done.')

    def test_online_tree(self):
        print('Testing online tree...')
        
        rdata3 = Data('rdata3')
        rdata3_lst = list(rdata3)
        
        cdata2 = Data('cdata2')
        cdata2_lst = list(cdata2)
        
        cdata5 = Data('cdata5')
        cdata5_lst = list(cdata5)
        
        tree = Tree(cdata2, metric=ENTROPY1)
        for row in cdata2:
#            print(row
            tree.train(row)
        node = tree._tree
        attr_gains = [(node.get_gain(attr_name), attr_name) for attr_name in node.attributes]
        attr_gains.sort()
#        print(attr_gains
        # With traditional entropy, a b and c all evenly divide the class
        # and therefore have the same gain, even though all three
        # have different value frequencies.
        self.assertEqual(attr_gains,
            [(0.0, 'd'), (1.0, 'a'), (1.0, 'b'), (1.0, 'c')])
        
        tree = Tree(cdata2, metric=ENTROPY2)
        for row in cdata2:
#            print(row
            tree.train(row)
        self.assertEqual(set(node.attributes), set(['a', 'b', 'c', 'd']))
        node = tree._tree
        attr_gains = [(node.get_gain(attr_name), attr_name) for attr_name in node.attributes]
        attr_gains.sort()
#        print(attr_gains
        # With entropy metric 2, attributes that have fewer unique values
        # will have a slightly greater gain relative to attributes with more
        # unique values.
        self.assertEqual(attr_gains,
            [(-0.375, 'd'), (0.625, 'c'), (0.875, 'b'), (1.0, 'a')])
        
        tree = Tree(rdata3, metric=VARIANCE1)
        for row in rdata3:
#            print(row
            tree.train(row)
        node = tree._tree
        self.assertEqual(set(node.attributes), set(['a', 'b', 'c', 'd']))
        attr_gains = [(node.get_gain(attr_name), attr_name) for attr_name in node.attributes]
        attr_gains.sort()
#        print(attr_gains
        # With entropy metric 2, attributes that have fewer unique values
        # will have a slightly greater gain relative to attributes with more
        # unique values.
        self.assertEqual([v for _, v in attr_gains], ['d', 'a', 'b', 'c'])
        
        tree = Tree(rdata3, metric=VARIANCE2)
        for row in rdata3:
#            print(row
            tree.train(row)
        node = tree._tree
        self.assertEqual(set(node.attributes), set(['a', 'b', 'c', 'd']))
        attr_gains = [(node.get_gain(attr_name), attr_name) for attr_name in node.attributes]
        attr_gains.sort()
#        print(attr_gains
        # With entropy metric 2, attributes that have fewer unique values
        # will have a slightly greater gain relative to attributes with more
        # unique values.
        self.assertEqual([v for _, v in attr_gains], ['d', 'c', 'b', 'a'])
        
        # Incrementally grow a classification tree.
        print("-"*80)
        print("Incrementally growing classification tree...")
        tree = Tree(cdata5, metric=ENTROPY2, splitting_n=17, auto_grow=True)
        for row in cdata5:
#            print(row
            tree.train(row)
        acc = tree.test(cdata5)
        print('Initial accuracy:', acc.mean)
        self.assertEqual(acc.mean, 0.25)
#        print('Current tree:'
#        pprint(tree.to_dict(), indent=4)
        # Update tree several times to give leaf nodes potential time to split.
        for _ in six.moves.range(5):
            for row in cdata5:
                #print(row
                tree.train(row)
            acc = tree.test(cdata5)
            print('Accuracy:', acc.mean)
        print('Final tree:')
        pprint(tree.to_dict(), indent=4)
        # Confirm no more nodes have split, since the optimal split has
        # already been found and the tree is fully grown.
        self.assertEqual(tree['b'][1].ready_to_split, False)
        self.assertEqual(tree['b'][1]._branches, {})
#        for attr in tree['b'][1].attributes:
#            print(attr, tree['b'][1].get_gain(attr)
        # Test accuracy of fully grown tree.
        acc = tree.test(cdata5)
        self.assertEqual(acc.mean, 1.0)
        
        # Incrementally grow a regression tree.
        print("-"*80)
        print("Incrementally growing regression tree...")
        tree = Tree(rdata3, metric=VARIANCE2, splitting_n=17, auto_grow=True, leaf_threshold=0.0)
        for row in rdata3:
#            print(row
            tree.train(row)
        mae = tree.test(rdata3)
        print('Initial MAE:', mae.mean)
        self.assertAlmostEqual(mae.mean, 0.4, 5)
        for _ in six.moves.range(20):
            for row in rdata3:
                #print(row
                tree.train(row)
            mae = tree.test(rdata3)
            print('MAE:', mae.mean)
        print("Final tree:")
        pprint(tree.to_dict(), indent=4)
        self.assertEqual(mae.mean, 0.0)
        print('Done.')

    def test_forest(self):
        print('Testing forest...')
        print('Growing forest incrementally...')
        
        cdata2 = Data('cdata2')
        cdata2_lst = list(cdata2)
        
        # Incrementally train and test the forest on the same data.
        forest = Forest(
            data=cdata2,
            size=10, # Grow 10 trees.
            sample_ratio=0.8, # Train each tree on 80% of all records.
            grow_method=GROW_AUTO_INCREMENTAL, # Incrementally grow each tree.
            #weighting_method=Forest.best_oob_mae_weight,
            weighting_method=Forest.mean_oob_mae_weight,
            tree_kwargs=dict(metric=ENTROPY2),
        )
        mae = None
        for _ in six.moves.range(10):
            for row in cdata2_lst:
                #print(row
                forest.train(row)
            mae = forest.test(cdata2_lst)
            print('Forest MAE:', mae.mean)
        self.assertEqual(mae.mean, 1.0)
        
        trees = list(forest.trees)
        trees.sort(key=lambda t: t.out_of_bag_mae.mean)
        print('Best tree:')
        pprint(trees[-1].to_dict(), indent=4)
        self.assertEqual(trees[-1].auto_grow, True)
#        for tree in trees:
#            pprint(tree.to_dict(), indent=4)
        print('Done.')
        
    def test_milksets(self):
        try:
            from milksets import wine, yeast
        except ImportError:
            print('Skipping milkset tests because milksets is not installed.')
            print('Run `sudo pip install milksets` and rerun these tests.')
            return
        
        def leave_one_out(all_data, metric=None):
            test_data, train_data = all_data.split(leave_one_out=True)
#            print('test:',len(test_data)
#            print('train:',len(train_data)
            tree = Tree.build(train_data, metric=metric)
            tree.set_missing_value_policy(USE_NEAREST)
            result = tree.test(test_data)
            return result.mean
        
        def cross_validate(all_data, epoches=10, test_ratio=0.25, metric=None):
            accuracies = []
            for epoche in six.moves.range(epoches):
#                print('Epoch:',epoche
                #test_data,train_data = all_data,all_data
                test_data, train_data = all_data.split(ratio=test_ratio)
#                print('\ttest:',len(test_data)
#                print('\ttrain:',len(train_data)
                tree = Tree.build(train_data, metric=metric)
                tree.set_missing_value_policy(USE_NEAREST)
                result = tree.test(test_data)
#                print('Epoch accuracy:',result.mean
                accuracies.append(result.mean)
            return sum(accuracies)/float(len(accuracies))
        
        # Load wine dataset.
        # Each record has 13 continuous features
        # and one discrete class containing 2 unique values.
        print('Loading UCI wine data...')
        wine_data = Data(
            [list(a)+[b] for a, b in zip(*wine.load())],
            order=map(str, range(13))+['cls'],
            #types=dict(a=DIS, b=DIS, c=DIS, d=DIS, cls=NOM),
            types=[CON]*13 + [DIS],
            modes=dict(cls=CLS))
        self.assertEqual(len(wine_data), 178)
        self.assertEqual(len(list(wine_data)), 178)
#        for row in wine_data:
#            print(row
            
        # Load yeast dataset.
        # Each record has 8 continuous features
        # and one discrete class containing 10 values.
        print('Loading UCI yeast data...')
        yeast_data = Data(
            [list(a)+[b] for a, b in zip(*yeast.load())],
            order=map(str, range(8))+['cls'],
            #types=dict(a=DIS, b=DIS, c=DIS, d=DIS, cls=NOM),
            types=[CON]*8 + [DIS],
            modes=dict(cls=CLS))
        self.assertEqual(len(yeast_data), 1484)
        self.assertEqual(len(list(yeast_data)), 1484)
        
        acc = leave_one_out(wine_data, metric=ENTROPY1)
        print('Wine leave-one-out accuracy: %0.2f' % (acc,))
        acc = cross_validate(wine_data, metric=ENTROPY1, test_ratio=0.01, epoches=25)
        print('Wine cross-validated accuracy: %0.2f' % (acc,))
        
        acc = leave_one_out(yeast_data, metric=ENTROPY1)
        print('Yeast leave-one-out accuracy: %0.2f' % (acc,))
        acc = cross_validate(yeast_data, metric=ENTROPY1, test_ratio=0.005, epoches=25)
        print('Yeast cross-validated accuracy: %0.2f' % (acc,))

    def test_entropy(self):
        # Lopsided distribution with mostly all events in one group
        # is low entropy.
        self.assertAlmostEqual(entropy({+1:10, -1:10, 0:980}), 0.1018576)
        # Everything in one group is 0 entropy.
        self.assertAlmostEqual(entropy({0:1000}), 0.0)
        # Everything equally divided is highest entropy.
        self.assertAlmostEqual(entropy({+1:500, -1:500}), 1.0)
#        
        data1 = {+1:1, -1:1}#,0:200-2}
        data2 = {+1:100, -1:100}
        
        # Entropy1 doesn't care about size.
        e11 = entropy(data1, method=ENTROPY1)
#        print(e11
        e21 = entropy(data2, method=ENTROPY1)
#        print(e21
        self.assertEqual(e11, 1.0)
        self.assertEqual(e11, e21)
        
        # Entropy2 takes size into account.
        e12 = entropy(data1, method=ENTROPY2)
#        print(e12
        e22 = entropy(data2, method=ENTROPY2)
#        print(e22
        self.assertEqual(e12, 0.5)
        self.assertEqual(e22, 0.995)
        
        # Entropy3 takes large numbers of values into account, but otherwise ignores size.
        e13 = entropy(data1, method=ENTROPY3)
#        print(e13
        e23 = entropy(data2, method=ENTROPY3)
#        print(e23
        self.assertEqual(e13, -49.0)
        self.assertEqual(e23, 0.5)

if __name__ == '__main__':
    unittest.main()
