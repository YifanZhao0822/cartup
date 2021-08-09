CART-UP - A pure-Python implementation for CART with user preferences
=================================================================

Overview
--------

Given a training data set and user preferences for the features, it constructs a CART-UP for classification or
regression in a single batch or incrementally.

It loads data from CSV files. It expects the first row in the CSV to be a
header, with each element conforming to the pattern "name:type:mode".
Mode is optional, and denotes the class attribute. Type identifies the
attribute as either a continuous, discrete, or nominal.

The user preferences are fitted as a Python dict with the pattern "name:preference/cost level".

The pure-Python CART framework was based on Chris Spencer's repo: 
[Dtree - A simple pure-Python decision tree construction algorithm](https://github.com/chrisspen/dtree).
I added the user preference mechanism, and refactored the implementation to be more object-oriented.

All attributes and response variable can be continuous, discrete, and nominal. 
The only issue is that too many continuous attributes will make the training extremely slow.

Installation
------------

Download the code and then run:

    python setup.py build
    sudo python setup.py install

Usage
-----

Classification and regression are handled through the same interface, and
differ only in the object returned by the predict() method and how the result
from test() is interpreted.

With classification, this object will always be a DDist instance, representing
a probability distribution over a set of discrete or nominal classes. In this
case, the result from test() will be a CDist instance representing the
classification accuracy.

With regression, this object will always be a CDist instance, representing a
mean and variance. In this case, the result from test() will be a CDist
instance representing the mean absolute error.

To incorporate user preferences, format the user preferences to a dict with feature names as the keys and preference level as the values.
Then fit the dict to the parameter "costs" in Tree.build. 

Features
--------

- building a classification or regression tree using batch or incremental/online methods

Todo
----

Add pip installation means

Make more comments and README files to make the demos clearer.