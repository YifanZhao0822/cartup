#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from distutils.core import setup, Command # pylint: disable=no-name-in-module

import cartup

class TestCommand(Command):
    description = "Runs unittests."
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('python cartup.py')

setup(
    name='cartup',
    version=cartup.__version__,
    description='A simple CART-UP implementation for user preferences.',
    author='Yifan Zhao',
    author_email='yifanzhao0822@gmail.com',
    url='https://github.com/YifanZhao0822/cartup',
    license='LGPL',
    py_modules=['cartup'],
    #https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    platforms=['OS Independent'],
    cmdclass={
        'test': TestCommand,
    },
)
