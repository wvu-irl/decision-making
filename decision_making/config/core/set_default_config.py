#!/usr/bin/python
"""
This script is intended to update configuration parameters such as file paths and other default settings.
Some include :

"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

def set_defaults(root : str):
    pass

if __name__=='__main__':