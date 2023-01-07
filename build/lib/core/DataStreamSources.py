'''
DataStreamSources.py
    File to store enumerations of all data sources
    Author: Akshay Aravamudan, January 2020
'''
from enum import Enum


#####################################################
# ProcessName enumeration
#   This enumeration is meant to be a starting point for
#   generating the list of processes, New processes can be added dynamically
#   If it is deemed that this enumeration is unneceassary, it shall be removed and
#  replaces by an appropriate use to dictionary in the relevant files.
#####################################################

class ProcessName(Enum):
    EXPLOIT = 1
    TWITTER = 2
    REDDIT = 3
    GITHUB = 4
    NEWS = 5
    HEEBEJEBEE = 6
