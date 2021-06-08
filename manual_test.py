#Test the network using a weights file provided from the command line
from rnn_test import performTest
import sys

#Check the argument has been provided
if len(sys.argv) == 2:
    performTest(sys.argv[1])
else:
    print("The weights file to be used must be passed as an argument")
