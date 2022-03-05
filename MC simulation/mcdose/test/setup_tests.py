import sys, os
from os.path import join as pjoin

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setup path to allow imports from mcdose package
sys.path.insert(0, os.path.abspath(pjoin(os.path.dirname(__file__), os.path.pardir)))
print(sys.path)

# setup common testing data path
test_data = pjoin(os.path.dirname(__file__), 'test_data')
