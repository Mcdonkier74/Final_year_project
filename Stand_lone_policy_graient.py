import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

policy_gradient_dataframe = pd.read_csv(r"/home/kieran/Deep_learning_Policy_Gradient/data/Book3.csv")

policy_gradient_dataframe = policy_gradient_dataframe.reindex(np.random.permutation(policy_gradient_dataframe.index))

policy_gradient_dataframe
display(policy_gradient_dataframe)
policy_gradient_dataframe.describe()

# Define the input feature: 



