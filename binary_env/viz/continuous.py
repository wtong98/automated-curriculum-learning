"""
Visualizations for continuous methods
"""

# <codecell>
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.append('../')
from env import *
from experiment import *

save_path = Path('fig/')