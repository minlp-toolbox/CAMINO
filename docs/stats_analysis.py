# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from camino.settings import GlobalSettings
from camino.problems.problem_collection import create_ocp_unstable_system
import pickle

from camino.utils import latexify
from camino.utils.conversion import to_0d

filename = "2025-07-09_17:21:38_cia_unstable_ocp_generic.pkl"

file_path = os.path.join(GlobalSettings.OUT_DIR, filename)
with open(file_path, 'rb') as f:
    stats = pickle.load(f)

stats_df = pd.DataFrame(stats)
print(stats_df.columns)
print(stats_df.head())
