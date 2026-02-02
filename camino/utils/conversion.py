# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""A set of conversion tools."""

import numpy as np
import casadi as ca


def to_bool(val):
    """String or bool value to bool."""
    if isinstance(val, bool):
        return val
    else:
        return not (val.lower() in ["0", "false", "false", "no"])


def to_float(val):
    """To single float."""
    if isinstance(val, np.ndarray):
        return to_float(val[0])
    elif isinstance(val, list):
        return to_float(val[0])
    elif isinstance(val, ca.DM):
        return to_float(val.full())
    return val


def to_0d(array):
    """To zero dimensions."""
    if isinstance(array, np.ndarray):
        ret = np.atleast_1d(array.squeeze())
    elif isinstance(array, list):
        ret = np.atleast_1d(np.array(array).squeeze())
    elif isinstance(array, float):
        ret = np.atleast_1d(np.array(array))
    else:
        ret = np.atleast_1d(array.full().squeeze())
    return ret


def convert_to_flat_list(nr, indices, data):
    """Convert data to a flat list."""
    out = np.zeros((nr,))
    for key, indices in indices.items():
        values = data[key]
        if isinstance(indices, list):
            for idx, val in zip(indices, values):
                out[idx] = val
        else:
            out[indices] = values
    return out
