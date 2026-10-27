"""Inspect one FRP epoch file and the participant table (data only)."""
import sys

import mne
import pandas as pd

ep = mne.read_epochs(sys.argv[1], verbose="error")
print(ep)
print("channels:", ep.ch_names)
print("times:", ep.times[0], ep.times[-1], "sfreq", ep.info["sfreq"], "n", len(ep))
x = pd.read_excel(sys.argv[2])
print(x.head(8).to_string())
print(list(x.columns))
