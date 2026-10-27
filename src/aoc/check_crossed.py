"""Sanity check of the crossed specification on Study 4: variance components and one model."""
import sys
import warnings
from pathlib import Path

import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent))
from study4 import prepare  # noqa: E402

df = prepare()
df["all"] = 1
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = smf.mixedlm("lfp_fz ~ surprisal_z * years_c", df, groups=df["all"], re_formula="0",
                    vc_formula={"participant": "0 + C(participant)", "pair": "0 + C(pair)"}).fit(reml=True, method="lbfgs")
print(r.summary().tables[1])
print("vcomp", dict(zip(r.model.exog_vc.names, r.vcomp.round(3), strict=True)), "resid", round(r.scale, 3))
