from itertools import product

import experiments.simulation_study.sim_coverage as cov
import numpy as np
import pandas as pd
from tqdm import tqdm

n_reps = 100
n_samples = [2700]
d_list = [12]
links = ["cbrt"]
clfs = ["Logistic"]
joint_regression = [True]
crossfits = [False, True]
SEED = 42

sim_settings = pd.DataFrame.from_dict(
    [
        dict(zip(("rep", "n", "d", "link", "clf", "joint", "crossfit"), v))
        for v in product(
            *(
                range(n_reps),
                n_samples,
                d_list,
                links,
                clfs,
                joint_regression,
                crossfits,
            )
        )
    ]
)
res_list = []
rng = np.random.default_rng(seed=SEED)

for i, row in tqdm(sim_settings.iterrows(), total=sim_settings.shape[0]):
    n_sample = row["n"]
    d = row["d"]
    link = row["link"]
    clf = row["clf"]
    joint_regression = row["joint"]
    crossfit = row["crossfit"]

    res_list.append(
        cov.run_single_setting(
            n_sample=n_sample,
            d=d,
            link=link,
            classifier=clf,
            crossfit=crossfit,
            joint_regression=joint_regression,
            rng=rng,
        )
    )

sim_df = pd.concat((sim_settings, pd.DataFrame.from_records(res_list)), axis=1)
sim_df.to_pickle("experiments/simulation_study/results/coverage_results.pkl")
