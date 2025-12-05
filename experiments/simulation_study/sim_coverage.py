import experiments.simulation_study.sim_estimators as sim
import numpy as np
import pandas as pd
from sklearn.utils import resample

# Run a single setting


def run_single_setting(
    n_sample=500,
    d=12,
    link="sin",
    classifier="Logistic",
    regressor="OLS",
    crossfit=True,
    joint_regression=True,
    rng=None,
    compute_IM_true=True,
    idxY=None,
    bootstrap_reps=200,
):

    idxY = np.r_[np.array([1, -2, 3]), np.zeros(d - 3)]

    W, T, Y, IM_true = sim.generate_data(
        n_sample=n_sample,
        d=d,
        link=link,
        compute_IM_true=compute_IM_true,
        rng=rng,
        idxY=idxY,
    )

    res = sim.estimate(
        W,
        T,
        Y,
        classifier=classifier,
        regressor=regressor,
        crossfit=crossfit,
        joint_regression=joint_regression,
    )

    bootstrap_res_list = []
    for ii in range(bootstrap_reps):
        T_, W_, Y_ = resample(T, W, Y, n_samples=len(T), random_state=ii)
        bootstrap_res = sim.estimate(
            W_,
            T_,
            Y_,
            classifier=classifier,
            regressor=regressor,
            crossfit=crossfit,
            joint_regression=joint_regression,
        )
        bootstrap_res_list.append(
            {
                key: value
                for key, value in bootstrap_res.items()
                if key[0:4] != "var_"
            }
        )
    boot_df = pd.DataFrame.from_records(bootstrap_res_list)
    boot_res = boot_df.std().to_dict()
    boot_res = {"bs_" + key: value for key, value in boot_res.items()}

    return {"IM": IM_true, **res, **boot_res}
