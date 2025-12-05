import main.estimators as es
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils import resample
from tqdm import tqdm

# auxiliary functions


def hstack(t, w):
    return np.hstack([t.reshape(-1, 1), w])


def stack0(w):
    return np.hstack([np.zeros((w.shape[0], 1)), w])


def stack1(w):
    return np.hstack([np.ones((w.shape[0], 1)), w])


def fitTmodel(name, W, T):
    return es.create_classifier(name).fit(W, T)


def cv_score(mod, X, y):
    return np.round(
        cross_val_score(mod, X, y, scoring="neg_log_loss", cv=5).mean(), 3
    )


def run_analysis(df, bootstrap_reps=1000):
    # Format data
    On = df.to_numpy()
    T, W, Y = On[:, 0], On[:, 1:-1], On[:, -1]

    idx_init = np.zeros(W.shape[1] + 1)

    # Check model fits and generate plot of propensity scores
    best_params = {
        "l2pen": 0.001,
        "n_iter": 1500,
        "lr": 0.1,
        "hidden_dim": 100,
    }
    clf = es.IndexClassifier(**best_params)
    clf.fit(hstack(T, W), Y, initial=idx_init)
    Z = clf.partial_predict(W).reshape(-1, 1)
    Q = clf.predict_conditional(1, W).reshape(-1, 1)

    clfW = fitTmodel("Logistic", W, T)
    clfZ = fitTmodel("Logistic", Z, T)
    clfQ = fitTmodel("Logistic", Q, T)

    pW = clfW.predict_proba(W)[:, 1]
    pZ = clfZ.predict_proba(Z)[:, 1]
    pQ = clfQ.predict_proba(Q)[:, 1]
    props = pd.DataFrame(
        {
            "prop_W": pW,
            "prop_Z": pZ,
            "prop_Q": pQ,
        }
    )

    # Estimate interventional means and ATE
    im0s = []
    im1s = []
    Vars = []

    im0s.append(Y[T == 0].mean())
    im1s.append(Y[T == 1].mean())
    Vars.append(((2 * T - 1) * Y).var())

    # Classical estimators based on linear methods
    x, y, z = es.ATE_est_binY(T, W, Y, "Logistic", "Logistic")
    im0s += list(x)
    im1s += list(y)
    Vars += list(z)

    # Estimators based on SI network
    x, y, z = es.SI_ATE_binY(
        T,
        W,
        Y,
        index_dim=1,
        clf="Logistic",
        initial=idx_init,
        l2pen=best_params["l2pen"],
        n_iter=best_params["n_iter"],
        joint_training=True,
    )
    im0s += list(x)
    im1s += list(y)
    Vars += list(z)

    # Estimates for full data set
    IM0s = np.array(im0s)
    IM1s = np.array(im1s)
    ATEs = IM1s - IM0s
    Vars = np.array(Vars)

    # compute bootstrap resamples
    ATE_raw, ATE_reg, ATE_ipw = ([] for _ in range(3))
    ATE_aipw, ATE_pru = ([] for _ in range(2))
    (
        ATE_sir,
        ATE_siaipw,
        ATE_sioapw,
        ATE_siobpw,
    ) = ([] for _ in range(4))

    for ii in tqdm(range(bootstrap_reps)):
        T_, W_, Y_ = resample(T, W, Y, n_samples=len(T), random_state=ii)
        ATE_raw.append([es.unadjusted(T_, Y_, 0), es.unadjusted(T_, Y_, 1)])
        # Classical estimators based on linear methods
        x, y, _ = es.ATE_est_binY(T_, W_, Y_, "Logistic", "Logistic")
        ATE_ipw.append([x[0], y[0]])
        ATE_reg.append([x[1], y[1]])
        ATE_aipw.append([x[2], y[2]])
        ATE_pru.append([x[3], y[3]])

        # Estimators based on SI network
        x, y, _ = es.SI_ATE_binY(
            T_, W_, Y_, clf="Logistic", initial=idx_init, **best_params
        )
        ATE_sir.append([x[0], y[0]])
        ATE_siaipw.append([x[1], y[1]])
        ATE_sioapw.append([x[2], y[2]])
        ATE_siobpw.append([x[3], y[3]])

    df = pd.DataFrame(
        {
            "Naive contrast": ATE_raw,
            "IPW (Logistic)": ATE_ipw,
            "Regr. (Logistic)": ATE_reg,
            "AIPW (Logistic)": ATE_aipw,
            "DOPE-BCL (Logistic)": ATE_pru,
            "Regr. (NN)": ATE_sir,
            "AIPW (NN)": ATE_siaipw,
            "DOPE-IDX (NN)": ATE_sioapw,
            "DOPE-BCL (NN)": ATE_siobpw,
        }
    )

    dfm = df.melt(var_name="estimator", value_name="intmeans")
    dfm["chi0"] = dfm["intmeans"].apply(lambda x: x[0])
    dfm["chi1"] = dfm["intmeans"].apply(lambda x: x[1])
    dfm["ATE"] = dfm["intmeans"].apply(lambda x: x[1] - x[0])

    df_grp = pd.DataFrame(
        dfm.groupby("estimator")["ATE"].var().rename("BS var")
    )
    df_grp["BS se"] = np.sqrt(df_grp["BS var"])
    df_grp["BS lower"] = dfm.groupby("estimator")["ATE"].quantile(0.025)
    df_grp["BS upper"] = dfm.groupby("estimator")["ATE"].quantile(0.975)

    est_names = [
        "Naive contrast",
        "IPW (Logistic)",
        "Regr. (Logistic)",
        "AIPW (Logistic)",
        "DOPE-BCL (Logistic)",
        "Regr. (NN)",
        "AIPW (NN)",
        "DOPE-IDX (NN)",
        "DOPE-BCL (NN)",
    ]
    ATEs_pd = pd.Series(ATEs, index=est_names)
    df_grp.insert(0, "estimate", ATEs_pd.reindex_like(df_grp))

    return df_grp, props


df = pd.read_pickle("experiments/NHANES/data/NHANES_imputed.pkl")
sim_df, props = run_analysis(df)
sim_df.to_pickle("experiments/NHANES/results/NHANES_imputed_results.pkl")
props.to_pickle("experiments/NHANES/results/NHANES_imputed_props.pkl")

df = pd.read_pickle("experiments/NHANES/data/NHANES_removed.pkl")
sim_df, props = run_analysis(df)
sim_df.to_pickle("experiments/NHANES/results/NHANES_removed_results.pkl")
props.to_pickle("experiments/NHANES/results/NHANES_removed_props.pkl")
