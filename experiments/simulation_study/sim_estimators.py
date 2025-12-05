import main.estimators as es
import numpy as np


def expit(x):
    return 1 / (1 + np.exp(-x))


eps_m0 = 0.01


def m0(w, _):
    return eps_m0 + (1 - 2 * eps_m0) * (w[:, 0] > 0.5)


def bump(w, idx):
    return (w[:, : len(idx) // 2] @ idx[: len(idx) // 2]) > 0.5 * (
        idx[: len(idx) // 2]
    ).sum()


def m1(w, idx):
    return eps_m0 + (1 - 2 * eps_m0) * (bump(w, idx))


def g_lin(t, z):
    return t + 3 * z


def g_sqr(t, z):
    return z ** (1 + t)


def g_sin(t, z):
    return (3 + t) * np.sin(np.pi * z)


def g_sinc(t, z):
    return (3 + t) * np.sinc(np.pi * z)


def g_cbrt(t, z):
    return (2 + t) * np.cbrt(z)


link_dic = {
    "lin": g_lin,
    "sin": g_sin,
    "sqr": g_sqr,
    "cbrt": g_cbrt,
    "sinc": g_sinc,
}


def sample_idx(rng, d=10):
    indexY = np.hstack([np.ones(1), rng.normal(size=(d - 1)) / np.sqrt(d - 1)])
    return indexY


def compute_IM(rng, indexY, link="sin", N=25000, t=1):
    d = len(indexY)
    g = link_dic[link]
    W = rng.uniform(0, 1, size=(N, d))
    return g(t * np.ones(N), W.dot(indexY)).mean()


def sample_SI(rng, indexY, n=500, link="sin", Ystd=1):
    d = len(indexY)
    g = link_dic[link]
    W = rng.uniform(0, 1, size=(n, d))
    T = rng.binomial(1, m0(W, indexY))
    Y = g(T, W.dot(indexY)) + Ystd * rng.normal(size=(n,))

    return W, T, Y


def generate_data(
    n_sample, d=4, link="sin", compute_IM_true=True, rng=None, idxY=None
):

    if rng is None:
        rng = np.random.default_rng()
    if idxY is None:
        idxY = sample_idx(rng, d)

    # Compute montecarlo estimate of groundtruth IM based 100000 samples
    if compute_IM_true:
        IM_true = np.mean(
            [
                compute_IM(rng, indexY=idxY, link=link, N=10000)
                for _ in range(100)
            ]
        )
    else:
        IM_true = np.nan
    W, T, Y = sample_SI(rng, indexY=idxY, n=n_sample, link=link, Ystd=1)
    return W, T, Y, IM_true


def estimate(
    W,
    T,
    Y,
    classifier="Logistic",
    regressor="OLS",
    crossfit=True,
    joint_regression=True,
):

    IM_con = Y[T == 1].mean()
    var_con = Y[T == 1].var()

    # Classical estimators based on linear methods
    if crossfit:
        x, y = es.IM_est_cf(
            T, W, Y, "Logistic", regressor, joint_regression=joint_regression
        )
    else:
        x, y = es.IM_est(
            T, W, Y, "Logistic", regressor, joint_regression=joint_regression
        )
    IM_ipw = x[0]
    var_ipw = y[0]
    IM_reg = x[1]
    var_reg = y[1]
    IM_aipw = x[2]
    var_aipw = y[2]
    IM_pru = x[3]
    var_pru = y[3]

    # Estimators based on SI network
    if crossfit:
        x, y = es.SI_IM_cf(
            T, W, Y, clf=classifier, joint_regression=joint_regression
        )
    else:
        x, y = es.SI_IM(
            T, W, Y, clf=classifier, joint_regression=joint_regression
        )

    IM_sir = x[0]
    var_sir = y[0]
    IM_siaipw = x[1]
    var_siaipw = y[1]
    IM_sioapw = x[2]
    var_sioapw = y[2]
    IM_siobpw = x[3]
    var_siobpw = y[3]

    res = {
        "Unadjusted": IM_con,
        "Regr": IM_reg,
        "IPW": IM_ipw,
        "AIPW": IM_aipw,
        "OBPW": IM_pru,
        "SI_R": IM_sir,
        "SI_AIPW": IM_siaipw,
        "SI_OAPW": IM_sioapw,
        "SI_OBPW": IM_siobpw,
        "var_Unadjusted": var_con,
        "var_Regr": var_reg,
        "var_IPW": var_ipw,
        "var_AIPW": var_aipw,
        "var_OBPW": var_pru,
        "var_SI_R": var_sir,
        "var_SI_AIPW": var_siaipw,
        "var_SI_OAPW": var_sioapw,
        "var_SI_OBPW": var_siobpw,
    }

    return res


# Run a single setting
def run_single_setting(
    n_sample=500,
    d=4,
    link="sin",
    classifier="Logistic",
    regressor="OLS",
    crossfit=True,
    joint_regression=True,
    rng=None,
    compute_IM_true=True,
    idxY=None,
):

    W, T, Y, IM_true = generate_data(
        n_sample=n_sample,
        d=d,
        link=link,
        compute_IM_true=compute_IM_true,
        rng=rng,
        idxY=idxY,
    )

    res = estimate(
        W,
        T,
        Y,
        classifier=classifier,
        regressor=regressor,
        crossfit=crossfit,
        joint_regression=joint_regression,
    )

    return {"IM": IM_true, **res}
