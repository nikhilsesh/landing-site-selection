import numpy as np

def precision_weighted_average(S_off, S_on, sigma_off, sigma_on):
    """
    Precision-weighted average of offline and online safety maps.

    S_fuse(p) = (S_off(p)/σ²_off(p) + S_on(p)/σ²_on(p))
                / (1/σ²_off(p) + 1/σ²_on(p))

    Map with higher certainty contributes more to the result.
    """
    prec_off = 1.0 / np.square(sigma_off)
    prec_on  = 1.0 / np.square(sigma_on)
    return (S_off * prec_off + S_on * prec_on) / (prec_off + prec_on)
