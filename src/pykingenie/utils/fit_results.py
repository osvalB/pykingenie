"""Data identity and statistics for saved kinetic fitting results."""

import hashlib

import numpy as np


def _hash_array(digest, value):
    array = np.ascontiguousarray(value, dtype='<f8')
    digest.update(repr(array.shape).encode('ascii'))
    digest.update(array.tobytes())


def fitting_data_fingerprint(fitter):
    """Identify numeric inputs, including trace order, Smax grouping and trimming.

    Hash values rather than Python container types, so converting a list to an
    equivalent array does not invalidate a fit. Display names are excluded.
    """
    digest = hashlib.sha256()
    for name in (
        'assoc_lst', 'disso_lst', 'time_assoc_lst', 'time_disso_lst',
        'lig_conc_lst', 'lig_conc', 'prot_conc', 'smax_id', 'is_single_cycle', 'trim_time',
    ):
        digest.update(name.encode('ascii'))
        value = getattr(fitter, name, None)
        if value is None:
            digest.update(b'None')
        elif name in ('assoc_lst', 'disso_lst', 'time_assoc_lst', 'time_disso_lst'):
            digest.update(str(len(value)).encode('ascii'))
            for trace in value:
                _hash_array(digest, trace)
        else:
            _hash_array(digest, value)
    return digest.hexdigest()


def saved_fit_residuals(fitter, state):
    """Return RSS, observation count and identity for exactly the fitted points.

    Missing or non-finite predictions are rejected instead of silently scoring
    a different set of observations. Dissociation-only time limits are applied
    to the current, validated raw inputs.
    """
    digest = hashlib.sha256()
    rss = 0.0
    n = 0
    for phase, prediction_name, signals_name, times_name in (
        ('steady_state', 'signal_ss_fit', 'signal_ss', 'lig_conc_lst_per_id'),
        ('association', 'signal_assoc_fit', 'assoc_lst', 'time_assoc_lst'),
        ('dissociation', 'signal_disso_fit', 'disso_lst', 'time_disso_lst'),
    ):
        predictions = state.get(prediction_name)
        if predictions is None:
            continue
        if phase == 'steady_state':
            signals, times = state.get(signals_name), state.get(times_name)
        else:
            signals = getattr(fitter, signals_name, None)
            times = getattr(fitter, times_name, None)
        if signals is None or times is None or not (len(predictions) == len(signals) == len(times)):
            raise ValueError(f'Saved {phase} predictions do not match the observed traces.')

        digest.update(phase.encode('ascii'))
        digest.update(str(len(signals)).encode('ascii'))
        for observed, time, predicted in zip(signals, times, predictions):
            if predicted is None:
                raise ValueError(f'Saved {phase} fit has missing predictions.')
            observed = np.asarray(observed, dtype=float)
            time = np.asarray(time, dtype=float)
            predicted = np.asarray(predicted, dtype=float)
            if phase == 'dissociation':
                time_limit = state['fit_settings'].get('time_limit', 0)
                if time_limit > 0:
                    mask = time < np.min(time) + time_limit
                    observed, time = observed[mask], time[mask]
            if observed.shape != predicted.shape or time.shape != observed.shape:
                raise ValueError(f'Saved {phase} predictions do not match the fitted time range.')
            if phase != 'steady_state':
                trim_time = getattr(fitter, 'trim_time', 0)
                keep = time - time[0] >= trim_time
                observed, predicted, time = observed[keep], predicted[keep], time[keep]
            if not (np.isfinite(observed).all() and np.isfinite(predicted).all() and
                    np.isfinite(time).all()):
                raise ValueError(f'Saved {phase} fit contains non-finite observations or predictions.')
            _hash_array(digest, time)
            _hash_array(digest, observed)
            residuals = observed - predicted
            rss += float(np.sum(residuals ** 2))
            n += observed.size

    if not n:
        raise ValueError('Saved fit has no fitted observations to compare.')
    return rss, n, digest.hexdigest()


def saved_fit_statistics(rss, n, k):
    """Unweighted reduced chi squared and Bayesian information criterion.

    BIC = n*log(RSS/n) + k*log(n). A perfect fit has BIC = -infinity.
    Positive residual degrees of freedom are required for comparison.
    """
    if n <= k:
        raise ValueError('Fit comparison requires more observations than fitted parameters (n > k).')
    if not np.isfinite(rss) or rss < 0:
        raise ValueError('Fit comparison requires a finite, non-negative residual sum of squares.')
    redchi = rss / (n - k)
    bic = -np.inf if rss == 0 else n * np.log(rss / n) + k * np.log(n)
    return redchi, bic
