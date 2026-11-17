"""Source-position-only threshold calibration; no downstream task inputs."""
import numpy as np


def select_threshold(scores, targets):
    """Maximize micro-F1 on the source tuning split, breaking ties upwards.

    Decisions are score >= threshold. Candidates are the distinct source
    tuning scores, including the no-positive prediction. No test scores,
    test counts, permutation, or composed answers are accepted.
    """
    scores=np.asarray(scores,dtype=np.float64);targets=np.asarray(targets)
    if scores.shape!=targets.shape or targets.dtype!=np.bool_ or not scores.size or not np.isfinite(scores).all():
        raise ValueError('Finite source scores and same-shape Boolean labels required')
    s=scores.ravel();y=targets.ravel();positives=int(y.sum())
    if not positives:
        return float('inf'),dict(source_micro_f1=0.,source_positive_positions=0,source_total_positions=len(y),source_predicted_positions=0,threshold='Infinity',reason='No positive source tuning positions; fixed empty mask')
    order=np.argsort(-s,kind='stable');s=s[order];truth=y[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(truth,dtype=np.int64)[ends];predicted=ends+1
    f1=2*tp/(predicted+positives)
    # Scores are descending, so the first maximum chooses the higher threshold.
    best=int(np.flatnonzero(f1==f1.max())[0]);threshold=float(s[ends[best]])
    return threshold,dict(source_micro_f1=float(f1[best]),source_positive_positions=positives,source_total_positions=len(y),source_predicted_positions=int(predicted[best]),threshold=threshold,reason='Source tuning micro-F1; equal-score groups kept together; highest-threshold tie')


def micro_f1(predicted, truth):
    p=np.asarray(predicted,dtype=bool);y=np.asarray(truth,dtype=bool)
    if p.shape!=y.shape:raise ValueError('Mask shapes differ')
    den=int(p.sum()+y.sum())
    return float(2*np.count_nonzero(p&y)/den) if den else 1.
