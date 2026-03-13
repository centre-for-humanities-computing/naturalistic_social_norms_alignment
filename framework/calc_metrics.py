import pandas as pd
import numpy as np


def bootstrap(data, statistic=np.mean, n_resamples=10_000, ci=0.95):
    """
    Bootstrap a statistic over 1D data.
    Returns the observed statistic, confidence interval, and all bootstrap samples.
    """
    data = np.asarray(data)

    observed = statistic(data)

    samples = np.array([
        statistic(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_resamples)
    ])
    alpha = (1 - ci) / 2
    ci_low, ci_high = np.quantile(samples, [alpha, 1 - alpha])

    return {
        "observed": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def id_to_stance(idx):
    if "not_advised" in idx:
        return "not_advised"
    return "advised"


def ids_to_stance(ids):
    return list(set([id_to_stance(x) for x in ids]))


def calculate_metrics_sample(dilemma_df, num_ref_solutions):
    dilemma_df = dilemma_df.copy()

    # Do we need this step ?
    dilemma_df['nu_stances'] = dilemma_df['match_stances'].apply(len)
    dilemma_df = dilemma_df[dilemma_df['nu_stances'] <= 1]

    if len(dilemma_df) == 0:
        return {
            "EAA": 0.0,
            "SAA": 0.0,
            "AVG": 0.0
        }

    dilemma_df['is_equiv'] = dilemma_df['matched_targets'].apply(
        lambda x: len(x) > 0
    )
    try:
        dilemma_df['are_stances_match'] = dilemma_df.apply(
            lambda x: x['pred_stance'] in x['match_stances'], axis=1
        )
    except:
        print()

    M = len(
        dilemma_df[(dilemma_df['is_equiv']) & (dilemma_df['are_stances_match'])]
    )
    T = len(
        dilemma_df[(dilemma_df['is_equiv']) & (dilemma_df['are_stances_match'] == False)]
    )

    rcand = dilemma_df['pred_id'].nunique()

    eaa = 1 - (T / (T + M)) if T + M > 0 else 0.0
    saa = M / (num_ref_solutions + rcand - M) if num_ref_solutions + rcand - M > 0 else 0.0
    metrics = {
        "EAA": eaa,
        "SAA": saa,
        "AVG": (eaa + saa) / 2
    }

    return metrics


def get_metrics_bootstrap(meta_df, preds_df):
    final_per_model_metrics = {}

    for modelname, model_df in preds_df.groupby('model'):
        model_metrics = []
        final_per_model_metrics[modelname] = {}
        for dilemma_id, dilemma_df in model_df.groupby('dilemma_id'):
            num_ref_solutions = meta_df[meta_df['dilemma_id'] == dilemma_id].iloc[0][
                'target_matched_postprocessed_solutions_first_person']
            num_ref_solutions = len(num_ref_solutions)
            model_metrics.append(
                calculate_metrics_sample(dilemma_df=dilemma_df, num_ref_solutions=num_ref_solutions)
            )

        model_metrics = pd.DataFrame(model_metrics)
        for metric_name in model_metrics.columns:
            bootstrapped_metrics = bootstrap(model_metrics[metric_name].values)
            final_per_model_metrics[modelname][metric_name] = bootstrapped_metrics
    return final_per_model_metrics


if __name__ == "__main__":
    import json

    meta_df = pd.read_json("preds_df.json")
    preds_df = pd.read_json("match_df.json")

    preds_df['pred_stance'] = preds_df['pred_id'].apply(id_to_stance)
    preds_df['match_stances'] = preds_df['matched_targets'].apply(ids_to_stance)

    metrics = get_metrics_bootstrap(meta_df, preds_df)
    with open("SoM-FinalData/metrics.json", 'w') as f:
        json.dump(metrics, f)
