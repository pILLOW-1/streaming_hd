"""Original event-recovery calculations on aligned original-step predictions."""
from collections import Counter, defaultdict
import math, statistics
VERSION = "expanded-prefix-recovery-v1"

def require(value, message):
    if not value:
        raise ValueError(message)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def validate_trajectory(record):
    """Validate a normalized trajectory; step positions cannot substitute for IDs."""
    ids, labels, probs = (record[k] for k in ("original_step_ids", "prefix_labels", "prefix_probabilities"))
    require(isinstance(record.get("trajectory_id"), str) and bool(record["trajectory_id"]), "Missing trajectory ID")
    require(isinstance(ids, list) and ids and all(type(x) is int for x in ids)
            and all(b > a for a, b in zip(ids, ids[1:])), "Original step IDs must be strictly increasing integers")
    require(isinstance(labels, list) and isinstance(probs, list)
            and len(labels) == len(probs) == len(ids), "Prediction, label and original-ID lengths differ")
    require(all(not isinstance(x, bool) and x in (0, 1) for x in labels), "Nonbinary prefix labels")
    require(all(not isinstance(x, bool) and isinstance(x, (int, float))
                and math.isfinite(x) and 0 <= x <= 1 for x in probs), "Invalid prefix probability")
    return record


def event_records(record):
    """Return adjacent recovery episodes and separately observed gapped boundaries.

    Delay is zero for recognition at the first gold-zero step. Observation ends
    at the next gold-one step, missing original step, or trajectory endpoint.
    Undetected events ending at the next gold error are episode failures;
    undetected events ending at a gap or endpoint are right-censored.
    """
    validate_trajectory(record)
    ids, labels, probs = (record[k] for k in ("original_step_ids", "prefix_labels", "prefix_probabilities"))
    hard = [p > .5 for p in probs]
    adjacent, gapped = [], []
    for i in range(1, len(ids)):
        if labels[i - 1] != 1 or labels[i] != 0:
            continue
        event = dict(trajectory_id=record["trajectory_id"], domain=record.get("domain", "unknown"),
                     previous_original_step_id=ids[i - 1], original_step_id=ids[i],
                     retained_position=i, previous_prediction=int(hard[i - 1]),
                     current_prediction=int(hard[i]), current_probability=probs[i])
        if ids[i] != ids[i - 1] + 1:
            event.update(unobserved_steps_between=ids[i] - ids[i - 1] - 1,
                         interpretation="observed 1-to-0 across a gap; exact transition time is unknown")
            gapped.append(event)
            continue
        end = i
        while end + 1 < len(ids) and ids[end + 1] == ids[end] + 1 and labels[end + 1] == 0:
            end += 1
        if end + 1 == len(ids):
            reason = "trajectory_end"
        elif ids[end + 1] != ids[end] + 1:
            reason = "original_step_gap"
        else:
            reason = "next_gold_error"
        found = next((j for j in range(i, end + 1) if not hard[j]), None)
        event.update(observed_zero_episode_steps=end - i + 1, observation_end=reason,
                     last_observed_original_step_id=ids[end],
                     detected=found is not None,
                     detection_original_step_id=ids[found] if found is not None else None,
                     detection_delay_steps=ids[found] - ids[i] if found is not None else None,
                     right_censored=found is None and reason != "next_gold_error",
                     episode_false_positive_steps=sum(hard[i:end + 1]))
        adjacent.append(event)
    return adjacent, gapped


def _boundary_summary(events):
    n = len(events)
    immediate = sum(e["current_prediction"] == 0 for e in events)
    armed = [e for e in events if e["previous_prediction"] == 1]
    switched = sum(e["current_prediction"] == 0 for e in armed)
    return dict(events=n, trajectories=len({e["trajectory_id"] for e in events}),
                predicted_recovered_at_boundary=immediate,
                boundary_recovery_accuracy=ratio(immediate, n),
                still_predicted_hallucinated_at_boundary=n - immediate,
                boundary_miss_rate=ratio(n - immediate, n),
                previously_predicted_hallucinated=len(armed),
                predicted_1_to_0_at_boundary=switched,
                predicted_1_to_0_rate_all_events=ratio(switched, n),
                predicted_1_to_0_rate_given_previous_prediction_1=ratio(switched, len(armed)),
                already_predicted_0_before_and_at_boundary=immediate - switched)


def _episode_summary(events):
    result = _boundary_summary(events)
    delays = [e["detection_delay_steps"] for e in events if e["detected"]]
    missed = [e for e in events if not e["detected"]]
    windows = sum(e["observed_zero_episode_steps"] for e in events)
    errors = sum(e["episode_false_positive_steps"] for e in events)
    armed = [e for e in events if e["previous_prediction"] == 1]
    armed_detected = sum(e["detected"] for e in armed)
    result.update(detected_within_observed_zero_episode=len(delays),
                  detected_within_observed_zero_episode_rate=ratio(len(delays), len(events)),
                  undetected_within_observed_zero_episode=len(missed),
                  undetected_before_next_gold_error=sum(e["observation_end"] == "next_gold_error" for e in missed),
                  right_censored_at_trajectory_end=sum(e["observation_end"] == "trajectory_end" for e in missed),
                  right_censored_at_original_step_gap=sum(e["observation_end"] == "original_step_gap" for e in missed),
                  detected_given_previous_prediction_1=armed_detected,
                  detected_rate_given_previous_prediction_1=ratio(armed_detected, len(armed)),
                  delay_steps_detected_events_only=dict(n=len(delays),
                      mean=statistics.mean(delays) if delays else None,
                      median=statistics.median(delays) if delays else None,
                      minimum=min(delays) if delays else None, maximum=max(delays) if delays else None,
                      histogram={str(k): v for k, v in sorted(Counter(delays).items())}),
                  observation_end_counts=dict(sorted(Counter(e["observation_end"] for e in events).items())),
                  post_recovery_observed_prefixes=windows,
                  post_recovery_false_positive_prefixes=errors,
                  post_recovery_prefix_false_positive_rate=ratio(errors, windows))
    # A next gold error ends the episode with known failure if still undetected.
    # Only an undetected gap/endpoint before k leaves the outcome unknown.
    result["detection_within_k_steps"] = {}
    for k in (0, 1, 2, 5, 10):
        known = [e for e in events if (e["detected"] and e["detection_delay_steps"] <= k)
                 or e["observed_zero_episode_steps"] > k or e["observation_end"] == "next_gold_error"]
        detected = sum(e["detected"] and e["detection_delay_steps"] <= k for e in known)
        result["detection_within_k_steps"][str(k)] = dict(detected=detected, outcome_observed=len(known),
            outcome_not_observed=len(events) - len(known), rate=ratio(detected, len(known)))
    return result


def summarize(records, include_events=False):
    """Aggregate normalized trajectories. Never concatenate across boundaries."""
    require(isinstance(records, list) and records, "No predictions to summarize")
    require(len({r["trajectory_id"] for r in records}) == len(records), "Duplicate trajectory ID")
    events, gaps, confusion, transitions = [], [], Counter(), Counter()
    for record in records:
        adjacent, gapped = event_records(record)
        events.extend(adjacent); gaps.extend(gapped)
        ids, labels, probs = (record[k] for k in ("original_step_ids", "prefix_labels", "prefix_probabilities"))
        hard = [int(p > .5) for p in probs]
        for label, prediction in zip(labels, hard):
            confusion[(int(label), prediction)] += 1
        for i in range(1, len(ids)):
            if ids[i] == ids[i - 1] + 1 and hard[i - 1] == 1 and hard[i] == 0:
                transitions["all"] += 1
                transitions["current_gold_1" if labels[i] == 1 else "current_gold_0"] += 1
                if labels[i - 1] == 1 and labels[i] == 0:
                    transitions["matches_gold_recovery_boundary"] += 1
    tp, fp, fn, tn = (confusion[key] for key in ((1, 1), (0, 1), (1, 0), (0, 0)))
    report = dict(trajectories=len(records), prefixes=tp + fp + fn + tn,
                  adjacent_recoveries=_episode_summary(events),
                  gapped_observed_1_to_0=_boundary_summary(gaps),
                  all_prefix_confusion=dict(tp=tp, fp=fp, fn=fn, tn=tn,
                      hallucination_recall=ratio(tp, tp + fn),
                      nonhallucinated_specificity=ratio(tn, tn + fp),
                      accuracy=ratio(tp + tn, tp + fp + fn + tn)),
                  adjacent_predicted_1_to_0=dict(count=transitions["all"],
                      current_gold_1=transitions["current_gold_1"],
                      current_gold_0=transitions["current_gold_0"],
                      matches_gold_recovery_boundary=transitions["matches_gold_recovery_boundary"],
                      false_recovery_fraction=ratio(transitions["current_gold_1"], transitions["all"])))
    if include_events:
        report["events"] = events
        report["gapped_events"] = gaps
    return report
