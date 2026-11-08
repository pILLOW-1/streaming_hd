"""User-approved RHD-only segmentation; frozen tokens and labels stay intact."""
from __future__ import annotations


def split_single_step_spans(spans: list[list[int]]) -> tuple[list[list[int]], bool]:
    """Split a sole native token span into two nonempty contiguous halves.

    A token midpoint is reproducible without rewriting or regenerating text.
    Multi-Step trajectories retain their exact original spans. A one-token
    trajectory cannot be split this way and is surfaced explicitly.
    """
    result=[list(span) for span in spans]
    if not result or any(len(s)!=2 or not 0<=s[0]<s[1] for s in result):
        raise ValueError('Nonempty valid native Step token spans required')
    if len(result)!=1:return result,False
    start,end=result[0]
    if end-start<2:raise ValueError('RHD_SINGLE_TOKEN_STEP_CANNOT_BE_SPLIT_WITHOUT_RETOKENIZATION')
    middle=start+(end-start)//2
    return [[start,middle],[middle,end]],True
