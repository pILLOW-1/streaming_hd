"""User-approved isolation of genuinely invalid ARS Yes/No auxiliary labels.

Every original reference stays available. Only invalid generated variants are
excluded from the projection's auxiliary agreement data. Computation failures
remain blocked for their separately authorized recovery; never coerce a label.
"""


def select_auxiliary(records):
    positions, original_ids, agreement, isolated, blocked = [], [], [], [], []
    seen = set()
    for original, record in enumerate(records):
        sample_id = record['sample_id']
        if sample_id in seen:
            raise ValueError('Duplicate ARS original trajectory')
        seen.add(sample_id)
        values = record['agreement']
        if len(values) != 5 or values[0] is not True:
            raise ValueError('Original reference and four ARS variants are required')
        missing = [i for i, value in enumerate(values) if value is None]
        if any(value is not None and type(value) is not bool for value in values):
            raise ValueError('Never coerce an ARS agreement label')
        errors = record.get('invalid_outputs', [])
        if len(missing) != len(errors):
            raise ValueError('Failed ARS output/variant mapping mismatch')
        if record['status'] == 'PASS' and missing:
            raise ValueError('Invalid PASS judgment')
        if record['status'] not in ('PASS', 'INVALID_OFFICIAL_JUDGE_OUTPUT'):
            raise ValueError('Unknown ARS judgment state')
        failures = dict(zip(missing, errors))
        for index, value in enumerate(values):
            if type(value) is bool:
                positions.append(5*original+index)
                original_ids.append(original)
                agreement.append(value)
                continue
            error = failures[index]
            if not isinstance(error, (list, tuple)) or len(error) != 2 or error[0] is not None or not isinstance(error[1], str):
                raise ValueError('Unknown official failed-output schema')
            item = dict(sample_id=sample_id, variant_index=index-1, source_position=5*original+index, raw_failed_output=error)
            (blocked if error[1].startswith('[ERROR]') else isolated).append(item)
    assert all(5*i in positions for i in range(len(records)))
    return dict(source_positions=positions, original_ids=original_ids, agreement=agreement,
                isolated_format_variants=isolated, unresolved_computation_variants=blocked,
                supervised_original_count=len(records), expected_generated_variants=4*len(records),
                retained_generated_variants=len(positions)-len(records))
