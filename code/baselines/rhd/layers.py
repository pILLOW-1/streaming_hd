"""Cross-backbone adaptation of the released RHD attention-layer convention."""
from math import floor

def official_families(native_depth, layer_types=None):
    if native_depth < 2:
        raise ValueError('RHD requires a decoder depth of at least two')
    if layer_types is None:
        full = list(range(1, native_depth + 1))
    else:
        if len(layer_types) != native_depth:
            raise ValueError('Layer type coverage mismatch')
        full = [i+1 for i,t in enumerate(layer_types) if t == 'full_attention']
    if not full:
        raise ValueError('No genuine global full-attention layer')
    result = []
    for name, reference, anchors, mature in [('official7B',32,[14,16,18,20,22,24,26],28),('official14B',48,[32,36,40,42,44,46],48)]:
        def mapped(value):
            return min(native_depth,max(1,floor(value*native_depth/reference+.5)))
        depths = sorted({mapped(x) for x in anchors})
        final = mapped(mature)
        if max(depths) >= final:
            raise ValueError('Mapped depth anchors reach the mature layer')
        requested = sorted({x-min(depths)+1 for x in depths+[final]})
        actual = sorted({max([x for x in full if x<=q],default=full[0]) for q in requested})
        result.append(dict(name=name+'_relative_native_depth',anchors=depths,mature=final,requested_attention_layers=requested,attention_layers=actual))
    return result
