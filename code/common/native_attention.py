"""Strict native attention mapping for dense and hybrid transformer outputs."""


def attention_blocks(config, attentions):
    text=getattr(config,'text_config',config)
    count=text.num_hidden_layers
    types=getattr(text,'layer_types',None)
    eligible=[i for i in range(count) if types is None or types[i] not in ('linear_attention','linear')]
    if attentions is None:raise ValueError('Backbone did not return native attention weights')
    if len(attentions)==count:
        mapped={i:a[0] for i,a in enumerate(attentions) if a is not None and i in eligible}
    elif len(attentions)==len(eligible):
        mapped={i:a[0] for i,a in zip(eligible,attentions) if a is not None}
    else:raise ValueError('Native attention tuple cannot be mapped to decoder indices')
    if sorted(mapped)!=eligible:
        raise ValueError('Native attention output is incomplete; no synthetic attention fallback')
    for value in mapped.values():
        if value.ndim!=3 or value.shape[-1]!=value.shape[-2]:
            raise ValueError('Expected native full-sequence [heads,tokens,tokens] attention')
    return mapped
