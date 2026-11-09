"""Preserve the official layer/head mean without a full duplicate attention stack."""
import ast
import inspect
import textwrap
import types
import torch


def mean_attention_layers(attentions, chunk=64):
    if not attentions or chunk<1:raise ValueError('Native attention tensors required')
    first=attentions[0]
    assert first.ndim==4 and first.shape[-1]==first.shape[-2]
    assert all(a.shape==first.shape and a.dtype==first.dtype and a.device==first.device for a in attentions)
    b,h,n,_=first.shape
    result=torch.empty((b,n,n),dtype=first.dtype,device=first.device)
    for start in range(0,n,chunk):
        end=min(n,start+chunk)
        block=torch.stack([a[:,:,start:end,:] for a in attentions],dim=0)
        reduced=torch.mean(torch.mean(block,dim=0),dim=1)
        result[:,start:end,:]=reduced
        del block,reduced
    # Same already-computed native tensors, with a one-query dense reference.
    # This gate adds no model forward and does not reselect any detector.
    for index in sorted({0,n-1}):
        reference=torch.mean(torch.mean(torch.stack([a[:,:,index:index+1,:] for a in attentions],dim=0),dim=0),dim=1)
        torch.testing.assert_close(result[:,index:index+1,:],reference,rtol=1e-6,atol=1e-7)
    return result


def install_chunked_attention_mean(scorer):
    original=type(scorer).calculate_uncertainty
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    changed={'stack':0,'mean':0}
    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                name=node.targets[0].id
                if name=='all_layers_attention':
                    assert ast.unparse(node.value)=='torch.stack(attentions, dim=0)'
                    node.value=ast.Constant(None);changed['stack']+=1
                elif name=='avg_attention':
                    assert ast.unparse(node.value)=='torch.mean(torch.mean(all_layers_attention, dim=0), dim=1)'
                    node.value=ast.Call(func=ast.Name(id='_race_mean_attention_layers',ctx=ast.Load()),args=[ast.Name(id='attentions',ctx=ast.Load())],keywords=[]);changed['mean']+=1
            return node
    rewritten=Rewrite().visit(tree);ast.fix_missing_locations(rewritten)
    assert changed=={'stack':1,'mean':1},changed
    namespace=dict(original.__globals__);namespace['_race_mean_attention_layers']=mean_attention_layers
    exec(compile(rewritten,'<RACE_exact_component_chunked_attention_mean>','exec'),namespace)
    scorer.calculate_uncertainty=types.MethodType(namespace['calculate_uncertainty'],scorer)
    return changed
