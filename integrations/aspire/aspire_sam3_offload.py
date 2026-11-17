"""Stage backbone Linear/Conv FP32 weights on CUDA only during their forward.

Inference only, one request at a time. No weights are quantized, and CUDA
autocast remains BF16 as in the upstream service. Disable its weight cache in
the backbone so staging does not retain an entire second GPU weight set.
"""
import functools
import torch


class BackboneOffload:
    def __init__(self,model,backbone,device):
        self.device=torch.device(device);self.handles=[];self.masters={}
        classes=(torch.nn.Linear,torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d)
        # MultiheadAttention reads out_proj.weight functionally; its child
        # Linear.forward hooks are never invoked. Keep those weights on GPU.
        functional_children={id(child) for parent in backbone.modules()
                             if isinstance(parent,torch.nn.MultiheadAttention)
                             for child in parent.modules()}
        candidates={id(m):m for m in backbone.modules()
                    if isinstance(m,classes) and id(m) not in functional_children}
        owners={}
        for m in model.modules():
            for p in m.parameters(recurse=False):owners.setdefault(id(p),set()).add(id(m))
        selected={id(p) for m in candidates.values() for p in m.parameters(recurse=False)
                  if owners[id(p)]<=set(candidates)}
        moved=set()
        for m in model.modules():
            for p in m.parameters(recurse=False):
                if id(p) in moved:continue
                moved.add(id(p))
                p.data=p.data.to('cpu' if id(p) in selected else self.device)
            for name,b in m.named_buffers(recurse=False):
                m._buffers[name]=b.to(self.device)
        for m in candidates.values():
            master={name:p.detach() for name,p in m.named_parameters(recurse=False) if id(p) in selected}
            if not master:continue
            self.masters[m]=master
            self.handles.append(m.register_forward_pre_hook(self._before))
            self.handles.append(m.register_forward_hook(self._after,always_call=True))
        self.parameter_elements=sum(p.numel() for values in self.masters.values() for p in values.values())
        def bounded(original):
            @functools.wraps(original)
            def forward(*args,**kwargs):
                with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                    return original(*args,**kwargs)
            return forward
        # SAM3 invokes these methods directly rather than always calling the
        # backbone as an nn.Module. Cover every inference entry point.
        for name in ('forward','forward_image','forward_text'):
            original=getattr(backbone,name,None)
            if callable(original):setattr(backbone,name,bounded(original))

    def _before(self,module,args):
        for name,master in self.masters[module].items():
            module._parameters[name].data=master.to(self.device)

    def _after(self,module,args,result):
        for name,master in self.masters[module].items():
            module._parameters[name].data=master


def stage_backbone(model,device='cuda:0'):
    return BackboneOffload(model,model.backbone,device)
