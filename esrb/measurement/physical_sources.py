"""Preserve compiled asset identities while using a repaired code capsule."""
from pathlib import Path
import hashlib
import json


def verify(root,manifest):
    files=json.loads(Path(manifest).read_text());root=Path(root)
    return all((root/name).is_file() and hashlib.sha256((root/name).read_bytes()).hexdigest()==value
               for name,value in files.items())


def bind(new_root,original_root,manifest):
    """Only restore immutable asset path spelling before XML compilation.

    Simulation parameters and loaded asset bytes stay identical. The task keeps
    its new capsule paths for bookkeeping; MJB retains the original asset paths.
    No runtime state, model array, action or event label is modified.
    """
    import mujoco
    if not verify(original_root,manifest):raise ValueError('original physical source capsule changed')
    prefix=str(Path(new_root));original=str(Path(original_root))
    factory=mujoco.MjModel.from_xml_string
    def compile_xml(xml,*args,**kwargs):
        if isinstance(xml,bytes):xml=xml.replace(prefix.encode(),original.encode())
        else:xml=xml.replace(prefix,original)
        return factory(xml,*args,**kwargs)
    mujoco.MjModel.from_xml_string=staticmethod(compile_xml)
