"""Canonical experiment names; no historical aliases."""
from importlib import import_module

SCENES = {'stack':'stack','pickplacecan':'pickup_can','two-arm-handover':'handover','nut-assembly':'nut','wipe':'wipe'}

def get_scene(name):
    if name not in SCENES:
        raise ValueError(f'unknown scene: {name}')
    return import_module(f'{__name__}.{SCENES[name]}').SCENE
