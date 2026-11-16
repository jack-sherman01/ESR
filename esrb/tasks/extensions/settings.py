"""Bounded physical difficulty presets, shared by execution and offline audit."""
from copy import deepcopy
from esrb.tasks.reset_domain import validate_disturbance


def configured_task(task_type, settings=None):
    if not settings:
        return task_type
    settings=deepcopy(settings)
    if set(settings)-{'difficulty','reset_reference','reset_bounds','max_steps'}:
        raise ValueError('unknown task setting')
    # Canonical defaults match YAML/Pydantic and direct programmatic callers.
    settings={'difficulty':'hard','reset_reference':{},'reset_bounds':{},**settings}
    difficulty=settings.get('difficulty','hard')
    if difficulty not in {'easy','medium','hard'}:
        raise ValueError('difficulty must be easy, medium or hard')
    reference={**task_type.reset_reference,**settings.get('reset_reference',{})}
    validate_disturbance(task_type.reset_bounds,reference)
    scale={'easy':.25,'medium':.6,'hard':1.}[difficulty]
    bounds={key:(center+(lo-center)*scale,center+(hi-center)*scale)
            for key,(lo,hi) in task_type.reset_bounds.items() for center in (reference[key],)}
    if settings.get('reset_bounds',{}).keys()-bounds.keys():
        raise ValueError('unknown reset coordinate in task settings')
    for key,pair in settings.get('reset_bounds',{}).items():
        if len(pair)!=2 or not task_type.reset_bounds[key][0]<=pair[0]<=reference[key]<=pair[1]<=task_type.reset_bounds[key][1]:
            raise ValueError(f'{key}: bounds must contain reference and remain inside supported domain')
        bounds[key]=tuple(pair)
    maximum=settings.get('max_steps',task_type.MAX_STEPS)
    if type(maximum) is not int or not 1<=maximum<=task_type.MAX_STEPS:
        raise ValueError('max_steps may only reduce the bounded episode limit')
    class ConfiguredTask(task_type):
        def reset(self,*args,**kwargs):
            obs=super().reset(*args,**kwargs)
            self._log['task_settings']=deepcopy(settings)
            return obs
    ConfiguredTask.engineering_settings=settings
    ConfiguredTask.reset_reference=reference
    ConfiguredTask.reset_bounds=bounds
    ConfiguredTask.MAX_STEPS=maximum
    return ConfiguredTask
