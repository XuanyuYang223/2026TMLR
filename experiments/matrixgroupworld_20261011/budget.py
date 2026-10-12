"""One serial GPU-stage span and bounded surrounding external CPU wall time."""
from __future__ import annotations
import time
from .common import DIAG,ROOT,SOURCE,read,json_write

def apparent_bytes():
    seen=set();total=0
    for base in (DIAG,ROOT,SOURCE):
        for path in base.rglob('*'):
            if path.is_file():
                resolved=path.resolve()
                if resolved not in seen:total+=path.stat().st_size;seen.add(resolved)
    return total
def cpu_elapsed():
    start=read(DIAG/'task_start.json');now=time.monotonic()
    if not (DIAG/'gpu_budget.json').exists():return now-start['monotonic']
    g=read(DIAG/'gpu_budget.json')
    outside=g['start_monotonic']-start['monotonic']
    return outside+max(0.,now-g['end_monotonic']) if 'end_monotonic' in g else outside
def cpu_deadline():
    start=read(DIAG/'task_start.json')
    remaining=max(0.,start['cpu_external_wall_limit_seconds']-cpu_elapsed())
    return time.monotonic()+remaining
def guard_cpu(reserve=120.):
    assert cpu_elapsed()+reserve<read(DIAG/'task_start.json')['cpu_external_wall_limit_seconds'], 'CPU external wall budget exhausted; stop and report partial scope'
    assert apparent_bytes()<read(DIAG/'task_start.json')['new_apparent_size_limit_bytes'], 'New apparent storage budget exhausted'
def start_gpu():
    assert not (DIAG/'gpu_budget.json').exists(),'GPU budget origin is saved once, never reset on resume'
    guard_cpu();now=time.monotonic();s=read(DIAG/'task_start.json')
    value=dict(status='running',start_monotonic=now,deadline_monotonic=now+s['gpu_wall_limit_seconds'],
        limit_seconds=s['gpu_wall_limit_seconds'],serial_no_parallel=True,
        scope='From true initialization/engineering setup through last requested GPU extraction, including validation, saving and stage gaps; not active GPU kernel time.',
        cpu_before_gpu_seconds=now-s['monotonic'])
    json_write(DIAG/'gpu_budget.json',value);return value
def finish_gpu(reason):
    g=read(DIAG/'gpu_budget.json');assert 'end_monotonic' not in g
    g.update(status='stopped',end_monotonic=time.monotonic(),reason=reason)
    g['stage_wall_seconds']=g['end_monotonic']-g['start_monotonic']
    json_write(DIAG/'gpu_budget.json',g);return g
def snapshot():
    s=read(DIAG/'task_start.json');g=read(DIAG/'gpu_budget.json') if (DIAG/'gpu_budget.json').exists() else None
    now=time.monotonic()
    value=dict(cpu_external_wall_seconds=cpu_elapsed(),cpu_limit_seconds=s['cpu_external_wall_limit_seconds'],
        GPU_stage_wall_seconds=(g.get('end_monotonic',now)-g['start_monotonic']) if g else 0.,
        GPU_stage_limit_seconds=s['gpu_wall_limit_seconds'],new_apparent_bytes=apparent_bytes(),
        new_apparent_limit_bytes=s['new_apparent_size_limit_bytes'],
        gpu_kernel_active_time_unknown=True,parallel_training=False,
        accounting_limitation='Task budget clock was saved at creation of the isolated directories, after initial read-only request/source discovery; preceding short discovery is not represented as GPU time.')
    json_write(DIAG/'cost_snapshot.json',value);return value
