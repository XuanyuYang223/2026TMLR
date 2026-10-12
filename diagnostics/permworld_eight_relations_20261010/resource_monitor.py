"""Read-only 45-second process/GPU snapshots for the frozen serial queue.

This script never imports training/model modules or launches, stops, or signals
training jobs.  Its writes are confined to new monitoring artifacts in DIAG.
CPU increments cover only observed adjacent snapshots of the same process.
They omit time before the first observation and an exiting process's final tail.
"""
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

DIAG = Path(__file__).resolve().parent
CONFIG = DIAG / 'protocol.json'
ROOT = Path('/mnt/yangx-projects-f/20261010/permworld_eight_relations')
INTERVAL_SECONDS = 45.0
HZ = os.sysconf('SC_CLK_TCK')
QUEUE_MODULE = 'experiments.permworld_eight_relations_20261010.queue'


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def proc_snapshot(pid):
    try:
        p = Path('/proc') / str(pid)
        raw = (p / 'stat').read_text()
        stat = raw[raw.rfind(')') + 2:].split()
        # Tokens start at stat field 3 (state); user/system are fields 14/15.
        cmd = (p / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        return dict(pid=int(pid), parent_pid=int(stat[1]), state=stat[0],
                    start_ticks=int(stat[19]), user_seconds=int(stat[11]) / HZ,
                    system_seconds=int(stat[12]) / HZ,
                    waited_child_user_seconds=int(stat[13]) / HZ,
                    waited_child_system_seconds=int(stat[14]) / HZ,
                    rss_bytes=int(stat[21]) * os.sysconf('SC_PAGE_SIZE'),
                    command=' '.join(x for x in cmd if x), argv=cmd)
    except (OSError, ValueError, IndexError):
        return None


def queue_processes():
    all_procs = {}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            q = proc_snapshot(int(p.name))
            if q is not None:
                all_procs[q['pid']] = q
    queues = [p for p in all_procs.values() if QUEUE_MODULE in p['argv']]
    selected = {p['pid'] for p in queues}
    while True:
        old = len(selected)
        selected.update(p['pid'] for p in all_procs.values() if p['parent_pid'] in selected)
        if len(selected) == old:
            break
    result = []
    for pid in sorted(selected):
        q = dict(all_procs[pid]); q.pop('argv')
        q['role'] = 'queue' if any(pid == p['pid'] for p in queues) else 'queue_descendant'
        result.append(q)
    return result


def gpu_snapshot():
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10,
                           check=True)
        out = []
        for line in p.stdout.splitlines():
            v = [x.strip() for x in line.split(',')]
            def numeric(x):
                try: return float(x)
                except ValueError: return None
            out.append(dict(gpu_index=int(v[0]), utilization_percent=numeric(v[1]),
                            memory_used_mib=numeric(v[2]), memory_total_mib=numeric(v[3]),
                            power_watts=numeric(v[4])))
        return out, None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as e:
        return [], str(e)


def main():
    if (DIAG / 'resource_monitor.csv').exists():
        raise RuntimeError('Existing monitoring CSV must not be overwritten or silently restarted.')
    config = read_json(CONFIG)
    protocol_sha = digest(CONFIG)
    training_freeze = read_json(DIAG / 'training_freeze.json')
    writer_freeze = read_json(DIAG / 'metadata_writer_freeze.json')
    fixed_sources = {**training_freeze['source_sha256'], **writer_freeze['source_sha256']}
    expected_version = writer_freeze['frozen_training_source_version_sha256']
    jobs = {j['run_id']: j for j in config['training_manifest']}
    assert len(jobs) == 48
    assert training_freeze['protocol_sha256'] == writer_freeze['protocol_sha256'] == protocol_sha
    started = utc(); started_mono = time.monotonic()
    previous = {}; totals = {}; verified = {}; completed_stats = {}; initial_completed = None
    observed_exited_processes = []; events = []; reported = set(); gpu_history = []; samples = 0
    metadata = dict(started_utc=started, sample_interval_seconds=INTERVAL_SECONDS,
                    queue_state_path=str(ROOT / 'queue_state.json'), protocol_sha256=protocol_sha,
                    monitor_source_sha256=digest(__file__),
                    monitoring_scope='Read-only /proc and nvidia-smi; no imports of training/model modules, no GPU launches or process signals.',
                    cpu_scope='Cumulative process user/system snapshots and observed adjacent-sample deltas only. Not complete CPU core-hours. Time before each process first observation and its final tail after last observation are unobserved; first source and first failed metadata writer predate monitoring.',
                    gpu_scope='Instantaneous 45-second GPU utilization, memory and power samples. They do not measure exact GPU active time or energy; shared-device samples are not attribution to this queue alone.')
    write_json(DIAG / 'resource_monitor_metadata.json', metadata)
    fields = ['sample', 'utc', 'elapsed_monitor_seconds', 'queue_status', 'completed_models',
              'current_run', 'queue_pids', 'queue_user_seconds', 'queue_system_seconds',
              'descendant_user_seconds', 'descendant_system_seconds', 'process_snapshot_json',
              'gpu_snapshot_json', 'gpu_query_error', 'observed_user_delta_seconds',
              'observed_system_delta_seconds', 'anomalies_json']
    with (DIAG / 'resource_monitor.csv').open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields); writer.writeheader(); csv_file.flush()
        next_sample = time.monotonic()
        while True:
            time.sleep(max(0.0, next_sample - time.monotonic()))
            sample_time = time.monotonic(); next_sample = sample_time + INTERVAL_SECONDS
            stamp = utc(); samples += 1; anomalies = []
            try:
                state = read_json(ROOT / 'queue_state.json')
            except (OSError, ValueError) as e:
                state = dict(status='unreadable'); anomalies.append('queue_state:' + str(e))
            if state.get('protocol_sha256') != protocol_sha:
                anomalies.append('queue protocol identity mismatch')
            if state.get('queue_source_sha256') != fixed_sources[str(DIAG.parents[1] / 'experiments/permworld_eight_relations_20261010/queue.py')]:
                anomalies.append('queue source identity mismatch')
            if state.get('status') == 'failed':
                anomalies.append('queue failed: ' + str(state.get('error')))
            for path, expected in fixed_sources.items():
                try:
                    if digest(path) != expected: anomalies.append('frozen source hash mismatch:' + path)
                except OSError as e:
                    anomalies.append('frozen source unreadable:' + path + ':' + str(e))
            if digest(CONFIG) != protocol_sha:
                anomalies.append('protocol bytes changed')
            for path in sorted((ROOT / 'models').glob('*/completed.json')):
                run = path.parent.name
                try:
                    rec = read_json(path)
                    if run not in jobs: raise ValueError('unauthorized completed model')
                    cp_path = Path(rec['checkpoint_path'])
                    stat = cp_path.stat(); identity = (stat.st_size, stat.st_mtime_ns)
                    if run in verified:
                        if identity != completed_stats[run]:
                            raise ValueError('completed checkpoint bytes/stat changed')
                        continue
                    arch = jobs[run]['architecture']; init = config['initializations'][arch]
                    for key, val in dict(run_id=run, architecture=arch, encoding=jobs[run]['encoding'],
                        task=jobs[run]['task'], step=20000, finite_gradient_updates=20000,
                        protocol_sha256=protocol_sha, execution_source_version_sha256=expected_version,
                        data_sha256=config['data_sha256'], batch_schedule_sha256=config['batch_schedule_sha256'],
                        initial_checkpoint_sha256=init['sha256'], initial_parameter_sha256=init['parameter_sha256'],
                        starting_rng_hash=config['new_task_start_rng'][arch]['rng_hash'],
                        actual_initial_identity_verified=True, exposures=1280000,
                        metadata_writer_sha256=writer_freeze['source_sha256'][str(DIAG.parents[1] / 'experiments/permworld_eight_relations_20261010/runner.py')]).items():
                        if rec.get(key) != val: raise ValueError(f'{key}: {rec.get(key)!r} != {val!r}')
                    if digest(cp_path) != rec['checkpoint_sha256']:
                        raise ValueError('completed checkpoint SHA256 mismatch')
                    curve_steps = [r['step'] for r in rec['curve']]
                    if curve_steps != list(range(1000, 20001, 1000)):
                        raise ValueError('stage curve is not the fixed 20k endpoint')
                    verified[run] = {k: rec[k] for k in ('run_id','architecture','encoding','task','step',
                        'elapsed_seconds','peak_cuda_allocated_bytes','checkpoint_sha256')}
                    verified[run]['monitor_verified_utc'] = stamp
                    completed_stats[run] = identity
                    if initial_completed is not None:
                        print(json.dumps(dict(event='source_completed', completed_verified=len(verified),
                                             total=48, **verified[run])), flush=True)
                except (OSError, ValueError, KeyError, TypeError) as e:
                    anomalies.append('completion:' + run + ':' + str(e))
            if initial_completed is None:
                initial_completed = sorted(verified)
                print(json.dumps(dict(event='monitor_started', initial_completed=initial_completed,
                                      current_run=state.get('current_run'), utc=stamp)), flush=True)
            procs = queue_processes(); current = {}; du = ds = 0.0
            for p in procs:
                key = str(p['pid']) + ':' + str(p['start_ticks']); current[key] = p
                totals.setdefault(key, dict(pid=p['pid'], role=p['role'], start_ticks=p['start_ticks'],
                                           first_observed_utc=stamp, observed_user_delta_seconds=0.0,
                                           observed_system_delta_seconds=0.0))
                if key in previous:
                    u = max(0.0, p['user_seconds'] - previous[key]['user_seconds'])
                    s = max(0.0, p['system_seconds'] - previous[key]['system_seconds'])
                    du += u; ds += s
                    totals[key]['observed_user_delta_seconds'] += u
                    totals[key]['observed_system_delta_seconds'] += s
                totals[key]['last_observed_utc'] = stamp
            for key in previous.keys() - current.keys():
                observed_exited_processes.append(dict(process_key=key, detected_missing_utc=stamp,
                                                     terminal_cpu_tail_unobserved=True))
            previous = current
            gpu, gpu_error = gpu_snapshot(); gpu_history.append(dict(utc=stamp, devices=gpu))
            queues = [p for p in procs if p['role'] == 'queue']
            children = [p for p in procs if p['role'] != 'queue']
            if len(queues) > 1: anomalies.append('multiple live fixed-queue processes')
            if state.get('status') == 'completed' and state.get('completed_models') != 48:
                anomalies.append('queue completed with non48 model count')
            row = dict(sample=samples, utc=stamp, elapsed_monitor_seconds=time.monotonic()-started_mono,
                       queue_status=state.get('status'), completed_models=state.get('completed_models'),
                       current_run=state.get('current_run'), queue_pids=json.dumps([p['pid'] for p in queues]),
                       queue_user_seconds=sum(p['user_seconds'] for p in queues),
                       queue_system_seconds=sum(p['system_seconds'] for p in queues),
                       descendant_user_seconds=sum(p['user_seconds'] for p in children),
                       descendant_system_seconds=sum(p['system_seconds'] for p in children),
                       process_snapshot_json=json.dumps(procs), gpu_snapshot_json=json.dumps(gpu),
                       gpu_query_error=gpu_error, observed_user_delta_seconds=du,
                       observed_system_delta_seconds=ds, anomalies_json=json.dumps(anomalies))
            writer.writerow(row); csv_file.flush()
            for anomaly in anomalies:
                if anomaly not in reported:
                    event = dict(event='anomaly', utc=stamp, detail=anomaly,
                                 queue_status=state.get('status'), current_run=state.get('current_run'))
                    print(json.dumps(event), flush=True); events.append(event); reported.add(anomaly)
            done = state.get('status') == 'completed' and state.get('completed_models') == len(verified) == 48
            audit = dict(**metadata, status='completed' if done else 'monitoring', last_sample_utc=stamp,
                         samples=samples, elapsed_monitor_seconds=time.monotonic()-started_mono,
                         initial_completed=initial_completed, completed_verified=len(verified),
                         completion_records=list(verified.values()), anomalies=events,
                         process_observation_totals=list(totals.values()),
                         observed_process_exit_gaps=observed_exited_processes,
                         observed_user_delta_seconds=sum(v['observed_user_delta_seconds'] for v in totals.values()),
                         observed_system_delta_seconds=sum(v['observed_system_delta_seconds'] for v in totals.values()),
                         final_queue_state=state, gpu_snapshot_count=len(gpu_history),
                         no_training_process_started_or_signalled=True)
            write_json(DIAG / 'resource_monitor_audit.json', audit)
            if done:
                audit['csv_sha256'] = digest(DIAG / 'resource_monitor.csv')
                write_json(DIAG / 'resource_monitor_audit.json', audit)
                print(json.dumps(dict(event='monitor_completed', completed_verified=48,
                                      samples=samples, anomalies=len(events), utc=stamp)), flush=True)
                break


if __name__ == '__main__':
    main()
