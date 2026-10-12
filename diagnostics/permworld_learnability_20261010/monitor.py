"""Read-only status, validation-only; no checkpoint loads or metric selection."""
import ast
import json
from pathlib import Path

ROOT = Path('/mnt/yangx-projects-f/20261010/permworld_learnability')


def main():
    models = []
    for directory in sorted((ROOT / 'models').glob('*')):
        if not directory.is_dir():
            continue
        stages = [json.loads(p.read_text()) for p in directory.glob('stage_*.json')]
        if not stages:
            continue
        last = max(stages, key=lambda x:x['step'])
        val = last.get('validation')
        models.append(dict(run_id=last['run_id'], step=last['step'], status=last['status'],
            validation_accuracy=val['overall_full_vocab'] if val else None,
            validation_answer_ce=val['answer_ce'] if val else None))
    log = ROOT / 'driver.log'
    recent = []
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-20000))
            lines = stream.read().decode(errors='replace').splitlines()
        for line in lines:
            try:
                parsed = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, dict) and 'run' in parsed:
                recent.append(parsed)
    execution = ROOT / 'development_execution.json'
    print(json.dumps(dict(saved_stages=models, recent_log_records=recent[-3:],
        execution=json.loads(execution.read_text()) if execution.exists() else None), ensure_ascii=False))


if __name__ == '__main__':
    main()
