"""Explicit pipeline stages; preparation never starts GPU work."""
import argparse
import json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,choices=['prepare','engineering','train-one','train-all','evaluate','report','verify'])
    p.add_argument('--run-id');p.add_argument('--stop-step',type=int,default=20000);p.add_argument('--output')
    a=p.parse_args()
    if a.stage=='prepare':
        from .setup import prepare_all
        prepare_all()
    elif a.stage=='verify':
        from .common import verify,source_version
        c=verify();print(json.dumps(dict(status='passed',version=source_version(),source_models=len(c['training_manifest']))))
    elif a.stage=='engineering':
        from .training import engineering
        engineering()
    elif a.stage=='train-one':
        assert a.run_id
        from .training import run_job
        run_job(a.run_id,a.stop_step,a.output)
    elif a.stage=='train-all':
        from .training import train_all
        train_all()
    elif a.stage=='evaluate':
        from .evaluation import evaluate
        evaluate()
    elif a.stage=='report':
        from .report import report
        report()

if __name__=='__main__':main()
