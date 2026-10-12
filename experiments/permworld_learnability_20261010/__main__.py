"""All new development stages are explicit; no formal or GroupWorld entry."""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['setup', 'prepare', 'cpu-check',
        'resume', 'train-one', 'develop', 'report'])
    parser.add_argument('--run-id'); parser.add_argument('--stop-step', type=int)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.stage == 'setup':
        from .setup import setup
        setup()
    elif args.stage == 'prepare':
        from .data import prepare
        prepare()
    elif args.stage == 'cpu-check':
        from .checks import cpu_check
        cpu_check()
    elif args.stage == 'resume':
        from .checks import resume_check
        resume_check()
    elif args.stage == 'train-one':
        if not args.run_id or args.stop_step is None:
            parser.error('--run-id and --stop-step required')
        from .training import run_job
        run_job(args.run_id, args.stop_step, args.output)
    elif args.stage == 'develop':
        from .driver import develop
        develop()
    else:
        from .report import report
        report()


if __name__ == '__main__':
    main()
