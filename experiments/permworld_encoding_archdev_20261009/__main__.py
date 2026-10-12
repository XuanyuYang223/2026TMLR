"""Every stage is explicit; setup and CPU validation never launch GPU fitting."""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from .common import setup
from .checks import cpu_check
from .training import prepare, overfit, resume, train_all, run_job
from .evaluation import evaluate, report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['setup','cpu-check','prepare','overfit','resume','train','train-one','evaluate','report'])
    parser.add_argument('--run-id');parser.add_argument('--stop-step',type=int);parser.add_argument('--output')
    args=parser.parse_args()
    if args.stage=='setup':setup()
    elif args.stage=='cpu-check':cpu_check()
    elif args.stage=='prepare':prepare()
    elif args.stage=='overfit':overfit()
    elif args.stage=='resume':resume()
    elif args.stage=='train':train_all()
    elif args.stage=='train-one':
        if args.run_id is None:parser.error('--run-id is required for train-one')
        run_job(args.run_id,args.stop_step,args.output)
    elif args.stage=='evaluate':evaluate()
    elif args.stage=='report':report()


if __name__=='__main__':main()
