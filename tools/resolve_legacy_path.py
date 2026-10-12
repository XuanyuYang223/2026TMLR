"""Resolve original scientific file references without host-path fallbacks.

Frozen JSON/source bytes are never rewritten. This is a publication wrapper:
repository source references map to the clone; released scientific binaries
map to verified regular content-addressed files under --restore-dir.
It does not run scientific code, a model or training.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path,PurePosixPath

ORIGINAL_REPO='/home/yangx/ICML/'

def resolve_legacy_path(original:str,repo:Path,restore:Path,manifest:dict,expected_sha:str|None=None)->Path:
 repo=repo.resolve();restore=restore.resolve()
 entries=[r for r in manifest['files'] if r['logical_path']==original]
 wanted=expected_sha
 if wanted is None and entries:
  values={r['sha256'] for r in entries}
  if len(values)!=1:raise ValueError('Ambiguous publication mapping')
  wanted=next(iter(values))
 if original.startswith(ORIGINAL_REPO):
  rel=PurePosixPath(original[len(ORIGINAL_REPO):])
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe original source reference')
  p=(repo/str(rel)).resolve()
  if p.is_relative_to(repo) and p.is_file():
   with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
   if wanted is None or digest==wanted:return p
 if wanted is None:raise FileNotFoundError('Not in the published logical-file manifest')
 objects={r['sha256']:r for r in manifest['objects']}
 if wanted not in objects:raise FileNotFoundError('The requested original version is not present in the published objects')
 p=(restore/'objects'/wanted).resolve()
 if not p.is_relative_to(restore) or not p.is_file():raise FileNotFoundError('Download and restore the mapped package first')
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 if p.stat().st_size!=objects[wanted]['bytes'] or digest!=wanted:raise ValueError('Restored original bytes do not match manifest')
 return p

def main():
 q=argparse.ArgumentParser(description=__doc__);q.add_argument('original');q.add_argument('--expected-sha',help='Require the exact frozen version, not merely the same historical filename');q.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1]);q.add_argument('--restore-dir',type=Path,default=Path('artifacts/restored'));q.add_argument('--manifest',type=Path,default=Path('artifacts_manifest.json'));a=q.parse_args()
 m=json.loads(a.manifest.read_text());p=resolve_legacy_path(a.original,a.repo_root,a.restore_dir,m,a.expected_sha)
 print(p)
if __name__=='__main__':main()
