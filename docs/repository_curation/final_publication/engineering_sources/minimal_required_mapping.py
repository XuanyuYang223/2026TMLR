#!/usr/bin/env python3
"""Audit the 206 required aliases after the fixed clean clone has succeeded.

Read-only engineering, except for a new external audit JSON. No network,
checkpoint loading, model imports, forwards, hidden metrics or training.
Default execution is preparation only; --run is gated on actual clone success.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
import time

BASE=Path('/mnt/yangx-projects-f/20261011/representation_publication')
RUN=BASE/'clean_clone_runs/run001'
MANIFEST_SHA='f1ef48da6915a4ad6632e100ab09171f2f74690e2d4fcd49d9254436a7ce394f'
CSV_SHA='7f732b783dce84faa9a1c2054524ec17d04cafe61c65ba97be29ae917b9a7e99'
MAIN='29ec0802987fda8d1c5b9986426233d4ceb4b4c8'
SOURCE='83325c2b6a2e41c69fc8b2e8f25e0cb58b80aca2'
TAG='representation-selectivity-2026-10-11-caveat'
SUCCESS='passed_actual_public_clean_clone_minimal_readback_with_preserved_scientific_caveat'

class MappingError(Exception):pass

def require(condition,message):
 if not condition:raise MappingError(message)

def sha(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def regular(root,relative):
 require(not root.is_symlink()and root.resolve().is_relative_to(BASE.resolve()),
         'Audit root is a symlink or outside the fixed publication tree')
 current=BASE
 for component in root.relative_to(BASE).parts:
  current=current/component
  require(not current.is_symlink(),'Symlink in a fixed publication root path')
 path=Path(relative)
 require(not path.is_absolute()and '..'not in path.parts,'Unsafe relative audit input')
 target=root/path
 current=root
 for component in path.parts:
  current=current/component
  require(not current.is_symlink(),'Symlink in a clone/restore input path')
 require(target.is_file()and stat.S_ISREG(target.stat().st_mode)and
         target.resolve().is_relative_to(root.resolve()),'Required regular clone/restore input is missing or outside its root')
 return target

def unique(rows,key,label):
 result={x[key]:x for x in rows}
 require(len(result)==len(rows),'Repeated '+label+' identity')
 return result

def identify(path):return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}

def run(rehash=False):
 start=time.monotonic()
 result={'status':'running_required_mapping_audit','utc':datetime.now(timezone.utc).isoformat(),
         'source_sha256':sha(Path(__file__)),'inputs':{},'records':[],
         'optional_current_CAS_SHA_rehash_requested':rehash,
         'remote_requests':0,'model_forwards':0,'checkpoint_loads':0,'training_updates':0,
         'new_scientific_metrics':0,'original_host_fallbacks':0}
 try:
  # Do not inspect a partial clone or download directory before success.
  cp=regular(RUN,'clean_clone_audit.json');completed=json.loads(cp.read_text())
  require(completed['status']==SUCCESS and completed['actual_clone_commit']==MAIN and
          completed['scientific_source_commit']==SOURCE and completed['release_tag']==TAG,
          'Fixed actual clean clone has not succeeded with its original bindings')
  result['inputs']['actual_clean_clone_audit']=identify(cp)
  clone=RUN/'clone';restore=RUN/'fresh_restored'
  manifest_path=regular(clone,'artifacts_manifest.json')
  require(sha(manifest_path)==MANIFEST_SHA and completed['manifest']['sha256']==MANIFEST_SHA,
          'Fixed manifest SHA binding changed')
  manifest=json.loads(manifest_path.read_text());result['inputs']['clone_manifest']=identify(manifest_path)
  require(manifest['scientific_source_commit']==SOURCE and manifest['release_tag']==TAG,'Manifest source/tag changed')
  csv_path=regular(clone,'docs/repository_curation/minimal_profile_required_files.csv')
  require(sha(csv_path)==CSV_SHA,'Required-file CSV original bytes changed')
  result['inputs']['required_file_CSV']=identify(csv_path)
  with csv_path.open(newline='')as f:
   reader=csv.DictReader(f)
   require(reader.fieldnames==['logical_path','restore_path','sha256','bytes','package_id','selection_reasons'],
           'Required-file CSV schema changed')
   required=list(reader)
  require(len(required)==206,'Required alias count differs from frozen 206')
  csv_by_path=unique(required,'logical_path','required logical path')
  detail=manifest['profile_details']['minimal'];reference=unique(detail['required_logical_files'],'logical_path','manifest required path')
  require(set(csv_by_path)==set(reference),'Required CSV and manifest logical aliases differ')
  objects=unique(manifest['objects'],'sha256','manifest content object')
  files=unique(manifest['files'],'logical_path','manifest logical path')
  packages=unique(manifest['packages'],'package_id','manifest package')
  assets=unique(manifest['assets'],'name','manifest asset')
  selected=manifest['profiles']['minimal']
  require(len(selected)==13 and len(set(selected))==13,'Fixed minimal 13-package profile changed')
  selected_objects={s for p in selected for s in packages[p]['object_sha256']}
  selected_names={n for p in selected for n in packages[p]['parts']}
  require(len(selected_objects)==1974 and len(selected_names)==13,'Fixed minimal content/asset counts changed')
  dp=regular(RUN,'public_minimal_download_audit.json');download=json.loads(dp.read_text())
  require(sha(dp)==completed['actual_remote_minimal_readback']['audit_sha256'],
          'Actual downloader audit no longer matches the completed clone binding')
  result['inputs']['actual_public_download_audit']=identify(dp)
  require(download['status']=='passed_selected_artifact_restore' and download['profile']=='minimal' and
          download['manifest_sha256']==MANIFEST_SHA and download['selected_package_ids']==selected and
          download['selected_packages']==13 and download['selected_unique_objects']==1974 and
          set(download['remote_assets_read'])==selected_names and download['reused_download_assets']==[] and
          download['remote_bytes_read']==3771371797,
          'Actual minimal remote readback scope/counts differ')
  verified_assets=unique(download['assets'],'name','downloaded asset')
  require(set(verified_assets)==selected_names,'Actual asset records incomplete')
  for name,actual in verified_assets.items():
   expected=assets[name]
   require(actual['source']=='remote_network'and actual['network_read']is True and actual['verified']is True and
           actual['sha256']==expected['sha256']and actual['bytes_read']==expected['bytes']and
           actual['bytes_expected']==expected['bytes'],'Asset is not exact genuinely remote verified bytes')
  verified_packages=unique(download['packages'],'package_id','verified package stream')
  require(set(verified_packages)==set(selected),'Verified compressed streams incomplete')
  for pid,actual in verified_packages.items():
   require(actual['compressed_stream_verified']is True and actual['package_index_present']is True and
           actual['extraction']=='verified_regular_expected_members_only' and
           actual['objects']==len(packages[pid]['object_sha256']),'Package stream verification scope differs')
  verified_objects=unique(download['objects'],'sha256','verified CAS object')
  require(set(verified_objects)==selected_objects,'Selected object stream-verification records incomplete')
  for digest,actual in verified_objects.items():
   require(actual['source']=='verified_tar_object' and actual['verified']is True and actual['bytes']==objects[digest]['bytes'],
           'Object is not verified from an actual selected tar stream')
  mp=regular(restore,'path_map.json');mapping=json.loads(mp.read_text())
  require(isinstance(mapping,dict)and sha(mp)==download['path_map']['sha256'],
          'Restored path map is invalid or differs from verified download binding')
  result['inputs']['restored_path_map']=identify(mp)
  expected_mapping={f['logical_path']:'objects/'+f['sha256']for f in manifest['files']if f['sha256']in selected_objects}
  require(mapping==expected_mapping and len(mapping)==2074 and download['path_map']['logical_paths']==2074,
          'Default full selected logical alias map differs')
  require(download['materialized']==[] and download['materialize_requested']is False,
          'Fixed default-CAS execution unexpectedly materialized logical copies')
  rehashed={};checked_unique=set()
  for row in required:
   logical=row['logical_path'];digest=row['sha256'];expected=reference[logical]
   require(all(row[k]==expected[k]for k in ['restore_path','sha256','package_id'])and
           int(row['bytes'])==expected['bytes'],'Required CSV identity differs from its manifest detail')
   require(files[logical]['sha256']==digest and files[logical]['bytes']==int(row['bytes'])and
           digest in selected_objects and objects[digest]['package_id']==row['package_id'],
           'Required alias is not the matching selected content object')
   require(mapping[logical]=='objects/'+digest,'Required alias maps to wrong CAS SHA')
   cas=regular(restore,'objects/'+digest)
   require(cas.stat().st_size==int(row['bytes']),'Required current regular CAS byte size differs')
   if rehash and digest not in rehashed:
    rehashed[digest]=sha(cas)
    require(rehashed[digest]==digest,'Optional required-CAS current-byte SHA mismatch')
   checked_unique.add(digest);pid=row['package_id']
   result['records'].append({'logical_path':logical,'manifest_restore_path':row['restore_path'],
       'CAS_relative_path':mapping[logical],'CAS_path':str(cas),'sha256':digest,'bytes':int(row['bytes']),
       'package_id':pid,'source_remote_asset_names':packages[pid]['parts'],
       'source_compressed_stream_sha256':packages[pid]['compressed_stream_sha256'],
       'actual_remote_asset_records_verified':True,'tar_stream_object_SHA_verified':True,
       'current_CAS_regular_file_and_size_checked':True,
       'optional_current_CAS_SHA256':rehashed.get(digest),
       'current_CAS_SHA_independently_rehashed':rehash})
  require(len(checked_unique)==206 and sum(objects[s]['bytes']for s in checked_unique)==1549329772,
          'Frozen required unique object scope changed')
  result.update(status='passed_required_logical_mapping_with_actual_remote_stream_verification',
     required_logical_aliases_checked=206,required_unique_content_objects_checked=206,
     required_unique_content_bytes=1549329772,actual_selected_package_assets=13,
     actual_selected_remote_bytes=3771371797,actual_selected_unique_CAS_objects=1974,
     all_selected_path_map_aliases=2074,collateral_unique_objects=1768,collateral_logical_aliases=1868,
     actual_logical_regular_materialized_copies=0,
     optional_current_SHA_rehashed_unique_objects=len(rehashed),
     optional_current_SHA_rehashed_bytes=sum(objects[s]['bytes']for s in rehashed),
     source_packages=[{'package_id':p,'parts':packages[p]['parts'],
         'compressed_stream_sha256':packages[p]['compressed_stream_sha256']}for p in selected],
     SHA_scope='All selected objects were SHA-verified by the bound actual public downloader. This script independently reconciles the206 required CSV/manifest/path-map identities and current regular file sizes. Current CAS content is independently rehashed only if optional_current_CAS_SHA_rehash_requested=true.',
     scope='206 frozen required aliases only, among1974 CAS objects and2074 logical aliases from13 actual remote assets. No full release reread, model replay or alteration of original numerical caveat.')
 except MappingError as exc:result.update(status='failed_required_mapping_audit',error=str(exc))
 except Exception as exc:result.update(status='failed_required_mapping_audit',error='Engineering failure ('+type(exc).__name__+'); raw URLs/credentials omitted')
 result['elapsed_process_wall_seconds']=time.monotonic()-start
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--run',action='store_true')
 parser.add_argument('--rehash-required',action='store_true',help='Extra current-byte SHA of only206 required CAS objects (1.55GB); never hashes collateral objects')
 parser.add_argument('--output',type=Path,default=BASE/'source_review/minimal_required_mapping_audit.json')
 args=parser.parse_args()
 if not args.run:
  print('Prepared only: run only after the fixed actual clean-clone audit succeeds. No pending clone/restore result inspected, remote request, model or training.');raise SystemExit(0)
 require(args.output.absolute().resolve().is_relative_to(BASE.resolve()),'Audit output must remain outside the repository under the publication BASE')
 require(not args.output.exists()and not args.output.is_symlink(),'Existing audit retained; choose a new explicit output instead of overwriting')
 result=run(args.rehash_required)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 with args.output.open('x')as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
 print(json.dumps({'status':result['status'],'output':str(args.output),'sha256':sha(args.output)}))
 raise SystemExit(not result['status'].startswith('passed_'))
