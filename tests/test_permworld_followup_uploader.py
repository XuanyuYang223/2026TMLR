"""Synthetic publication-engineering checks: no network or scientific bytes."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("followup_uploader", Path(__file__).resolve().parents[1] / "tools/upload_permworld_followups.py")
u = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(u)


def remote(path):
    raw = path.read_bytes()
    return dict(name=path.name, state="uploaded", size=len(raw), digest="sha256:"+hashlib.sha256(raw).hexdigest())


class UploaderTests(unittest.TestCase):
    def test_asset_guards(self):
        a=dict(name="x", state="uploaded", size=1, digest="sha256:abc")
        u.validate_asset(a, "x", 1, "abc")
        for bad in [dict(a,state="starter"),dict(a,size=2),dict(a,digest=None),dict(a,name="z")]:
            with self.assertRaises(ValueError):u.validate_asset(bad,"x",1,"abc")

    def test_proc_zombie_and_live(self):
        with patch.object(Path,"read_text",return_value="Name:\tgh\nState:\tZ (zombie)\n"):
            self.assertFalse(u.process_alive(123))
        with patch.object(Path,"read_text",return_value="Name:\tgh\nState:\tS (sleeping)\n"):
            self.assertTrue(u.process_alive(123))
        with patch.object(Path,"read_text",side_effect=FileNotFoundError):
            self.assertFalse(u.process_alive(123))

    def test_reuse_is_provider_confirmed_without_upload(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"representation-one.tar.zst";p.write_bytes(b"fake-package")
            with patch.object(u,"get_draft",return_value=({}, {p.name:remote(p)})),patch.object(u,"gh") as gh:
                got=u.upload_one(p,1,u.Progress(Path(folder)/"progress.jsonl"))
            self.assertEqual(got["bytes"],12);gh.assert_not_called()

    def test_partial_retained_never_clobbered(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"representation-one.tar.zst";p.write_bytes(b"x")
            with patch.object(u,"get_draft",return_value=({}, {p.name:dict(remote(p),state="starter")})),patch.object(u,"gh") as gh:
                with self.assertRaises(ValueError):u.upload_one(p,1,u.Progress(Path(folder)/"progress.jsonl"))
            gh.assert_not_called()

    def test_lost_response_confirmed_no_duplicate_upload(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"representation-one.tar.zst";p.write_bytes(b"x")
            with patch.object(u,"get_draft",side_effect=[({},{}),({}, {p.name:remote(p)})]),patch.object(u,"gh",side_effect=RuntimeError("network")) as gh:
                u.upload_one(p,1,u.Progress(Path(folder)/"progress.jsonl"))
            self.assertEqual(gh.call_count,1)

    def test_network_retry_and_post_upload_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"representation-one.tar.zst";p.write_bytes(b"x")
            responses=[({},{}),({},{}),({},{}),({}, {p.name:remote(p)})]
            with patch.object(u,"get_draft",side_effect=responses),patch.object(u,"gh",side_effect=[RuntimeError("network"),""]) as gh,patch.object(u.time,"sleep"):
                u.upload_one(p,1,u.Progress(Path(folder)/"progress.jsonl"))
            self.assertEqual(gh.call_count,2)
            self.assertTrue(all("--clobber" not in c.args for c in gh.call_args_list))

    def test_refresh_network_retry(self):
        with patch.object(u,"get_draft",side_effect=[RuntimeError("network"),({}, {})]) as get,patch.object(u.time,"sleep"):
            self.assertEqual(u.refresh_draft(1),({},{}))
            self.assertEqual(get.call_count,2)

    def test_three_workers_and_final_provider_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);assets={};lock=threading.Lock();barrier=threading.Barrier(3);activity=[0,0]
            paths=[]
            for i in range(3):
                p=base/f"representation-{i}.tar.zst";p.write_bytes(bytes([i])*8);paths.append(p)
            final=base/"final.json"
            final.write_text(json.dumps(dict(repository=u.REPO,release_tag=u.TAG,assets=[dict(name=p.name,bytes=p.stat().st_size,sha256=remote(p)["digest"][7:]) for p in paths])))
            def inventory(_):
                with lock:return {},dict(assets)
            def upload(*args):
                p=Path(args[3])
                with lock:activity[0]+=1;activity[1]=max(activity)
                barrier.wait(timeout=5)
                with lock:assets[p.name]=remote(p);activity[0]-=1
                return ""
            args=argparse.Namespace(packages=base,poll_seconds=.005,workers=3,release_id=1,inflight_pid=None,inflight_name=None,manifest=final)
            with patch.object(u,"get_draft",side_effect=inventory),patch.object(u,"gh",side_effect=upload):
                u.run(args)
            self.assertEqual(activity[1],3)
            events=[json.loads(x) for x in (base/"upload_progress.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["event"],"all_manifest_packages_provider_confirmed")
            self.assertFalse(events[-1]["local_full_stream_verification_asserted"])

    def test_inherited_alive_upload_is_not_claimed(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);p=base/"representation-inflight.tar.zst";p.write_bytes(b"x")
            final=base/"final.json";final.write_text(json.dumps(dict(repository=u.REPO,release_tag=u.TAG,assets=[dict(name=p.name,bytes=1,sha256=remote(p)["digest"][7:])])))
            alive=[True,True,False,False,False]
            def status(_):return alive.pop(0) if alive else False
            args=argparse.Namespace(packages=base,poll_seconds=.001,workers=3,release_id=1,inflight_pid=123,inflight_name=p.name,manifest=final)
            with patch.object(u,"get_draft",return_value=({}, {p.name:remote(p)})),patch.object(u,"process_alive",side_effect=status),patch.object(u,"gh") as gh:
                u.run(args)
            gh.assert_not_called()
            events=[json.loads(x)["event"] for x in (base/"upload_progress.jsonl").read_text().splitlines()]
            self.assertLess(events.index("inherited_upload_finished"),events.index("asset_claim"))


if __name__ == "__main__":
    unittest.main()
