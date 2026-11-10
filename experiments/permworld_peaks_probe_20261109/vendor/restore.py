import hashlib
import json
import pathlib
import re
import subprocess
import tarfile

manifest = json.loads(pathlib.Path('artifacts_manifest.json').read_text())
objects = {obj['sha256']: obj for obj in manifest['objects']}
selection = {}
for item in manifest['files']:
    path = item['logical_path']
    name = None
    if path.endswith('/encoding_archdev/worlds/archdev001/data.npz'):
        name = 'data.npz'
    elif path.endswith('/encoding_archdev/protocol.json'):
        name = 'protocol.json'
    elif path.endswith('/encoding_archdev/initializations/gru/step0.pt'):
        name = 'initial.pt'
    else:
        match = re.search(r'/models/archdev001_gru_(canonical_cycle|one_line)_(peaks|valleys|descents|recoils)/checkpoint\.pt$', path)
        if match:
            name = '-'.join(match.groups()) + '.pt'
    if name:
        selection[name] = {**item, **objects[item['sha256']]}
assert len(selection) == 11, selection.keys()
for asset in manifest['assets']:
    wanted = {x['archive_member']: (name, x) for name, x in selection.items() if x['package_id'] == asset['package_id']}
    if not wanted:
        continue
    path = pathlib.Path('downloads', asset['name'])
    with path.open('rb') as source:
        assert hashlib.file_digest(source, 'sha256').hexdigest() == asset['sha256'], path
    with subprocess.Popen(['zstd', '-dc', str(path)], stdout=subprocess.PIPE) as proc:
        with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
            for member in archive:
                if member.name not in wanted:
                    continue
                name, obj = wanted.pop(member.name)
                assert member.isfile() and member.size == obj['bytes']
                data = archive.extractfile(member).read()
                assert hashlib.sha256(data).hexdigest() == obj['sha256']
                pathlib.Path('probe_inputs', name).write_bytes(data)
                print(name, member.size, flush=True)
        assert not wanted, wanted
        proc.stdout.read(1)
        assert proc.wait() == 0
pathlib.Path('probe_inputs/source_inventory.json').write_text(json.dumps(selection, indent=2) + '\n')
