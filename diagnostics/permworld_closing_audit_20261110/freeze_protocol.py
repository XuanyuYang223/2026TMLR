"""Freeze the post-result closing audit before its new calculations."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OLD_NAMES = ['permworld_answer_controls_20261109', 'permworld_orthogonal_alignment_20261110',
             'permworld_exceedance_alignment_20261110', 'permworld_lemma_composition_20261110',
             'permworld_local_causal_20261110', 'permworld_local_timing_20261110']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main():
    if (HERE / 'freeze.json').exists():
        frozen = read(HERE / 'freeze.json')
        assert sha(HERE / 'protocol.json') == frozen['protocol_sha256']
        for path, expected in frozen['spec_hashes'].items():
            assert sha(ROOT / path) == expected, path
        assert sha(__file__) == frozen['freeze_helper_sha256']
        print('Existing closing-audit protocol verified', frozen['protocol_sha256'])
        return
    paths = [HERE / name / 'spec.json' for name in ['alignment', 'gradient', 'ed']]
    for path in paths:
        assert path.is_file(), path
    specs = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    assert all(read(path).get('exploratory', read(path).get('exploratory_post_results_audit'))
               for path in paths)
    protocol = dict(version='PermWorld-three-closing-checks/v1',
        created_utc=datetime.now(timezone.utc).isoformat(), git_head=git('rev-parse', 'HEAD'),
        audit_after_existing_results=True, confirmation_preregistered=False,
        formal_units=['unit02', 'unit03', 'unit04', 'unit05'], historical_reference='archdev001',
        spec_hashes=specs,
        scope='Training-specificity control; one-dimensional early-path geometry/gradient; E/D encoding recheck; writing-only evidence integration. No experimental expansion.',
        preservation='Old reports, papers, F17/DMLR artifacts and caches remain read-only. New files are confined to the new diagnostic/package and external output roots.',
        source_policy='Prefer saved scores, maps, coefficients, scaler, states, logits and exact saved checkpoints. Stage scripts verify actual input hashes. Missing inputs recorded, no training to replace them.',
        analysis_order='Each stage records its code/input freeze before newly requested scores, derivatives or derived comparisons. Rules here and in bound specs are frozen now; engineering failures and fixes must remain logged, with unchanged scientific endpoints/tolerances.',
        training_specificity='Same exact trained peaks source probe/scaler for valleys trained, actual saved step0 target and fixed_points trained. Source identity vs target complement. Only absent fixed-method target-to-source maps fitted on original fit anchors, same budget/centering. Full-test main; original count-matched sensitivity. Old initial-source probe result is not this target control.',
        geometric_coordinates='Raw unnormalized L1 recurrent h_n after right-neighbor value, raw Euclidean metric. Preserve actual cached FP32 displacement separately from ideal donor projection; audit scaler conversion. Report projection energy, gradient norm/angle, actual/linearized signed logit-margin changes and eight original random directions with norm-comparable derivatives.',
        behavior='m=z[c+s]-z[c], c true recipient count and s fixed local sign even for cumulative-count-matched pairs. Autograd at the recipient state through current upper replay from recipient previous states and original suffix. Model weights frozen/eval. No cached/detached intermediate on gradient path.',
        derivative_check='Fixed120 historical+formal cases, n10/20/30,i6, first pair per task/group/sign. Symmetric FP64 smooth fixed-weight shadow graph, epsilon1e-4, atol1e-7+rtol1e-3|analytic|, local and first saved random directions. FP32 primary-logit guards retain1e-4. No adaptive epsilon/tolerance.',
        numerical_undefined='Gradient/displacement/donor norm<=1e-12 leaves relevant cosine/energy undefined. This numerical flag is not a scientific practical-zero threshold. Record finite/undefined counts, no zero imputation.',
        ed_definition='Existing designated inverse CKA minus strongest eligible identity/complement CKA. Raw pooled ordinary, fixed-length U-CKA, fixed-length specified dual-answer additive residual U-CKA, initial and trained. Max after original per-transform aggregation. Keep distinct from old one-wrong Bt/Delta.',
        statistics='Preserve original support and fixed1/21 n10..30 weights. Missing lengths not reweighted. Within-center geometry/gradient weights follow old registered centers. Per-unit contrasts first, four-unit mean/sampleSD/min/max/all raw values. Units are repetitions, positions/lengths/controls/conditions are dependent.',
        interpretation='Report effect sizes and direction. No new significance/equivalence thresholds. Readout specificity does not establish shared computation or causal use; local gradient diagnoses only this intervention neighborhood; answer mean residualization does not remove all answer information. Original encoding double-dissociation prediction remains failed.',
        writing='Separate geometry, fitted readout transfer, analyst-supplied composition, within-model intervention questions; not a shrinking single effect or direct test of cross-model shared computation. New draft files only, no overwrite of manuscript.',
        stop='Finish these three checks and writing assets, then stop. No base training, probe reselection/refit, direction/layer/time/dose search, DAS, cross-model exchange, new-world or downstream tests, automatic upload/publication or DMLR-direction edits.')
    # Named metadata/source/manuscript protection only; large payloads are
    # separately bound and verified by the stages that actually consume them.
    protected = {ROOT / 'CURRENT_RESEARCH.txt', ROOT / 'README.md'}
    for name in OLD_NAMES:
        directory = ROOT / 'diagnostics' / name
        for path in directory.rglob('*'):
            if (path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts
                    and path.suffix in {'.py', '.json', '.md', '.txt', '.html', '.csv'}
                    and path.stat().st_size <= 20_000_000):
                protected.add(path)
        experiment = ROOT / 'experiments' / name
        protected.update(path for path in experiment.rglob('*.py') if '__pycache__' not in path.parts)
    protected.update(path for path in (ROOT / 'manuscripts').rglob('*') if path.is_file())
    protected.update(ROOT / 'tests' / name for name in ['test_permworld_local_timing_replay.py',
                                                       'test_permworld_local_causal_recurrent.py'])
    files = {str(path): sha(path) for path in sorted(protected) if path.is_file()}
    write(HERE / 'protection_start.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        git_head=protocol['git_head'], git_status=git('status', '--porcelain=v1'),
        scope='Explicit named reports/source/metadata<=20MB and all working-manuscript files; excludes symlinked raw production rows, large caches and unrelated files. Per-stage input manifests bind payloads actually used. Endpoint evidence only.',
        files=files))
    write(HERE / 'protocol.json', protocol)
    write(HERE / 'freeze.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(HERE / 'protocol.json'), spec_hashes=specs,
        freeze_helper_sha256=sha(__file__), protection_start_sha256=sha(HERE / 'protection_start.json'),
        before_new_requested_computations=True, prior_results_known=True,
        stage_execution_code_frozen_separately_before_each_first_new_calculation=True))
    print(json.dumps(dict(status='frozen', protocol_sha256=sha(HERE / 'protocol.json'),
                          named_protected_files=len(files), specs=specs), indent=2))


if __name__ == '__main__':
    main()
