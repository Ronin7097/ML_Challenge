"""Verify and package the selected DARPA submission; standard library only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile


SOURCES = {
    'prepare.py', 'encode.py', 'train_encoder.py', 'retrieve_owners.py',
    'pair_features.py', 'build_features.py', 'fit_owner.py', 'predict_frozen.py',
    'rich_context.py', 'build_rich_context.py', 'fit_rich_context.py',
    'rich_policy.py', 'predict_rich.py', 'evaluate_tune.py', 'audit_submission.py',
    'validate_submission_official.py', 'test_gradient_cache.py',
    'test_pair_features.py', 'test_rich_context.py', 'test_rich_inference.py',
    'package_submission.py',
}
EXTERNAL = {'anyascii', 'duckdb', 'lightgbm', 'numpy', 'pyarrow', 'rapidfuzz', 'torch', 'transformers'}
STDLIB = set(getattr(sys, 'stdlib_module_names', ())) | {
    '__future__', 'argparse', 'ast', 'collections', 'concurrent', 'contextlib',
    'copy', 'csv', 'functools', 'hashlib', 'importlib', 'itertools', 'json', 'math',
    'multiprocessing', 'os', 'pathlib', 'random', 're', 'shutil', 'struct',
    'subprocess', 'sys', 'tempfile', 'time', 'types', 'typing', 'unittest', 'unicodedata', 'zipfile',
}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def zip_digest(archive, name):
    value = hashlib.sha256()
    with archive.open(name) as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def verify(root):
    code = root / 'code/business_entity_resolution'
    source = code / 'src'
    present = {p.name for p in source.glob('*.py')}
    if present != SOURCES:
        raise ValueError(f'Submission source differs: missing={SOURCES-present}; extra={present-SOURCES}')
    local_modules = {Path(n).stem for n in SOURCES}
    for path in sorted(source.glob('*.py')):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [n.name for n in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                if module and module.split('.')[0] not in local_modules | EXTERNAL | STDLIB:
                    raise ValueError(f'Undeclared dependency in {path.name}: {module}')
    validation = json.loads((code / 'validation.json').read_text())
    run = json.loads((code / 'run.json').read_text())
    if any(validation[n] != 'PASS' for n in ('official_both_files_and_ids', 'match_subset', 'source1_order')):
        raise ValueError('Full output validation is missing')
    if not run['complete_submission'] or run['source1_rows'] != validation['source1_rows']:
        raise ValueError('Incomplete producing run')
    for name, key in [('matching_results.tsv', 'matching_sha256'), ('candidate_pairs.tsv', 'candidate_sha256')]:
        actual = digest(root / 'output' / name)
        if actual != validation[key] or actual != run[key]:
            raise ValueError(f'{name} differs from the fully audited prediction file')
    if digest(source / 'validate_submission_official.py') != validation['official_validator_sha256']:
        raise ValueError('Official validator is not the audited version')
    for name, expected in run['code_sha256'].items():
        if digest(source / name) != expected:
            raise ValueError(f'Producing inference source changed: {name}')
    manifest = json.loads((code / 'models/manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid model manifest path')
        if digest(code / 'models' / path) != expected:
            raise ValueError(f'Model file changed: {name}')
    policy_path = code / 'models/rich/policy.json'
    complete_path = code / 'models/rich/complete.json'
    policy = json.loads(policy_path.read_text())
    complete = json.loads(complete_path.read_text())
    if digest(policy_path) != run['policy_sha256'] or digest(complete_path) != run['model_complete_sha256']:
        raise ValueError('Producing rich policy/model identity differs')
    if policy['rich_complete_sha256'] != digest(complete_path):
        raise ValueError('Policy does not identify the packaged rich model')
    for name, expected in policy['base_model_sha256'].items():
        if digest(code / 'models/owner' / name) != expected:
            raise ValueError(f'Base model changed: {name}')
    for name, expected in complete['model_sha256'].items():
        if digest(code / 'models/rich' / name) != expected:
            raise ValueError(f'Rich weights changed: {name}')
    files = [root / 'Documentation_template.md', root / 'output/matching_results.tsv', root / 'output/candidate_pairs.tsv']
    files += [code / name for name in ('README.md', 'requirements.txt', 'requirements-rich.txt', 'LICENSE', 'run.json', 'validation.json', 'models/manifest.json')]
    files += [source / name for name in sorted(SOURCES)]
    files += [code / 'models' / name for name in sorted(manifest['files_sha256'])]
    files += [code / 'splits/previous_development_ids.txt']
    files += [code / 'licenses' / name for name in ('Granite-APACHE-2.0.txt', 'Granite-model-card.md', 'MODEL_NOTICE.md')]
    files += sorted((code / 'validation').glob('*.json'))
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Submission needs a regular file: {path}')
    result = dict(source1_rows=validation['source1_rows'], matches=validation['matches'],
                  candidate_pairs=validation['candidate_pairs'], output_identities='PASS',
                  model_identities='PASS', producing_code_identities='PASS', dependency_closure='PASS',
                  official_both_files_and_ids=validation['official_both_files_and_ids'])
    return files, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    archive = args.archive.resolve() if args.archive else root / 'DARPA_submission.zip'
    if not args.check_only and archive.exists():
        raise ValueError('Refuse to overwrite an archive; preserve or rename the existing file first')
    files, result = verify(root)
    if args.check_only:
        print(json.dumps(result, indent=2))
        return
    expected = {p.relative_to(root).as_posix(): digest(p) for p in files}
    with zipfile.ZipFile(archive, 'w', allowZip64=True) as output:
        for path in sorted(files):
            compression = zipfile.ZIP_STORED if path.suffix == '.safetensors' else zipfile.ZIP_DEFLATED
            output.write(path, path.relative_to(root).as_posix(), compress_type=compression, compresslevel=5)
    with zipfile.ZipFile(archive) as output:
        if len(output.namelist()) != len(expected) or set(output.namelist()) != set(expected):
            raise ValueError('ZIP members differ from submission manifest')
        if output.testzip() is not None:
            raise ValueError('ZIP CRC failed')
        for name, expected_hash in expected.items():
            if zip_digest(output, name) != expected_hash:
                raise ValueError(f'ZIP member differs: {name}')
    result.update(archive=archive.name, archive_sha256=digest(archive), archive_crc='PASS',
                  archive_member_identities='PASS', entries=len(expected), file_sha256=expected)
    (root / 'submission_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'file_sha256'}, indent=2))


if __name__ == '__main__':
    main()
