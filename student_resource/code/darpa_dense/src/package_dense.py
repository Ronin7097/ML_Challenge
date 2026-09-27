"""Build and verify the self-contained DARPA dense-owner submission archive."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def zip_digest(archive,name):
    h=hashlib.sha256()
    with archive.open(name) as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True,
                   help='Directory with student_resource and original DARPA archive')
    p.add_argument('--output',type=Path,required=True,
                   help='Validated new output directory with TSVs and run.json')
    p.add_argument('--validation',type=Path,required=True,
                   help='Final validator result JSON with streaming/official PASS')
    p.add_argument('--archive',type=Path,required=True,
                   help='New DARPA_submission.zip path; original archive is never replaced')
    a=p.parse_args();project=a.project.resolve();output=a.output.resolve()
    experiment=project/'student_resource/code/darpa_dense'
    if a.archive.resolve()==(project/'DARPA_submission.zip').resolve():
        raise ValueError('Preserve the original baseline archive')
    if a.archive.exists():raise ValueError('Refuse to overwrite an existing archive')
    run=json.loads((output/'run.json').read_text())
    valid=json.loads(a.validation.read_text())
    if not run['complete_submission'] or run['source1_rows']!=1732544:
        raise ValueError('Full test output not complete')
    if set(run['countries'])!={'France','India','US'}:
        raise ValueError('Test country coverage differs')
    for name,key in [('matching_results.tsv','matching_sha256'),
                     ('candidate_pairs.tsv','candidate_sha256')]:
        if digest(output/name)!=run[key]:raise ValueError(f'{name} differs from run manifest')
    if (valid.get('streaming')!='PASS' or valid.get('official_match_ids')!='PASS' or
            valid.get('matching_sha256')!=run['matching_sha256'] or
            valid.get('candidate_sha256')!=run['candidate_sha256']):
        raise ValueError('Validator manifest is incomplete or mismatched')
    frozen=json.loads((experiment/'reports/frozen_policy.json').read_text())
    if not frozen['frozen'] or frozen['selected_policy']!='owner_context':
        raise ValueError('Selected policy differs from frozen report')
    source=experiment/'src'
    prefix='code/business_entity_resolution/'
    files={
        'output/matching_results.tsv':output/'matching_results.tsv',
        'output/candidate_pairs.tsv':output/'candidate_pairs.tsv',
        prefix+'README.md':experiment/'README.md',
        prefix+'EXPERIMENTS.md':experiment/'EXPERIMENTS.md',
        prefix+'PUBLIC_APPROACH_REVIEW.md':experiment/'PUBLIC_APPROACH_REVIEW.md',
        prefix+'requirements.txt':experiment/'requirements.txt',
        prefix+'LICENSE':experiment/'LICENSE',
        prefix+'validation.json':a.validation,
        prefix+'splits/previous_development_ids.txt':experiment/'splits/previous_development_ids.txt',
        'Documentation_template.md':experiment/'Documentation_template.md',
    }
    for path in sorted(source.glob('*.py')):
        files[prefix+'src/'+path.name]=path
    for path in sorted((experiment/'cluster').iterdir()):
        if path.suffix in ('.sh','.sbatch'):
            files[prefix+'src/cluster/'+path.name]=path
    encoder=experiment/'work/full_encoder/model'
    for path in sorted(encoder.iterdir()):
        if path.is_file():files[prefix+'models/encoder/'+path.name]=path
    owner=experiment/'work/full_owner'
    for name,expected in frozen['model_sha256'].items():
        path=owner/name
        if digest(path)!=expected:raise ValueError(f'Frozen owner model {name} changed')
        files[prefix+'models/owner/'+name]=path
    files[prefix+'models/owner/complete.json']=owner/'complete.json'
    for path in sorted((experiment/'reports').glob('*.json')):
        files[prefix+'reports/'+path.name]=path
    if digest(encoder/'model.safetensors')!='192974ed0c8ed02ee2c48dad6dbf3d7447143b18fe7cd91d2bb8b5a106b931bd':
        raise ValueError('Full encoder checkpoint changed')
    if len(files)!=len(set(files)):raise ValueError('Duplicate archive entry')
    for name,path in files.items():
        if not path.is_file():raise FileNotFoundError(path)
    a.archive.parent.mkdir(parents=True,exist_ok=True)
    expected={name:digest(path) for name,path in files.items()}
    with zipfile.ZipFile(a.archive,'w',allowZip64=True) as archive:
        for name,path in sorted(files.items()):
            compression=(zipfile.ZIP_STORED if path.suffix=='.safetensors'
                         else zipfile.ZIP_DEFLATED)
            archive.write(path,name,compress_type=compression,compresslevel=5)
    with zipfile.ZipFile(a.archive) as archive:
        if set(archive.namelist())!=set(files):raise ValueError('Archive entries differ')
        broken=archive.testzip()
        if broken is not None:raise ValueError(f'Archive CRC failed: {broken}')
        for name in sorted(files):
            if zip_digest(archive,name)!=expected[name]:
                raise ValueError(f'Archive file identity differs: {name}')
    report={'archive':str(a.archive),'archive_sha256':digest(a.archive),
            'entries':len(files),'source1_rows':run['source1_rows'],
            'matching_sha256':run['matching_sha256'],
            'candidate_sha256':run['candidate_sha256'],
            'file_sha256':expected,'archive_crc':'PASS','archive_identities':'PASS'}
    report_path=a.archive.with_suffix('.package_report.json')
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('archive','archive_sha256','entries',
                                            'source1_rows','archive_crc','archive_identities')}))


if __name__=='__main__':main()
