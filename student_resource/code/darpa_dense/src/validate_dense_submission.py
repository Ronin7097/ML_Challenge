"""Run streaming and official ID validators on the complete final TSVs."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def run(args):
    completed=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,check=False)
    print(completed.stdout,flush=True)
    if completed.returncode!=0 or 'PASS' not in completed.stdout:
        raise RuntimeError(f'Validator failed: {args[1]} (exit {completed.returncode})')
    return completed.stdout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--student-resource',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    a=p.parse_args();student=a.student_resource.resolve();output=a.output.resolve()
    matching=output/'matching_results.tsv';candidate=output/'candidate_pairs.tsv'
    manifest=json.loads((output/'run.json').read_text())
    if not manifest['complete_submission'] or manifest['source1_rows']!=1732544:
        raise ValueError('Run manifest does not certify complete test inference')
    hashes={'matching_sha256':sha(matching),'candidate_sha256':sha(candidate)}
    if any(hashes[key]!=manifest[key] for key in hashes):
        raise ValueError('TSV hash differs from run manifest')
    streaming=run([sys.executable,str(student/'code/business_entity_resolution/src/validate_outputs.py'),
                   '--source1',str(student/'dataset/test/test_source1.tsv'),
                   '--matching',str(matching),'--candidate',str(candidate)])
    match=re.search(r'PASS: ([\d,]+) Source 1 rows; ([\d,]+) matches; ([\d,]+) candidates',streaming)
    if match is None or int(match.group(1).replace(',',''))!=1732544:
        raise ValueError('Streaming validator row count differs')
    # The official candidate-ID mode materializes huge lists. The streaming
    # validator above already proves candidate/match consistency on every row.
    official=run([sys.executable,str(student/'utils/validate_submission.py'),
                  '--matching',str(matching),'--test-dir',str(student/'dataset/test'),
                  '--check-ids'])
    if 'skipping the' in official.lower() and 'id' in official.lower():
        raise ValueError('Official target-ID check was skipped')
    result={'streaming':'PASS','official_match_ids':'PASS',
            'source1_rows':1732544,'matches':int(match.group(2).replace(',','')),
            'candidate_pairs':int(match.group(3).replace(',','')),
            **hashes,'run_sha256':sha(output/'run.json'),
            'streaming_output':streaming.strip(),'official_output':official.strip()}
    a.validation.parent.mkdir(parents=True,exist_ok=True)
    if a.validation.exists():raise ValueError('Refuse to replace validation manifest')
    a.validation.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('source1_rows','matches','candidate_pairs',
                                           'streaming','official_match_ids')}))


if __name__=='__main__':main()
