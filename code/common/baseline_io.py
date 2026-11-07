"""Synchronous immutable input staging and committed-output mirroring."""
import os
from pathlib import Path
import shutil


def mirror_file(path):
    scratch, durable = os.environ.get('BASELINE_SCRATCH_OUTPUT'), os.environ.get('BASELINE_DURABLE_OUTPUT')
    if not scratch or not durable:
        return
    path=Path(path).resolve(); root=Path(scratch).resolve()
    if not path.is_relative_to(root):
        return
    target=Path(durable)/path.relative_to(root)
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix(target.suffix+'.sync.tmp')
    shutil.copy2(path,temp)
    temp.replace(target)


def stage_immutable(path, expected_sha256):
    root=os.environ.get('BASELINE_STAGE_ROOT')
    if not root:
        return path
    from common.baseline_data import file_sha, NotReady
    target=Path(root)/expected_sha256/path.name
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix(target.suffix+'.tmp')
        shutil.copyfile(path,temp)
        if file_sha(temp)!=expected_sha256:
            raise NotReady('STAGING_INPUT_HASH_MISMATCH: '+str(path))
        temp.replace(target)
    if file_sha(target)!=expected_sha256:
        raise NotReady('STAGED_INPUT_HASH_MISMATCH: '+str(target))
    return target
