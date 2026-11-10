"""Fast-forward the shared checkout after normal Git publication, preserving local work.

Caller holds RESULT_PUBLICATION.lock. This helper never resets, force-pushes,
applies a stash, deletes source files, or chooses a resolution for unique edits.
"""
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path('/shared/homes/u26487468/streaming_hd')
STATE = Path('/shared/homes/u26487468/agent_uncertainty_quantification/state/streaming_hd/repository_sync')


def sync(repo=REPO, state=STATE):
    repo, state = Path(repo), Path(state)
    state.mkdir(parents=True, exist_ok=True)
    def git(*args, check=True):
        result = subprocess.run(['git', '-C', str(repo), *args], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, timeout=180)
        if check and result.returncode:
            raise RuntimeError(result.stderr.strip())
        return result
    def raw(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], timeout=180)
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    with (state/'sync.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert git('branch', '--show-current').stdout.strip() == 'main', 'Unexpected local branch; preserve it'
        assert not git('diff', '--name-only', '--diff-filter=U').stdout.strip(), 'Unresolved local conflicts; preserve them'
        git('fetch', 'origin', 'main')
        before = git('rev-parse', 'HEAD').stdout.strip()
        target = git('rev-parse', 'origin/main').stdout.strip()
        ahead, behind = map(int, git('rev-list', '--left-right', '--count', 'HEAD...origin/main').stdout.split())
        assert ahead == 0, 'Local commits must be reconciled explicitly; no reset or force push'
        record = dict(before=before, target=target, time=time.time())
        paths = sorted(set(git('diff', '--name-only', 'HEAD').stdout.splitlines() +
                           git('ls-files', '--others', '--exclude-standard').stdout.splitlines()))
        existing = {}
        for name in paths:
            path = repo/name
            assert repo.resolve() in path.resolve().parents, 'Local symlink escapes checkout; preserve it'
            existing[name] = digest(path) if path.is_file() else None
        record['preserved_local_files'] = existing
        if behind:
            incoming = set(git('diff', '--name-only', 'HEAD..origin/main').stdout.splitlines())
            overlaps = sorted(incoming.intersection(paths))
            equal = []
            for name in overlaps:
                path = repo/name
                try:
                    remote_bytes = raw('show', 'origin/main:'+name)
                except subprocess.CalledProcessError:
                    raise RuntimeError('Incoming deletion overlaps local change: '+name)
                if not path.is_file() or path.read_bytes() != remote_bytes:
                    raise RuntimeError('Unique local change overlaps incoming commit; preserved: '+name)
                equal.append(name)
            # Ordinary ff-only preserves non-overlapping unique local changes.
            result = git('merge', '--ff-only', 'origin/main', check=False)
            if result.returncode:
                assert git('rev-parse', 'HEAD').stdout.strip() == before
                assert equal, result.stderr
                assert not git('diff', '--cached', '--name-only').stdout.strip(), 'Preserve staged edits; do not stash them automatically'
                for name in equal:
                    assert digest(repo/name) == existing[name]
                archive = state/('already_published_'+str(time.time_ns())+'.tar')
                with tarfile.open(archive, 'w') as backup:
                    for name in equal:
                        backup.add(repo/name, arcname=name, recursive=False)
                git('stash', 'push', '--include-untracked', '-m',
                    'backup: preserve already-published files before fast-forward', '--', *equal)
                record['backup'] = str(archive)
                record['preserved_stash'] = git('rev-parse', 'refs/stash').stdout.strip()
                git('merge', '--ff-only', 'origin/main')
        after = git('rev-parse', 'HEAD').stdout.strip()
        assert after == target
        for name, expected in existing.items():
            assert (digest(repo/name) if (repo/name).is_file() else None) == expected, 'Local file changed: '+name
        record.update(after=after, ahead_behind=git('rev-list', '--left-right', '--count', 'HEAD...origin/main').stdout.strip(),
                      worktree_status=git('status', '--short').stdout.strip(), status='PASS_FAST_FORWARD_LOCAL_WORK_PRESERVED')
        tmp=state/'LATEST.tmp';tmp.write_text(json.dumps(record, indent=2)+'\n');tmp.replace(state/'LATEST.json')
        return record


if __name__ == '__main__':
    print(json.dumps(sync()))
