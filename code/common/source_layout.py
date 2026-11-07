"""Explicit source/data path separation for dataset pipeline orchestration."""
from pathlib import Path
import fcntl
import json
import os
import shutil
import subprocess

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = WORKSPACE.parents[1]
RUNS = PROJECT/'experiments/streaming_hd/runs'


def source_path(path):
    """Return a real canonical source path; never create compatibility links."""
    path = Path(path)
    if path.is_relative_to(WORKSPACE):
        return path
    relative = path.relative_to(RUNS)
    name = relative.parts[0]
    family = 'review' if any(value in name for value in ['small_model', 'three_model_review']) else 'dataset'
    return WORKSPACE/'pipelines'/family/relative


def payload_path(data_root, relative):
    relative = Path(relative)
    if relative.parts[0] == 'model_configs':
        return model_metadata_path(data_root)/Path(*relative.parts[1:])
    if relative.parts[0] == 'runtime':
        return source_path(Path(data_root)/'payload'/relative)
    return Path(data_root)/'payload'/relative


def model_metadata_path(data_root):
    descriptor = source_path(Path(data_root)/'payload/MODEL_CONFIG_ASSET.json')
    asset = Path(json.loads(descriptor.read_text())['asset'])
    assert asset.is_relative_to(PROJECT/'assets/runtime_model_metadata')
    return asset


def stage_payload(data_root, destination):
    """Stage question data plus canonical runtime source to the same scratch layout."""
    data_root, destination = Path(data_root), Path(destination)
    data_payload = data_root/'payload'
    def exclude_source(folder, names):
        return ['runtime', 'model_configs'] if Path(folder) == data_payload else []
    shutil.copytree(data_payload, destination, dirs_exist_ok=True, ignore=exclude_source)
    for name in ['runtime', 'model_configs']:
        folder = model_metadata_path(data_root) if name == 'model_configs' else source_path(data_payload/name)
        if folder.exists():
            shutil.copytree(folder, destination/name, dirs_exist_ok=True)


def freeze_generated_source(paths, message):
    """Version only named source/config files before admitting a generated entry."""
    selected=[]
    for value in paths:
        path=Path(value).resolve()
        assert path.is_relative_to(WORKSPACE) and path.is_file() and not path.is_symlink()
        assert path.suffix in {'.py','.pbs','.sh','.json','.md','.toml','.yaml','.yml'}
        assert path.stat().st_size < 10*1024**2, str(path)
        selected.append(str(path.relative_to(PROJECT)))
    assert selected
    with (WORKSPACE/'migration/generated_source_git.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run(['git','-C',str(PROJECT),'add','--',*selected],check=True)
        changed=subprocess.run(['git','-C',str(PROJECT),'diff','--quiet','HEAD','--',*selected]).returncode
        if changed:
            subprocess.run(['git','-C',str(PROJECT),'-c','user.name=Codex','-c','user.email=codex@localhost',
                            'commit','--quiet','--only','-m',message,'--',*selected],check=True)
        return subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip()
