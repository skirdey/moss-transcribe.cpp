"""Install the exact opt48 release checked by the 24-clip CPU benchmark.

Supply private connection settings as arguments. No rebuild is performed: source,
binary, weights and all three shared libraries must match the published evidence.
An existing installation is backed up and restored if startup checks fail.
"""
import argparse
import base64
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE_TREE = "818f4743420e70d7c5407d6758bebc39b579273e"
BINARY_SHA = "488fee1617aa52a5fcba5527a03d545fbc46e6a9457a5413faeac81725f1fb55"
MODEL_SHA = "ed6c35d0d527c5d03171c3eb448e2150a42c76a51e3e73aa821e351c3da8307c"
LIBRARIES = {
    "libggml-base.so": "868cd6fc4a3034371d8781a650830f8e6dd3d1e34369e51928643d89618b5f07",
    "libggml-cpu.so": "bcf5924c28d05aa7e125eb686c522504e46a6294d274472c0accef60c170b102",
    "libggml.so": "edec3dd18cb788dfcbcbaf088bff5032930d5b60fe1139c5a1a156f6612911a4",
}

REMOTE_SCRIPT = r'''
import base64, hashlib, json, os, secrets, shutil, subprocess, time, urllib.request
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()

def run(*args, **kwargs):
    return subprocess.check_output(list(args), text=True, **kwargs).strip()

def health(env):
    req = urllib.request.Request('http://' + env['MOSS_BIND_ADDRESS'] + ':' + env.get('MOSS_API_PORT', '8787') + '/health',
                                 headers={'Authorization': 'Bearer ' + env['MOSS_API_TOKEN']})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=5) as r:
        return json.loads(r.read())

os.umask(0o077)
source, build, model = (Path(cfg[k]).expanduser().resolve() for k in ('source', 'build', 'model'))
service = Path(cfg['service_root']).expanduser().resolve()
config = Path.home() / '.config/moss-api/service.env'
unit = Path.home() / '.config/systemd/user/moss-api.service'
assert run('git', 'rev-parse', 'HEAD^{tree}', cwd=source) == cfg['source_tree'], 'Source tree mismatch'
assert not run('git', 'status', '--porcelain', cwd=source), 'Source must be clean'
assert sha(build / 'moss-transcribe') == cfg['binary_sha'], 'Binary mismatch'
assert sha(model) == cfg['model_sha'], 'Weights mismatch'
libs = {}
for name, expected in cfg['libraries'].items():
    matches = list(build.rglob(name))
    assert len(matches) == 1 and sha(matches[0]) == expected, 'Library mismatch: ' + name
    libs[name] = matches[0]
assert set(range(16)) <= os.sched_getaffinity(0), 'Validated release needs CPUs 0 through 15'

old_env = {}
if config.exists():
    old_env = dict(line.split('=', 1) for line in config.read_text().splitlines() if '=' in line)
    assert not health(old_env)['busy'], 'Worker is busy; wait before upgrading'
token = old_env.get('MOSS_API_TOKEN') or secrets.token_urlsafe(48)
env = dict(old_env)
env.update(MOSS_API_TOKEN=token, MOSS_BIND_ADDRESS=cfg['bind_address'], MOSS_API_PORT=str(cfg['port']),
           MOSS_CPP_MODEL=str(model), MOSS_CPP_EXPECTED_SHA256=cfg['binary_sha'],
           OMP_NUM_THREADS='16', OMP_DYNAMIC='FALSE', OMP_PROC_BIND='spread',
           OMP_PLACES=','.join('{'+str(n)+'}' for n in range(16)))
service.mkdir(parents=True, exist_ok=True)
backup = service / ('backup-' + str(time.time_ns()))
backup.mkdir(mode=0o700)
targets = [config, unit, service / 'moss_api.py', service / 'moss_output.py']
existing = {}
for i, target in enumerate(targets):
    existing[i] = target.exists()
    if existing[i]: shutil.copy2(target, backup / str(i))
was_active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'moss-api.service']).returncode == 0
was_enabled = subprocess.run(['systemctl', '--user', 'is-enabled', '--quiet', 'moss-api.service'], stdout=subprocess.DEVNULL).returncode == 0
release = service / 'releases' / cfg['binary_sha']
(release / 'bin').mkdir(parents=True, exist_ok=True)
(release / 'lib').mkdir(exist_ok=True)
shutil.copy2(build / 'moss-transcribe', release / 'bin/moss-transcribe')
for name, path in libs.items(): shutil.copy2(path, release / 'lib' / name)
env.update(MOSS_CPP_BINARY=str(release / 'bin/moss-transcribe'), LD_LIBRARY_PATH=str(release / 'lib'))
config.parent.mkdir(parents=True, exist_ok=True)
unit.parent.mkdir(parents=True, exist_ok=True)
try:
    for name, contents in cfg['files'].items():
        (service / name).write_bytes(base64.b64decode(contents))
    config.write_text(''.join(k+'='+v+'\n' for k,v in env.items()))
    config.chmod(0o600)
    text = base64.b64decode(cfg['unit']).decode().replace('%h/moss-api', str(service))
    unit.write_text(text)
    if run('loginctl', 'show-user', str(os.getuid()), '--property=Linger', '--value') != 'yes':
        subprocess.run(['loginctl', 'enable-linger', str(os.getuid())], check=True)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', 'moss-api.service'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['systemctl', '--user', 'restart', 'moss-api.service'], check=True)
    for attempt in range(30):
        try:
            h = health(env)
            assert h['status'] == 'ok' and h['identity']['binarySha256'] == cfg['binary_sha']
            assert h['identity']['cpuOpt'] == 48 and h['identity']['threads'] == 16
            break
        except Exception:
            if attempt == 29: raise
            time.sleep(1)
except BaseException:
    for i, target in enumerate(targets):
        if existing[i]: shutil.copy2(backup / str(i), target)
        else: target.unlink(missing_ok=True)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    if not was_enabled: subprocess.run(['systemctl', '--user', 'disable', 'moss-api.service'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['systemctl', '--user', 'restart' if was_active else 'stop', 'moss-api.service'], check=True)
    raise
# Only the caller receives this secret; it is saved locally, never printed there.
print(json.dumps({'token': token, 'identity': h['identity'], 'backup': str(backup)}))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('host', 'source', 'build', 'model', 'bind-address', 'api-url', 'client-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--service-root', default='~/moss-api')
    p.add_argument('--port', type=int, default=8787)
    args = p.parse_args()
    cfg = vars(args) | {'source_tree': SOURCE_TREE, 'binary_sha': BINARY_SHA,
                        'model_sha': MODEL_SHA, 'libraries': LIBRARIES,
                        'files': {name: base64.b64encode((ROOT/name).read_bytes()).decode()
                                  for name in ('moss_api.py', 'moss_output.py')},
                        'unit': base64.b64encode((ROOT/'moss-api.service').read_bytes()).decode()}
    script = 'cfg = json.loads(' + repr(json.dumps(cfg)) + ')\n'
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             args.host, 'python3 -'],
                            input='import json\n' + script + REMOTE_SCRIPT,
                            text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError('Deployment failed; inspect the remote service and preserved backup. ' +
                           result.stderr[-2000:])
    value = json.loads(result.stdout)
    private = Path(args.client_dir)
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    client = private / 'client.env'
    client.write_text('MOSS_BACKEND=remote\nMOSS_API_URL=' + args.api_url +
                      '\nMOSS_API_TOKEN=' + value['token'] +
                      '\nMOSS_API_EXPECTED_BINARY_SHA256=' + BINARY_SHA +
                      '\nMOSS_CHUNK_SECONDS=60\nMOSS_MAX_TOKENS=4096\n')
    client.chmod(0o600)
    print(json.dumps({'installed': True, 'cpuOpt': 48, 'binarySha256': BINARY_SHA}))


if __name__ == '__main__':
    main()
