"""Install the private MOSS API on stan@hp-fury using existing benchmark assets.

The token is stored in ignored, mode-600 files and is never printed. Deployment
does not reboot the shared host. User lingering and the enabled systemd unit
provide boot startup; Restart=always provides crash recovery.
"""
from pathlib import Path
import json
import secrets
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = "stan@hp-fury"


def main():
    private = ROOT / "data/moss-api"
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    token_path = private / "token"
    if not token_path.exists():
        token_path.write_text(secrets.token_urlsafe(48))
    token_path.chmod(0o600)
    token = token_path.read_text().strip()
    client = private / "client.env"
    client.write_text(f"MOSS_BACKEND=remote\nMOSS_API_URL=http://hp-fury:8787\nMOSS_API_TOKEN={token}\nMOSS_CHUNK_SECONDS=60\nMOSS_MAX_TOKENS=4096\n")
    client.chmod(0o600)
    subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE,
                    "mkdir -p ~/hw-moss-api/bin ~/hw-moss-api/lib ~/.config/moss-api ~/.config/systemd/user"], check=True)
    for source, destination in (
        (ROOT / "moss_api.py", "hw-moss-api/moss_api.py"),
        (ROOT / "moss_output.py", "hw-moss-api/moss_output.py"),
        (ROOT / "moss-api.service", ".config/systemd/user/moss-api.service"),
        (ROOT / "hp_fury_moss_bench.py", "hw-moss-api/cpu_patch.py"),
    ):
        subprocess.run(["scp", str(source), f"{REMOTE}:{destination}"], check=True)
    # Feed the secret through stdin, avoiding command-line arguments and output.
    script = '''
from pathlib import Path
import os, shutil, subprocess
home = Path.home()
source = home / 'hw-audio-bench/moss-transcribe.cpp'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip() == '190a569c13b4b247450f2fb3b2a431244e84833e'
service = home / 'hw-moss-api'
subprocess.run(['python3',str(service / 'cpu_patch.py'),'--root',str(home / 'hw-audio-bench'),'--instrument','--recovery'],check=True)
with (service / 'build.log').open('w') as log:
    subprocess.run(['cmake','--build','build','-j','8'],cwd=source,stdout=log,stderr=log,check=True)
shutil.copy2(home / 'hw-audio-bench/results/instrumentation.patch',service / 'inference.patch')
shutil.copy2(source / 'build/moss-transcribe', service / 'bin/moss-transcribe')
for lib in (source / 'build/third_party/ggml/src').glob('libggml*.so*'):
    shutil.copy2(lib, service / 'lib' / lib.name)
model = service / 'moss-transcribe-q8_0.gguf'
if model.is_symlink():
    model.unlink()
if not model.exists():
    os.link(home / 'hw-audio-bench/models/moss-transcribe-q8_0.gguf', model)
address = subprocess.check_output(['tailscale','ip','-4'],text=True).strip()
env = 'MOSS_API_TOKEN=' + TOKEN + '\\nMOSS_BIND_ADDRESS=' + address + '\\nMOSS_API_PORT=8787\\nMOSS_CPP_BINARY=' + str(service / 'bin/moss-transcribe') + '\\nMOSS_CPP_MODEL=' + str(model) + '\\nLD_LIBRARY_PATH=' + str(service / 'lib') + '\\n'
config = home / '.config/moss-api/service.env'
config.write_text(env)
config.chmod(0o600)
subprocess.run(['loginctl','enable-linger',home.name],check=True)
subprocess.run(['systemctl','--user','daemon-reload'],check=True)
subprocess.run(['systemctl','--user','enable','moss-api.service'],check=True)
subprocess.run(['systemctl','--user','restart','moss-api.service'],check=True)
print('MOSS API installed; user boot persistence enabled')
'''
    subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE, "python3 -"],
                   input="TOKEN = " + json.dumps(token) + "\n" + script, text=True, check=True)


if __name__ == "__main__":
    main()
