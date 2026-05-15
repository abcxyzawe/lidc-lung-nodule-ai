"""SSH helper wrapping paramiko + password auth.

Used by retrain loop to talk to a remote GPU VPS.
Set credentials via environment variables:
  LIDC_VPS_HOST=<host>
  LIDC_VPS_PORT=<port>
  LIDC_VPS_USER=root
  LIDC_VPS_PASS=<password>          # only needed for first login (key install)
  LIDC_VPS_KEY=~/.ssh/lidc_remote   # private key path

CLI:
  python vps_helper.py exec  "nvidia-smi"
  python vps_helper.py put   local_path  remote_path
  python vps_helper.py get   remote_path local_path
  python vps_helper.py setup-key                   # install ~/.ssh/lidc_remote.pub
"""
import os
import sys
import time
from pathlib import Path

import paramiko

HOST = os.environ.get("LIDC_VPS_HOST", "")
PORT = int(os.environ.get("LIDC_VPS_PORT", "22"))
USER = os.environ.get("LIDC_VPS_USER", "root")
PASS = os.environ.get("LIDC_VPS_PASS", "")
KEY_PATH = Path(os.environ.get("LIDC_VPS_KEY",
                                  str(Path.home() / ".ssh" / "lidc_remote")))


def get_client(use_key=True, timeout=20):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if use_key and KEY_PATH.exists():
        try:
            c.connect(HOST, port=PORT, username=USER,
                      key_filename=str(KEY_PATH), timeout=timeout,
                      look_for_keys=False, allow_agent=False)
            return c, "key"
        except paramiko.AuthenticationException:
            pass
    c.connect(HOST, port=PORT, username=USER, password=PASS,
              timeout=timeout, look_for_keys=False, allow_agent=False)
    return c, "password"


_VPS_ENV_PREFIX = (
    "export LIDC_ROOT=/workspace; "
    "export PATH=/opt/conda/bin:$PATH; "
    "export PYTHONIOENCODING=utf-8; "
    "export PYTHONUNBUFFERED=1; "
)


def run(cmd, timeout=300, with_env=True):
    """Run cmd on VPS. Returns (exit_code, stdout, stderr).

    with_env=True (default) prepends LIDC_ROOT + conda PATH so commands
    behave like an interactive bash login.
    """
    if with_env and not cmd.startswith("export "):
        cmd = _VPS_ENV_PREFIX + cmd
    c, auth = get_client()
    try:
        stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status()
        return rc, out, err
    finally:
        c.close()


def put(local: str, remote: str):
    """Upload via SFTP, fallback to ssh exec stream if SFTP unavailable."""
    try:
        c, _ = get_client()
        sftp = c.open_sftp()
        sftp.put(local, remote)
        sftp.close(); c.close(); return
    except Exception as e:
        try: c.close()
        except: pass
        print(f"  SFTP failed ({e}); fallback to exec stream", file=sys.stderr)
    # Fallback: pipe local file to remote via ssh exec
    c, _ = get_client()
    try:
        with open(local, "rb") as f:
            data = f.read()
        # Use base64 to avoid binary issues over channel
        import base64
        b64 = base64.b64encode(data).decode()
        chunk_size = 32768
        # Truncate target first
        c.exec_command(f"true > {remote}")[0].close()
        for i in range(0, len(b64), chunk_size):
            chunk = b64[i:i+chunk_size]
            cmd = f"printf %s '{chunk}' | base64 -d >> {remote}"
            stdin, stdout, stderr = c.exec_command(cmd, timeout=60)
            stdout.channel.recv_exit_status()
        # Verify
        stdin, stdout, stderr = c.exec_command(f"wc -c < {remote}")
        size = int(stdout.read().decode().strip())
        if size != len(data):
            raise IOError(f"Size mismatch after upload: {size} vs {len(data)}")
        print(f"  Streamed {size} bytes to {remote}")
    finally:
        c.close()


def get(remote: str, local: str):
    try:
        c, _ = get_client()
        sftp = c.open_sftp()
        sftp.get(remote, local)
        sftp.close(); c.close(); return
    except Exception as e:
        try: c.close()
        except: pass
        print(f"  SFTP failed ({e}); fallback to exec stream", file=sys.stderr)
    c, _ = get_client()
    try:
        stdin, stdout, stderr = c.exec_command(f"base64 -w 0 {remote}", timeout=600)
        b64 = stdout.read().decode()
        import base64
        data = base64.b64decode(b64)
        with open(local, "wb") as f:
            f.write(data)
        print(f"  Streamed {len(data)} bytes from {remote}")
    finally:
        c.close()


def setup_key():
    """Install local ~/.ssh/lidc_remote.pub into VPS authorized_keys."""
    pub = KEY_PATH.with_suffix(".pub")
    if not pub.exists():
        sys.exit(f"Public key not found at {pub}")
    pub_text = pub.read_text().strip()
    cmd = (
        f"mkdir -p ~/.ssh && echo {pub_text!r} >> ~/.ssh/authorized_keys && "
        f"chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys && "
        f"sort -u ~/.ssh/authorized_keys -o ~/.ssh/authorized_keys && "
        f"echo INSTALLED"
    )
    # Force password auth for install
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20,
              look_for_keys=False, allow_agent=False)
    try:
        stdin, stdout, stderr = c.exec_command(cmd, timeout=20)
        out = stdout.read().decode(); err = stderr.read().decode()
        if "INSTALLED" in out:
            print(f"Key installed via password auth from {pub}")
        else:
            print(f"Install failed. stdout={out!r} stderr={err!r}")
            sys.exit(1)
    finally:
        c.close()
    # Verify key auth
    try:
        rc, o, e = run("echo KEY_AUTH_OK")
        if "KEY_AUTH_OK" in o:
            print("Key auth verified.")
        else:
            print(f"Key auth still failing: {o!r} {e!r}")
    except paramiko.AuthenticationException as e:
        print(f"Key auth failed: {e}")


def main():
    # Force utf-8 stdout to avoid cp1252 errors with non-ASCII output from VPS
    if hasattr(sys.stdout, "reconfigure"):
        try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass
        try: sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    op = sys.argv[1]
    if op == "exec":
        cmd = " ".join(sys.argv[2:])
        rc, out, err = run(cmd, timeout=600)
        if out: print(out, end="")
        if err: print(err, end="", file=sys.stderr)
        sys.exit(rc)
    elif op == "put":
        put(sys.argv[2], sys.argv[3])
        print(f"Uploaded {sys.argv[2]} -> {sys.argv[3]}")
    elif op == "get":
        get(sys.argv[2], sys.argv[3])
        print(f"Downloaded {sys.argv[2]} -> {sys.argv[3]}")
    elif op == "setup-key":
        setup_key()
    else:
        sys.exit(f"Unknown op: {op}")


if __name__ == "__main__":
    main()
