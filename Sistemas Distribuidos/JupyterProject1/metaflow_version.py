import subprocess
from subprocess import CalledProcessError

def find_git_on_windows():
    try:
        subprocess.check_output(["where", "/Q", "git"])
    except CalledProcessError:
        return None
    return "git"