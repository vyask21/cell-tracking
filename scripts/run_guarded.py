r"""Run a local job behind a wake lock and a memory floor.

Two separate failures on this machine have looked identical from a remote
session, and both read as "the PC fell asleep".

The first is real sleep. It is not the idle timer, which is already set to never
on both AC and DC, but anything else that can put the machine into a low power
state while a long run has no keyboard input behind it. A process that wants to
survive that has to say so, because Windows does not count CPU load as activity.
`SetThreadExecutionState` is how it says so.

The second is the one that actually happened, twice on 2026-09-16, and it is a
hang rather than a sleep. Kernel-Power event 41 both times. A single ILP solve on
the largest graphs in this set peaks above 2 GB, this machine has 15.7 GB, and
the C: pagefile is system managed, which means it grows on demand and a solve
allocates faster than it grows. When commit runs out there is nowhere to page to,
the machine thrashes and then stops responding, and a run that was going to
finish in 20 minutes instead costs a reboot and whatever was in flight.

The floor is the defence. Available physical memory is sampled every few seconds,
and if it stays under the floor for long enough to be a trend rather than a spike
the child process tree is killed. A killed screen is cheap here: the solve cache
writes to a temporary name and atomically replaces, so progress survives a kill
intact, measured across three attempts on 2026-09-15 at 1 then 4 then 8 solves of
19 with no corrupt files. A wedged desktop is not cheap.

    .venv\Scripts\python.exe scripts/run_guarded.py -- ^
        .venv\Scripts\python.exe scripts/screen_calibration.py --arms base,foo

Everything after the bare `--` is the command, passed through untouched. The exit
code is the child's, except that a memory kill exits 137 so a wrapper script can
tell it apart from the job's own failure.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time
from ctypes import wintypes

# ES_CONTINUOUS keeps the state set until it is cleared rather than resetting it
# after one idle check. ES_SYSTEM_REQUIRED is the system sleep lock. The display
# is deliberately not held: keeping the screen awake for a 70 minute run is rude
# and buys nothing, since the remote session needs the system up and not the
# panel lit.
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def memory() -> tuple[float, float]:
    """Available physical memory and available commit, both in GB.

    Commit is the one that decides whether an allocation fails, and it is the
    column that was missing from every diagnosis of this before. Physical can
    look comfortable while commit is nearly gone, because a system managed
    pagefile reports what it has grown to and not what it could grow to.
    """
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(st)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        raise ctypes.WinError()
    return st.ullAvailPhys / 1024 ** 3, st.ullAvailPageFile / 1024 ** 3


def set_wake_lock(on: bool) -> bool:
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED if on else ES_CONTINUOUS
    return ctypes.windll.kernel32.SetThreadExecutionState(flags) != 0


def resolve(program: str) -> str:
    """Make the child's program name something CreateProcess will accept.

    CreateProcess does not take a relative path written with forward slashes,
    and every usage line in this repo writes one, because the shell here is
    sometimes bash. `.venv/Scripts/python.exe` fails with WinError 2 while the
    same path with backslashes works. Normalising is enough for a path that
    exists; anything else goes to `which`, and if that finds nothing the
    original string is passed through so the error names what was asked for.
    """
    if os.path.exists(program):
        return os.path.abspath(program)
    found = shutil.which(program)
    return found if found else program


def kill_tree(pid: int) -> None:
    # The screen spawns a worker pool, so killing the parent leaves the solves
    # running and the memory held. taskkill /T is the only reliable way to get
    # the whole tree on Windows without a third party dependency.
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                   capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-gb", type=float, default=1.5,
                    help="kill the job when available physical memory sits "
                         "below this. 1.5 is about where this machine stopped "
                         "responding rather than where it started to swap.")
    ap.add_argument("--commit-floor-gb", type=float, default=1.0,
                    help="same, for available commit. A solve that cannot get "
                         "commit fails its allocation, which is usually where "
                         "the wedge begins.")
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--breaches", type=int, default=6,
                    help="consecutive polls under a floor before killing. Six "
                         "at five seconds is half a minute, long enough that a "
                         "single large allocation does not trip it.")
    ap.add_argument("--no-wake-lock", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given. Put it after a bare -- .")

    locked = False
    if not args.no_wake_lock:
        locked = set_wake_lock(True)
        if not locked:
            print("run_guarded: WAKE LOCK REFUSED, the run is not protected "
                  "from sleep", flush=True)

    phys, commit = memory()
    print(f"run_guarded: start, avail phys {phys:.1f} GB, avail commit "
          f"{commit:.1f} GB, wake lock {'held' if locked else 'off'}",
          flush=True)
    print(f"run_guarded: floors phys {args.floor_gb} GB commit "
          f"{args.commit_floor_gb} GB, {args.breaches} consecutive polls "
          f"at {args.poll_s}s", flush=True)

    t0 = time.time()
    low_phys = phys
    low_commit = commit
    breaches = 0
    killed = False

    proc = None
    try:
        proc = subprocess.Popen([resolve(cmd[0])] + cmd[1:])
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(args.poll_s)
            phys, commit = memory()
            low_phys = min(low_phys, phys)
            low_commit = min(low_commit, commit)
            if phys < args.floor_gb or commit < args.commit_floor_gb:
                breaches += 1
                print(f"run_guarded: low memory {breaches}/{args.breaches}, "
                      f"phys {phys:.2f} GB commit {commit:.2f} GB", flush=True)
                if breaches >= args.breaches:
                    print("run_guarded: KILLING THE JOB. The solve cache is "
                          "atomic, so finished solves survive. Re-run to "
                          "continue from them.", flush=True)
                    kill_tree(proc.pid)
                    killed = True
                    proc.wait()
                    break
            else:
                breaches = 0
    except OSError as exc:
        print(f"run_guarded: could not start {cmd[0]!r}: {exc}", flush=True)
    except KeyboardInterrupt:
        if proc is not None:
            kill_tree(proc.pid)
            proc.wait()
        print("run_guarded: interrupted, child tree killed", flush=True)
    finally:
        if locked:
            set_wake_lock(False)
    if proc is None:
        return 1

    mins = (time.time() - t0) / 60.0
    print(f"run_guarded: done in {mins:.1f} min, low water phys "
          f"{low_phys:.2f} GB, commit {low_commit:.2f} GB", flush=True)
    if killed:
        return 137
    return proc.returncode if proc.returncode is not None else 0


if __name__ == "__main__":
    sys.exit(main())
