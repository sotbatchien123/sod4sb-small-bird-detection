"""Small deterministic replay recipes instead of retaining normalized images."""
import random
import numpy as np
import torch

def capture_rng():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None)

def restore_rng(state):
    random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
    if state[3] is not None:torch.cuda.set_rng_state(state[3])

def replay_recipe(recipe, run):
    current=capture_rng()
    try:
        index,rng=recipe
        restore_rng(rng)
        return run(index)
    finally:restore_rng(current)

def memory_snapshot():
    import ctypes
    from ctypes import wintypes
    class Memory(ctypes.Structure):
        _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(n,ctypes.c_ulonglong) for n in (
            'total_physical','available_physical','commit_limit','available_commit',
            'total_virtual','available_virtual','available_extended_virtual')]
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in (
            'PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
            'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
    mem=Memory();mem.length=ctypes.sizeof(mem)
    kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=wintypes.HANDLE
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    query=ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    result={}
    if kernel.GlobalMemoryStatusEx(ctypes.byref(mem)):
        result.update(ram_available_gib=round(mem.available_physical/2**30,3),
            commit_available_gib=round(mem.available_commit/2**30,3))
    if query(kernel.GetCurrentProcess(),ctypes.byref(counters),ctypes.sizeof(counters)):
        result.update(process_working_gib=round(counters.WorkingSetSize/2**30,3),
            process_private_gib=round(counters.PrivateUsage/2**30,3))
    return result
