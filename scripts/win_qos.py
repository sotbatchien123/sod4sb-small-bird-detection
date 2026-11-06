"""Keep a hidden/background trainer out of Windows EcoQoS.

Windows 11 treats processes without a foreground window (e.g. started by the queue watcher) as
background work and may schedule their threads on the E-cores of hybrid CPUs. On the i7-14650HX this
halved YOLO training speed (image decode threads on E-cores, GPU starved: ~10 vs ~18 img/s).
Opting out of execution-speed throttling restores P-core scheduling. No effect off Windows.
Used through run_qos.py so the hashed trainer sources stay unchanged.
"""
import sys


def disable_power_throttling():
    if sys.platform != 'win32': return False
    import ctypes
    from ctypes import wintypes
    class State(ctypes.Structure):
        _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG), ('StateMask', wintypes.ULONG)]
    # ProcessPowerThrottling = 4; control EXECUTION_SPEED (1) with state 0 = never throttle.
    state = State(1, 0x1, 0x0)
    kernel32 = ctypes.WinDLL('kernel32')
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    kernel32.SetProcessInformation.restype = wintypes.BOOL
    return bool(kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state)))
