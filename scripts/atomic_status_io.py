"""Atomic JSON writes resilient to short-lived Windows reader/AV file locks."""
import json, os, time, uuid

def atomic_json(path,value):
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.'+uuid.uuid4().hex+'.tmp')
    try:
        tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        for attempt in range(40):
            try:
                os.replace(tmp,path)
                return True
            except PermissionError:
                if attempt==39:
                    # Telemetry must never abort training or invalidate a checkpoint.
                    if path.name=='status.json' or path.name.startswith('status_'):
                        print('Status update deferred because file is locked: '+str(path),flush=True)
                        return False
                    raise
                time.sleep(.05)
    finally:
        try:tmp.unlink(missing_ok=True)
        except OSError:pass
