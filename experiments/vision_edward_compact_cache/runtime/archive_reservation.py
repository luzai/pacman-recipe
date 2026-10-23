"""Reserve actual archive blocks before optimizer work; no sparse-file promises."""
import json,os
from pathlib import Path
GIB=1024**3
RESERVATION='.next-archive-reservation'
def ensure_reservation(archive):
 archive=Path(archive);latest=json.loads((archive/'latest.json').read_text()) if (archive/'latest.json').exists() else {'size':96*GIB}
 need=int(latest['size']*1.005)+GIB
 p=archive/RESERVATION
 if p.exists():
  assert p.is_file() and not p.is_symlink()
  assert p.stat().st_size>=need and p.stat().st_blocks*512>=need-4096
  return p
 fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
 try:os.posix_fallocate(fd,0,need);os.fsync(fd)
 except BaseException:
  os.close(fd);p.unlink();raise
 else:os.close(fd)
 print('ARCHIVE_BLOCKS_RESERVED',need,flush=True)
 return p
