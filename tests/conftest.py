"""Give numba a compile cache per version of the simulator sources.

numba's on-disk cache does not notice edits to module-level tables it compiled in from
other files (for example nes.engine.GRAVITY used by nes_original.fast), so a test run
could silently use stale machine code. Keying the cache folder on the source contents
forces a recompile whenever any of them changes.
"""
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if 'NUMBA_CACHE_DIR' not in os.environ:
    digest = hashlib.sha1()
    for path in sorted((ROOT/'nes').glob('*.py'))+sorted((ROOT/'nes_original').glob('*.py')):
        digest.update(path.name.encode()+path.read_bytes())
    os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'.numba-cache'/digest.hexdigest()[:16])
