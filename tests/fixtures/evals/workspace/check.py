from pathlib import Path
import sys


answer = Path('answer.txt')
if not answer.is_file() or answer.read_bytes() != Path('expected.txt').read_bytes():
    print('answer does not match expected input', file=sys.stderr)
    sys.exit(1)
