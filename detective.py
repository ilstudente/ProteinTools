"""Direct checkout entrypoint, including environments that suppress editable .pth files."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from protein_detective.cli import main

if __name__=='__main__':
    main()
