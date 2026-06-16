import os
import runpy
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent
    model_dir = project_root / "difm2sr"
    os.chdir(model_dir)
    sys.path.insert(0, str(model_dir))
    runpy.run_path(str(model_dir / "run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
