import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from absl import app

from msprior_scripts.train import main

if __name__ == "__main__":
    app.run(main)
