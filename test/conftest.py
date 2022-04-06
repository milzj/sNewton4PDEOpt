from pathlib import Path
import sys

here = Path(__file__).parent
sys.path.insert(0, str(here.parent) + "/problem")
sys.path.insert(0, str(here.parent) + "/problem/prox")
sys.path.insert(0, str(here.parent) + "/algorithm")
