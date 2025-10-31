#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rllib_marl.train import main

if __name__ == "__main__":
    main()