# train_segmentation_debug.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from configs.vton_config_default import config
from train_segmentation import main

# override config buat debug
config["training"]["epochs"] = 1       # cuma 1 epoch
config["training"]["batch_size"] = 1   # batch kecil biar cepat
config["training"]["save_interval"] = 1

if __name__ == "__main__":
    main()
