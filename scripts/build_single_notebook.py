import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import re

def clean_code(code_str):
    lines = code_str.split('\n')
    cleaned = []
    for line in lines:
        if line.startswith('from .') or line.startswith('from src.'):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)

def read_file(filepath):
    with open(filepath, 'r') as f:
        return clean_code(f.read())

nb = new_notebook()

nb.cells.append(new_markdown_cell(
    "# 🚀 X-Ray Object Detection — Standalone Colab Notebook\n"
    "This notebook contains the **entire** project consolidated into a single file.\n\n"
    "**Instructions:**\n"
    "1. Go to **Runtime > Change runtime type** and select **T4 GPU**.\n"
    "2. Upload your raw dataset ZIP to `/content/data/raw/` if it's not already there.\n"
    "3. Click **Run All**."
))

nb.cells.append(new_code_cell(
    "!pip install -q ultralytics pyyaml pandas opencv-python matplotlib tqdm\n"
    "import os\n"
    "import sys\n"
    "from pathlib import Path\n"
    "import shutil\n"
    "import glob\n"
    "import time\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import cv2\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from torch.utils.data import Dataset, DataLoader\n"
    "import matplotlib.pyplot as plt\n"
    "from collections import Counter\n"
    "import yaml\n"
    "import random\n"
    "from tqdm.auto import tqdm\n"
))

nb.cells.append(new_code_cell(
    "# --- Google Colab / Local Paths ---\n"
    "IN_COLAB = 'google.colab' in sys.modules\n"
    "if IN_COLAB:\n"
    "    from google.colab import drive\n"
    "    drive.mount('/content/drive')\n"
    "    WORKSPACE = '/content/drive/MyDrive/Object-Detection-in-Luggage-Scanner-for-Security'\n"
    "else:\n"
    "    WORKSPACE = os.path.abspath('.')\n\n"
    "DATA_ROOT = os.path.join(WORKSPACE, 'data')\n"
    "RAW_DIR = os.path.join(DATA_ROOT, 'raw')\n"
    "PROCESSED_DIR = os.path.join(DATA_ROOT, 'processed')\n"
    "CHECKPOINT_DIR = os.path.join(WORKSPACE, 'checkpoints')\n"
    "RESULTS_DIR = os.path.join(WORKSPACE, 'results')\n\n"
    "for d in [RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR]:\n"
    "    os.makedirs(d, exist_ok=True)\n\n"
    "print(f\"Workspace set to: {WORKSPACE}\")"
))

import yaml
with open('configs/default.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

nb.cells.append(new_code_cell(
    "# --- Master Configuration ---\n"
    f"config = {repr(config_dict)}\n"
))

files_to_inject = [
    ("src/utils/config.py", "Configuration Utils"),
    ("src/preprocessing/segmentation.py", "Segmentation & Preprocessing"),
    ("src/preprocessing/augmentation.py", "Data Augmentation"),
    ("src/dataset/property_annotator.py", "Property Annotator"),
    ("src/dataset/splits.py", "Dataset Splits Generator"),
    ("src/dataset/download.py", "Dataset Downloader & Organizer"),
    ("src/dataset/xray_dataset.py", "PyTorch X-Ray Dataset"),
    ("src/model1/architecture.py", "PropertyYOLO Architecture"),
    ("src/model1/loss.py", "Multi-Task Loss"),
    ("src/model1/train.py", "Trainer Logic"),
    ("src/utils/visualization.py", "Visualization Tools"),
]

for filepath, title in files_to_inject:
    if os.path.exists(filepath):
        nb.cells.append(new_markdown_cell(f"### {title}"))
        nb.cells.append(new_code_cell(read_file(filepath)))

nb.cells.append(new_markdown_cell("## 🚀 Execution Phase"))
nb.cells.append(new_code_cell(
    "# 1. Organize Dataset\n"
    "from collections import defaultdict\n\n"
    "print('\\n📦 Checking dataset structure...')\n"
    "sixray_dir = os.path.join(RAW_DIR, 'sixray')\n"
    "img_dir = os.path.join(PROCESSED_DIR, 'images')\n"
    "label_dir = os.path.join(PROCESSED_DIR, 'labels')\n"
    "os.makedirs(img_dir, exist_ok=True)\n"
    "os.makedirs(label_dir, exist_ok=True)\n\n"
    "if os.path.isdir(sixray_dir) and os.listdir(sixray_dir):\n"
    "    is_roboflow = any(os.path.isdir(os.path.join(sixray_dir, split)) for split in ['train', 'valid', 'test'])\n"
    "    if is_roboflow:\n"
    "        print('Found Roboflow pre-split format. Flattening into processed directory...')\n"
    "        for split in ['train', 'valid', 'test', 'val']:\n"
    "            split_dir = os.path.join(sixray_dir, split)\n"
    "            if not os.path.isdir(split_dir): continue\n"
    "            split_img_dir = os.path.join(split_dir, 'images')\n"
    "            if os.path.isdir(split_img_dir):\n"
    "                for f in os.listdir(split_img_dir):\n"
    "                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):\n"
    "                        shutil.copy2(os.path.join(split_img_dir, f), os.path.join(img_dir, f))\n"
    "            split_label_dir = os.path.join(split_dir, 'labels')\n"
    "            if os.path.isdir(split_label_dir):\n"
    "                for f in os.listdir(split_label_dir):\n"
    "                    if f.endswith('.txt'):\n"
    "                        shutil.copy2(os.path.join(split_label_dir, f), os.path.join(label_dir, f))\n"
    "stats = validate_dataset(PROCESSED_DIR)\n"
    "assert stats['total_images'] > 0, 'No images found. Please upload dataset.'\n"
    "print(f'\\u2713 Dataset ready: {stats[\"total_images\"]} images')\n"
))

nb.cells.append(new_code_cell(
    "# 2. Compute Properties\n"
    "annotator = PropertyAnnotator()\n"
    "image_files = sorted(glob.glob(os.path.join(img_dir, '*.*')))\n"
    "label_files = [os.path.join(label_dir, os.path.splitext(os.path.basename(f))[0] + '.txt') for f in image_files]\n"
    "output_csv = os.path.join(DATA_ROOT, 'annotations', 'properties.csv')\n"
    "os.makedirs(os.path.dirname(output_csv), exist_ok=True)\n"
    "if not os.path.exists(output_csv):\n"
    "    print(f'\\U0001f52c Computing properties for {len(image_files)} images...')\n"
    "    props_df = batch_annotate(image_files, label_files, output_csv, annotator)\n"
    "else:\n"
    "    print('Properties already computed.')\n"
))

nb.cells.append(new_code_cell(
    "# 3. Create Stratified Splits\n"
    "split_dir = os.path.join(DATA_ROOT, 'splits')\n"
    "data_yaml = os.path.join(DATA_ROOT, 'data.yaml')\n"
    "train_files, val_files, test_files = create_splits(\n"
    "    img_dir, label_dir, split_dir,\n"
    "    train_ratio=config['dataset']['train_ratio'],\n"
    "    val_ratio=config['dataset']['val_ratio'],\n"
    "    test_ratio=config['dataset']['test_ratio'],\n"
    "    create_symlinks=True\n"
    ")\n"
    "create_yolo_data_yaml(split_dir, config['dataset']['threat_classes'], data_yaml)\n"
))

nb.cells.append(new_code_cell(
    "# 4. Initialize Model & Dataloaders\n"
    "config['data_yaml_path'] = data_yaml\n"
    "property_csv = os.path.join(DATA_ROOT, 'annotations', 'properties.csv')\n"
    "train_dataset = XRayDataset(\n"
    "    image_dir=os.path.join(split_dir, 'train', 'images'),\n"
    "    label_dir=os.path.join(split_dir, 'train', 'labels'),\n"
    "    property_csv=property_csv,\n"
    "    image_size=config['dataset']['image_size']\n"
    ")\n"
    "val_dataset = XRayDataset(\n"
    "    image_dir=os.path.join(split_dir, 'val', 'images'),\n"
    "    label_dir=os.path.join(split_dir, 'val', 'labels'),\n"
    "    property_csv=property_csv,\n"
    "    image_size=config['dataset']['image_size']\n"
    ")\n"
    "train_loader = DataLoader(train_dataset, batch_size=config['model1']['stage2']['batch_size'], shuffle=True, collate_fn=collate_fn)\n"
    "val_loader = DataLoader(val_dataset, batch_size=config['model1']['stage2']['batch_size'], shuffle=False, collate_fn=collate_fn)\n\n"
    "device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')\n"
    "model = PropertyYOLO(\n"
    "    model_size=config['model1']['backbone'],\n"
    "    num_classes=len(config['dataset']['threat_classes']) + 1,\n"
    "    num_properties=config['model1']['property_head']['num_outputs'],\n"
    "    input_channels=config['model1']['input_channels'],\n"
    "    pretrained=config['model1']['pretrained']\n"
    ")\n"
    "trainer = Trainer(config=config, model=model, device=device, checkpoint_dir=CHECKPOINT_DIR)\n"
))

nb.cells.append(new_code_cell(
    "# 5. Train Stage 1 (Detection)\n"
    "print('\\nStarting Stage 1: Detection Training...')\n"
    "stage1_results = trainer.train_stage1(train_loader=train_loader, val_loader=val_loader)\n"
))

nb.cells.append(new_code_cell(
    "# 6. Train Stage 2 (Property Regression)\n"
    "print('\\nStarting Stage 2: Property & Material Training...')\n"
    "stage2_results = trainer.train_stage2(train_loader=train_loader, val_loader=val_loader)\n"
))

with open('notebooks/standalone_colab.ipynb', 'w') as f:
    nbformat.write(nb, f)

print("✅ Successfully generated notebooks/standalone_colab.ipynb!")
