import json
import os

NOTEBOOK_PATH = "notebooks/03_model1_training.ipynb"

def update_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: Could not find {NOTEBOOK_PATH}")
        return

    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # 1. Markdown cell
    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Stage 2: Property Head Training\n",
            "**Objective:** Freeze the backbone and train the custom 11-dimensional Property Regression Head using the Stage 1 weights."
        ]
    }

    # 2. Code cell
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "from torch.utils.data import DataLoader\n",
            "from src.model1.architecture import PropertyYOLO\n",
            "from src.model1.loss import MultiTaskLoss\n",
            "from src.model1.train import Trainer\n",
            "from src.dataset.xray_dataset import XRayDataset, collate_fn\n",
            "\n",
            "# 1. Instantiate the PyTorch Dataset/DataLoader\n",
            "train_dataset = XRayDataset(\n",
            "    image_dir='/content/dataset/raw',\n",
            "    label_dir='/content/dataset/raw',\n",
            "    property_csv='/content/dataset/raw/properties.csv',\n",
            "    image_size=640\n",
            ")\n",
            "train_loader = DataLoader(\n",
            "    train_dataset, \n",
            "    batch_size=16, \n",
            "    shuffle=True, \n",
            "    num_workers=2, \n",
            "    collate_fn=collate_fn\n",
            ")\n",
            "\n",
            "# 2. Initialize the model with Stage 1 weights\n",
            "model = PropertyYOLO(\n",
            "    model_size='yolov8m',\n",
            "    num_classes=6, # 5 threats + 1 background\n",
            "    num_properties=11,\n",
            "    weights_path='weights/yolov8_xray_stage1_final.pt'\n",
            ")\n",
            "\n",
            "# 3. Freeze backbone\n",
            "model.set_training_stage(2)\n",
            "\n",
            "# 4. Trigger the 50-epoch training run\n",
            "config = {'model1': {'stage2': {'epochs': 50, 'learning_rate': 0.001}}}\n",
            "trainer = Trainer(\n",
            "    config=config,\n",
            "    model=model,\n",
            "    device='cuda' if torch.cuda.is_available() else 'cpu'\n",
            ")\n",
            "trainer.train_stage2(train_loader, epochs=50)"
        ]
    }

    # Append cells
    notebook["cells"].extend([markdown_cell, code_cell])

    # Save notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
        # Ensure trailing newline for JSON file
        f.write("\n")
        
    print(f"Successfully appended Stage 2 cells to {NOTEBOOK_PATH}!")

if __name__ == '__main__':
    update_notebook()
