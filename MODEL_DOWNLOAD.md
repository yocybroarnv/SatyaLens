# Model Download Instructions

The SatyaLens project requires the pretrained model weights file to be placed in the root directory.

## Model Details
- **Filename**: `satyalens_v6_efficientnetb0.keras`
- **Size**: ~29 MB (29,062,952 bytes)
- **Model Type**: EfficientNetB0 Transfer Learning (TensorFlow/Keras)
- **Classes**: `0` -> Real, `1` -> Fake

## Setup Instructions

If the model is missing, please ensure that:
1. **Git LFS (Large File Storage)** is installed and initialized if checking out this repository from Git:
   ```bash
   git lfs install
   git lfs pull
   ```
2. Alternatively, place the model file manually in the root folder of the project alongside `app.py`:
   ```text
   SatyaLens/
   ├── app.py
   ├── satyalens_v6_efficientnetb0.keras  <-- Place here
   └── requirements.txt
   ```
3. Once placed, the sidebar status in the app will update to **Model Loaded (Ready)**.
