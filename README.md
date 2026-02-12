# Pothole Detection Model

A deep learning model to detect potholes in images using transfer learning with PyTorch.

## Overview

This project uses a convolutional neural network (CNN) based on pre-trained models (ResNet18, ResNet50, or EfficientNet) to classify images as containing potholes or not.

## Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision
- PIL/Pillow
- numpy
- scikit-learn
- matplotlib
- tqdm

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

**Important**: This is a binary classification task, so you need both:
- **Pothole images** (you already have these in `Pothole_img/`)
- **Non-pothole images** (road images without potholes)

### Step 1: Organize Your Data

Run the data preparation script to split your pothole images into train/validation/test sets:

```bash
python prepare_data.py
```

This will create a `dataset/` folder with the following structure:
```
dataset/
├── train/
│   ├── pothole/      (contains your pothole images)
│   └── no_pothole/   (you need to add non-pothole images here)
├── val/
│   ├── pothole/
│   └── no_pothole/   (you need to add non-pothole images here)
└── test/
    ├── pothole/
    └── no_pothole/   (you need to add non-pothole images here)
```

### Step 2: Add Non-Pothole Images

You need to add non-pothole road images to balance your dataset. You can:

1. **Download from the internet**: Search for road images without potholes
2. **Collect manually**: Take photos of roads without potholes
3. **Use existing datasets**: 
   - [Roboflow Universe](https://universe.roboflow.com/) - search for road/no-pothole images
   - [Kaggle Datasets](https://www.kaggle.com/datasets) - road classification datasets
   - [ImageNet](https://www.image-net.org/) - search for road/street categories

For best results, try to match the number of non-pothole images with pothole images in each split.

**Recommended structure**:
- Train: ~70% of data
- Validation: ~15% of data  
- Test: ~15% of data

## Training

Once you have both classes of images organized, train the model:

```bash
python train.py
```

### Training Parameters

You can modify the training parameters in `train.py`:

- `data_dir`: Dataset directory (default: "dataset")
- `model_name`: Model architecture - 'resnet18', 'resnet50', or 'efficientnet' (default: 'resnet18')
- `num_epochs`: Number of training epochs (default: 20)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 0.001)
- `img_size`: Image size for input (default: 224)

The best model will be saved as `best_pothole_model.pth` based on validation accuracy.

## Prediction

### Single Image Prediction

To check if a single image contains a pothole:

```bash
python predict.py path/to/image.jpg
```

You can also specify a custom model path:

```bash
python predict.py path/to/image.jpg path/to/model.pth
```

### Batch Prediction

To predict on multiple images in a directory, modify `predict.py` or use it in a Python script:

```python
from predict import predict_batch

results = predict_batch("path/to/images/", "best_pothole_model.pth")
for result in results:
    if result['success']:
        print(f"{result['image_path']}: {result['has_pothole']} (confidence: {result['confidence']:.2%})")
```

### Output

The prediction script outputs:
- **Has Pothole**: YES or NO
- **Predicted Class**: 'pothole' or 'no_pothole'
- **Confidence**: Overall prediction confidence
- **Pothole Probability**: Probability that the image contains a pothole
- **No Pothole Probability**: Probability that the image does not contain a pothole

## Model Architecture

The model uses transfer learning with pre-trained ImageNet weights:

- **ResNet18**: Fast and lightweight, good for quick experiments
- **ResNet50**: Deeper network, potentially better accuracy
- **EfficientNet**: Modern architecture, good balance of speed and accuracy

The final classification layer is replaced to output 2 classes: pothole vs no_pothole.

## Tips for Better Results

1. **Balanced Dataset**: Ensure roughly equal numbers of pothole and non-pothole images
2. **Data Quality**: Use clear, high-resolution images
3. **Augmentation**: The training script includes data augmentation (already enabled)
4. **More Epochs**: Increase `num_epochs` if the model is still improving
5. **Hyperparameter Tuning**: Experiment with learning rates and batch sizes
6. **Larger Model**: Try ResNet50 or EfficientNet for potentially better accuracy

## Troubleshooting

**Error: "no_pothole class not found"**
- You need to add non-pothole images to your dataset. See "Step 2: Add Non-Pothole Images" above.

**Low Accuracy**
- Ensure your dataset is balanced
- Check that images are clear and properly labeled
- Try training for more epochs
- Consider using a larger model (ResNet50 or EfficientNet)

**CUDA/GPU Issues**
- The model will automatically use CPU if CUDA is not available
- For faster training, ensure PyTorch with CUDA is installed

## File Structure

```
pothole_check/
├── Pothole_img/          # Your original pothole images
├── dataset/              # Organized train/val/test splits (created by prepare_data.py)
├── prepare_data.py       # Script to organize images into train/val/test
├── train.py              # Training script
├── predict.py            # Inference/prediction script
├── best_pothole_model.pth  # Saved trained model (created after training)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## License

This project is open source and available for use.

## Hosting on Render (FastAPI)

This repo already includes a FastAPI service in `serve.py` with:
- `GET /health`
- `POST /predict` (multipart file upload)

### Option A (recommended): deploy with Render Blueprint

1. Push this repo to GitHub.
2. In Render, click **New** → **Blueprint** and select your repo.
    - Render will read `render.yaml`.
3. Set environment variable `POTHOLE_MODEL_URL` in Render.
    - This should be a direct-download link to your `best_pothole_model.pth` (S3/GCS/Azure Blob, etc.).
    - The API will download the model at startup if it’s not present on disk.
4. Deploy.

#### Using Google Drive for the model file

1. Upload `best_pothole_model.pth` to Google Drive.
2. Right click → **Share** → set to **Anyone with the link** (Viewer).
3. Copy the share link and either:
     - Paste the share link directly into `POTHOLE_MODEL_URL` (supported), OR
     - Convert it to a direct link using the file id:
         - `https://drive.google.com/uc?export=download&id=<FILE_ID>`

Note: Google Drive sometimes shows a “file is too large to scan” confirmation page. This project supports that flow on Render by using `gdown` for Drive links.

The service runs with:
- Build: `pip install -r requirements.render.txt`
- Start: `uvicorn serve:app --host 0.0.0.0 --port $PORT`

### Option B: manual Render Web Service setup

1. Push this repo to GitHub.
2. In Render, click **New** → **Web Service** → connect your repo.
3. Settings:
    - Environment: **Python**
    - Build command: `pip install -r requirements.render.txt`
    - Start command: `uvicorn serve:app --host 0.0.0.0 --port $PORT`
4. Add env vars (Render Dashboard → Environment):
    - `POTHOLE_MODEL_URL` = direct URL to `best_pothole_model.pth`
    - Optional: `POTHOLE_THRESHOLD` = e.g. `0.5`

### Testing your deployed API

After deploy, your base URL will look like:
`https://<your-service-name>.onrender.com`

Health check:
```bash
curl https://<your-service-name>.onrender.com/health
```

Predict (upload an image):
```bash
curl -X POST https://<your-service-name>.onrender.com/predict \
  -F "file=@test_img/your_image.jpg"
```

OpenAPI docs:
- `https://<your-service-name>.onrender.com/docs`

### Notes

- If you commit `best_pothole_model.pth` into the repo, you can skip `POTHOLE_MODEL_URL`. (Large model files often can’t be pushed to GitHub without LFS.)
- Recommended for large models: host the `.pth` file somewhere downloadable and set `POTHOLE_MODEL_URL`.
- This repo’s default [render.yaml](render.yaml) is free-plan friendly and does not use a persistent disk.
- Optional (paid plans): add a Render disk and set `POTHOLE_MODEL_PATH=/var/data/best_pothole_model.pth` to avoid re-downloading the model after restarts.
- Free Render instances can sleep when idle; first request after idle may be slow.
