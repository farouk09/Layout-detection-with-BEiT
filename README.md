
# CV Parsing with BEIT-based Layout Detection


## Overview

Welcome to the CV Parsing project! This project focuses on Computer Vision (CV) parsing and leverages the BEIT model for layout detection. The model consists of three key components:

1. **Backbone**: We employ BEIT (BERT for Image Transformers) as the backbone for pretraining image transformers.
2. **Neck**: Our model incorporates a Feature Pyramid Network (FPN) for improved feature extraction.
3. **Head**: We use Faster R-CNN for object detection and recognition.

## Model Pretraining

To achieve accurate layout detection, our backbone (BEIT) is pretrained on a self-supervised task based on masked image modeling. For detailed information on the pretraining process, please refer to the [pretraining readme](link-to-pretraining-readme).

## Prerequisites

Before you start working with this project, ensure you have the following prerequisites in place:

- **MMdetection 3.1.0**: Make sure you have MMdetection version 3.1.0 installed.
- **BEIT Backbone Integration**: Move the file `layout detection/backbone/beit.py` to `mmdetection/mmdet/models/backbones` within your MMdetection installation. Additionally, import BEIT in `mmdetection/mmdet/models/backbones/__init__.py`.

The CV dataset used in this project is stored on SharePoint. To access it, you'll need to insert the dataset path into `layout detection/configs/_base_/datasets/CV_dataset.py`.

## Fine-Tuning

To fine-tune the model, you can use the following command:

```
python tools/train.py <config> --resume-from <last_checkpoint>
```

## Testing

To test the model, you can use the following command:

```
python tools/test.py <config> <checkpoint> --show-dir <directory_results>
```

## User Interface

For user interface examples and inference, please refer to the Gradio UI notebook included in this repository. You'll find examples of how to interact with the model through the Gradio user interface.

## Extraction of Small Information

If you need to extract small pieces of information, such as names and company details, we've provided a notebook where we've implemented Pix2Struct to perform this task.

