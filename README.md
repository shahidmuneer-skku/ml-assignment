<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
</head>

<body>

<h1>Residual Supervised Deepfake Detection and Localization</h1>

<h3>Final Presentation &mdash; Group 1</h3>

<hr>

<h2>1. Project Description</h2>

<p>
This repository contains the implementation for the course assignment titled
<strong>“Residual Supervised Deepfake Detection and Localization.”</strong>
The objective of this project is to detect manipulated facial images and
localize forged regions using a residual-supervised deep learning approach.
</p>

<p>
With the rapid advancement of generative models, deepfake images and videos have
become increasingly realistic and difficult to detect. This project explores
the use of residual information and transformer-based vision models to improve
both detection accuracy and interpretability through localization.
</p>

<hr>

<h2>2. Objectives</h2>

<ul>
  <li>Perform deepfake classification (Real vs Fake).</li>
  <li>Localize manipulated regions within facial images.</li>
  <li>Leverage residual supervision for improved interpretability.</li>
  <li>Analyze model behavior through visualization.</li>
</ul>

<hr>

<h2>3. Repository Structure</h2>

<pre>
ML_assignment/
├── CLIP/                   # CLIP-based components
├── models/                 # Model architectures and checkpoints
├── vit/                    # Vision Transformer modules
├── utils/                  # Utility and preprocessing functions
├── temp/                   # Temporary outputs
│
├── inference_image.py      # Image inference script
├── inference_image.sh      # Shell script for inference
├── test.png                # Sample input image
├── testnotebook.ipynb      # Experimental Jupyter notebook
└── README.md               # Project documentation
</pre>

<hr>

<h2>4. Environment Setup</h2>

<p>
The project requires Python 3.8 or higher. The following libraries are required:
</p>

<pre>
torch
torchvision
numpy
opencv-python
pillow
matplotlib
</pre>

<p>
Optional: CUDA-enabled PyTorch is recommended for GPU acceleration.
</p>

<hr>

<h2>5. Running the Code</h2>

<h3>5.1 Image Inference</h3>

<p>
To run inference on a single image:
</p>

<pre>
python inference_image.py --image_path test.png
</pre>

<p>
Alternatively, use the provided shell script:
</p>

<pre>
bash inference_image.sh
</pre>

<p>
The inference pipeline performs preprocessing, classification, and (optionally)
forgery localization.
</p>

<hr>

<h2>6. Experimental Notebook</h2>

<p>
The file <code>testnotebook.ipynb</code> contains experiments and visualizations,
including:
</p>

<ul>
  <li>Model loading and configuration</li>
  <li>Residual map visualization</li>
  <li>Forgery localization analysis</li>
</ul>

<hr>

<h2>7. Outputs</h2>

<p>
The system produces the following outputs:
</p>

<ul>
  <li>Deepfake classification result (Real / Fake)</li>
  <li>Confidence scores</li>
  <li>Localization heatmaps highlighting manipulated regions</li>
</ul>

<hr>

<h2>8. Team Members</h2>

<ul>
  <li>Razaib Tariq</li>
  <li>Muhammad Shahid Muneer</li>
  <li>Le Van Hieu</li>
  <li>Yu Hanghao</li>
  <li>황래낙</li>
</ul>

<hr>

<h2>9. Academic Context</h2>

<p>
This project was developed as part of a machine learning / computer vision course
assignment. All experiments and results are intended for academic evaluation
only.
</p>

<hr>

<h2>10. Disclaimer</h2>

<p>
This repository is intended strictly for educational and research purposes.
The authors do not support the misuse of this work for unethical or invasive
applications.
</p>

<hr>

<h2>11. License</h2>

<p>
This project is provided under an academic-use license. Refer to course
guidelines for usage and redistribution policies.
</p>

</body>
</html>
