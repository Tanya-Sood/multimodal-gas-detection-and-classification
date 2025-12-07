# multimodal-gas-detection-and-classification
A) Preprocessing
● Images resized to 224×224.
● Grayscale images converted to 3 channels.
● Pixel intensities normalized using ImageNet mean and std.
● Stratified dataset split:
    - 70% Training
    - 20% Validation
    - 10% Test
● Data Augmentation:
    - Random Erasing (p=0.2)
● Sensor values cleaned:
    - Conversion to float
    - Missing or non-numeric values replaced with 0

B) Model Architectures
1) Base Model — Sensor-Only MLP
This model uses only the 7 MQ sensor features (MQ2, MQ3, MQ5, MQ6, MQ7, MQ8, MQ135). 
It serves as a baseline to compare unimodal vs multimodal performance.
Architecture:
• Input: 7-dimensional sensor vector
• Linear (7 → 64), BatchNorm, ReLU
• Linear (64 → 128), BatchNorm, ReLU
• Output: 4 gas classes
Advantages:
• Lightweight
• Fast training
Limitations:
• Cannot interpret temperature patterns from images
• Accuracy significantly lower than multimodal models

2) Transfer Learning Model — ImageOnly ResNet18
This model uses thermal images only.  
ResNet18 pretrained on ImageNet is used as the feature extractor.
Architecture:
• ResNet18 without final FC layer → outputs 512 features
• Custom Linear Layer: 512 → 4 classes
Advantages:
• Learns strong texture/pattern information
• Faster convergence due to pretrained weights
Limitations:
• Sensors contain gas concentration info that images lack

3) Hybrid Multimodal Model (ResNet18 + Sensor MLP)
This architecture merges image and sensor modalities for highest accuracy.
Image Path:
• ResNet18 backbone → 512-dim feature vector
Sensor Path:
• MLP: Linear → BatchNorm → ReLU → Linear → BatchNorm → ReLU → 256-dim output
Fusion:
• Concatenation of 512 (image) + 256 (sensor) = 768 features
• Classifier:
    - Linear (768 → 256) → ReLU → Dropout
    - Linear (256 → 128) → ReLU → Dropout
    - Linear (128 → 4 classes)
This model captures both:
• visual thermal patterns
• chemical sensor signatures

C) Training Details
Parameters:
• Optimizer: AdamW
• Loss Function: CrossEntropyLoss
• Learning Rate: 1e-4
• Scheduler: ReduceLROnPlateau
• Epochs: 5
• Batch Size: 32
• Metric: Accuracy

Each epoch computes:
• Training accuracy & loss
• Validation accuracy & loss
• Confusion matrix post-training

3. Results & Evaluation
The training pipeline generates:
1. Accuracy curves (Train vs Validation)
2. Loss curves (Train vs Validation)
3. Confusion Matrix for each model

Expected Observations:
• Base Model — Lowest accuracy; struggles with complex patterns.
• Transfer Learning — Strong improvement; ResNet18 learns image textures.
• Hybrid Model — Best performance; fusing modalities reduces misclassifications.

Confusion Matrix Interpretation:
• Errors usually between visually similar gas types.
• Hybrid reduces ambiguity by using sensor data.
<img width="900" height="341" alt="image" src="https://github.com/user-attachments/assets/2e83ff79-e0c6-4529-ba6c-e5b9bc60ed6e" />
