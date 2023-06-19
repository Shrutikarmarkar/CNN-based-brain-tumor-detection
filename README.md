# CNN-based-brain-tumor-detection
### Project description
Our objective was to develop a deep learning model using Convolutional Neural Networks (CNN) to accurately predict brain tumor presence in MRI images. We explored various CNN architectures and achieved a promising accuracy of 97% with our custom architecture. To further improve results, we incorporated transfer learning, combining VGG16 with our architecture, resulting in the highest accuracy of 99.17% for our dataset.

### Dataset
Dataset: Our dataset, "Br35H:: Brain Tumor Detection 2020," was sourced from Kaggle. It comprises 400 MRI brain images, evenly split between brain tumors and regular brain scans. Data augmentation techniques were applied to enhance dataset diversity and size. The data was divided into 70% for training, 15% for validation, and 15% for testing. We had 1679 training examples, 360 development examples, and 360 test examples.

### Feature Extraction
During the feature extraction stage, we utilized the pre-trained VGG-16 Convolutional Neural Network to extract high-level features from the brain MRI images. By leveraging the knowledge learned by VGG-16, we captured distinctive patterns and structures from the data, transforming the raw photos into meaningful numerical features for brain tumor detection.

### Network Selection
For network selection, we initially achieved a notable accuracy of 97% with our baseline CNN architecture. To further enhance performance, we explored transfer learning with pre-trained models including Inception, ResNet, VGG16, and VGG19. After extensive experimentation, we found that combining our novel architecture with the VGG16 pre-trained model yielded the highest accuracy for our dataset. This fusion leveraged VGG16's deep representation learning capabilities and the adaptability of our custom architecture, surpassing the baseline accuracy.

### CNN Architecture
The proposed model architecture for brain tumor detection combines pre-trained VGG16 layers with additional convolutional and pooling layers. The VGG16 model is loaded with pre-trained weights and frozen during transfer learning. Batch normalization is applied for stability, and zero-padding preserves border-related features. Hyperparameters such as learning rate, momentum, dropout rate, batch size, and optimizer (RMSprop) are fine-tuned for improved accuracy. The model includes a fully connected layer with a sigmoid activation function for binary classification. It is optimized with the RMSprop optimizer and compiled as the "BrainDetectionModel" for training and evaluation.



### Result
The proposed model achieved exceptional results in brain tumor detection. During training, it achieved an accuracy of 98.69%, while the validation accuracy was 98.33%. The RMSprop optimizer was employed to optimize the model's performance. In the final evaluation on the test dataset, the model demonstrated a remarkable accuracy of 99.17%. These results highlight the effectiveness of the model in accurately detecting brain tumors, showcasing its potential for enhancing medical diagnostics and improving patient care.
