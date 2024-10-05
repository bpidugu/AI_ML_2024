# Transfer learning 

### Topics
- Regularization in CNN's
- Transfer learning
- Different types of Transfer learning
---
- Features: They are the characterstics extracted from data, such as edges, corners, or textures, that are crucial for tasks like object recognition, and image classification in Computer vision
- Individual filters are not invariant to the rotation of an image. To make our CNN model rotation-invariant, we must train it with rotated images. For that, we have to include such images in the training dataset.
- Batch Normalization adds noise to the data, which results in regularization. weak regularizer in CNNs
- In Spatial Dropout, the entire feature maps are dropped at random by setting all cell values to 0.
- ImageData Generator is used for data augmentation. It takes input images and transforms them to create new images from the input data.
- 