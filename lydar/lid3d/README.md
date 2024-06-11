# lid3d

Feature extraction.

lidar&python code: [tuto](https://medium.com/@christhaliyath/lidar-and-pointcloud-simple-tutorial-for-op-3c5d7cd35ad4)

## PointNet

PointNet is a deep learning architecture designed for processing point clouds, which are sets of points in a three-dimensional space. PointNet can be used for various tasks, primarily classification and semantic segmentation. Here’s an overview of the differences between these two tasks as performed by PointNet:

- Classification
Objective: To categorize the entire point cloud into one of several predefined classes.
Input: A point cloud representing a single object or scene.
Output: A single label representing the class of the entire input point cloud (e.g., chair, table, car).
Architecture Focus: Global feature extraction. PointNet uses a symmetric function (e.g., max pooling) to aggregate information from all points into a global feature vector, which is then used to predict the class label.

- Semantic Segmentation
Objective: To assign a class label to each individual point in the point cloud.
Input: A point cloud, potentially representing a complex scene with multiple objects and surfaces.
Output: A label for each point in the input point cloud, indicating the category of the object or surface that the point belongs to (e.g., points belonging to the ground, building, tree).
Architecture Focus: Local and global feature extraction. PointNet processes points individually and then aggregates global features. However, it also combines these global features with local features to provide context-aware predictions for each point.
