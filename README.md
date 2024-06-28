# mnist

## Summary of Work

The work showcases a comprehensive understanding of dataset handling, visualization, implementation of machine learning models, and performance optimization techniques in Python.

### Dataset Preparation and Visualization

Used libraries `numpy`, `scikit-learn`, `matplotlib`, `scikit-learn`, and `seaborn`
Fetched data with `fetch_openml` from `scikit-learn` to load the MNIST dataset; processed data by converting images to a numpy array and targets to integer type; and visualized individual images and the first instance of each digit using `matplotlib`.

### Standard Machine Learning Models

Tested Random Forest Classifier, Logistic Regression, Support Vector Machine (SVM). Gradient Boosting Machines (GBM), and K-Nearest Neighbors (KNN).

- Results are Random Forest: 6.45 seconds, 5.44% error rate; Logistic Regression: 56.96 seconds, 10.12% error rate; SVM: 34.23 seconds, 3.81% error rate, GBM: 382.74 seconds, 7.57% error rate; KNN: 3.88 seconds, 5.48% error rate
- Best Model: SVM with the lowest error rate of 3.81%.

### Custom Model Implementations

Implemented a function to calculate mean squared error between images and built a simple custom model to predict digits by comparing each image to the first instance of each digit (resulted in an error rate of approximately 48.17%).

- Displayed incorrectly classified images to analyze performance.

Implemented an optimized KNN classifier from scratch, improving performance from 0.178 seconds per image for 100 images to 0.0671 seconds per image for 1000 images.

- Trained the classifier on 10,000 images and tested on 1,000 evaluation images.
- Used numerous optimization techniques, including preallocation (assigning memory before performing operations), views (manipulating current data intead of recreating it in necessary format), in-olace calculation (using numpy calculations instead of python-based iteration and loops)

This approach uses a partition algorithm instead of a spatial data structure. Future efforts will be directed to implementing a spatial data structure to improve performance.
