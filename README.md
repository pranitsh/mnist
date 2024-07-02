# mnist

## Significance

The results achieved are ranked 35th globally:
According to papers with code, the accuracy rate achieved (96.2%) places 35th place on the scoreboard out of 70 research papers. https://paperswithcode.com/sota/image-classification-on-mnist?metric=Accuracy

Statistics say that market penetration of AI in US could be expanded:
According to a penetration report led by Forbes Advisor, 25% of companies [use] AI and 43% [are] exploring its potential applications. https://www.forbes.com/advisor/business/ai-statistics/

Initial investment in the dataset created an industry worth billions:
The global Optical Character Recognition(OCR) Software market size was valued at USD 7891.05 million in 2021 and is expected to expand at a CAGR of 13.85% during the forecast period, reaching USD 17180.93 million by 2027. https://www.precisionreports.co/enquiry/request-sample/21438909

## Summary of Work

The work showcases a comprehensive understanding of dataset handling, visualization, implementation of machine learning models, and performance optimization techniques in Python.

### Dataset Preparation and Visualization

Used libraries `numpy`, `scikit-learn`, `matplotlib`, `scikit-learn`, and `seaborn`
Fetched data with `fetch_openml` from `scikit-learn` to load the MNIST dataset; processed data by converting images to a numpy array and targets to integer type; and visualized individual images and the first instance of each digit using `matplotlib`.

### Standard Machine Learning Models

Tested random forest classifier, logistic regression, support vector machine (SVM), gradient boosting machines (GBM), and k-nearest neighbors (KNN).

- Random Forest: 6.45 seconds, 5.44% error rate
- Logistic Regression: 56.96 seconds, 10.12% error rate
- SVM: 34.23 seconds, 3.81% error rate
- GBM: 382.74 seconds, 7.57% error rate
- KNN: 3.88 seconds, 5.48% error rate
- **Best Model:** SVM with the lowest error rate of 3.81%.

### Custom Model Implementations

Implemented a function to calculate mean squared error between images and built a simple custom model to predict digits by comparing each image to the first instance of each digit (resulted in an error rate of approximately 48.17%).

- Displayed incorrectly classified images to analyze performance.

Implemented an optimized KNN classifier from scratch, improving performance from 0.178 seconds per image for 100 images to 0.0671 seconds per image for 1000 images.

- Trained the classifier on 10,000 images and tested on 1,000 evaluation images.
- Used numerous optimization techniques, including preallocation (assigning memory before performing operations), views (manipulating current data intead of recreating it in necessary format), in-olace calculation (using numpy calculations instead of python-based iteration and loops)

This approach uses a partition algorithm instead of a spatial data structure. Future efforts will be directed to implementing a spatial data structure to improve performance.

### Trained Feed Forward Neural Net

Implemented activation functions `tanh` and `sigmod` and their derivatives. Defined loss functions logarithmic loss `logloss` and its derivative. Initialized a neural network layer with weights and biases and defined forward and back propogation. 

- Network with two layers: first layer with 3 neurons and `tanh` activation, second layer with 1 neuron and `sigmoid` activation
- Set number of epochs to 20,000. Calculated and recorded the cost at each epoch to monitor training progress.
- Plotted the training cost over epochs to visualize the learning curve.
