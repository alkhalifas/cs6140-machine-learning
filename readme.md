# **CS 6140 Machine Learning Repository**

Welcome to my **CS 6140 Machine Learning Repository**, where I showcase my hands-on work in machine learning, focusing on cutting-edge AI/ML algorithms, data visualization, and real-world applications. Each assignment in this repository highlights practical implementations of key machine learning concepts combined with theoretical insights.

---

### **Assignment 1: Regression and Classification Fundamentals**

#### **Ordinary Least Squares (OLS) Regression**
- **Objective:** Implement and evaluate OLS regression on the Boston Housing dataset.
- **Highlights:**
  - Explored the relationship between predictor variables and housing prices.
  - Visualized regression results to assess goodness-of-fit.
  - Implemented gradient descent for optimizing regression coefficients.

#### **Ridge and Lasso Regression**
- **Objective:** Regularize regression models to prevent overfitting.
- **Highlights:**
  - Applied Ridge and Lasso regression on the Boston Housing dataset.
  - Analyzed the effect of regularization parameters on model performance.
  - Compared the sparsity and robustness of coefficients between Ridge and Lasso.

#### **Gaussian Naive Bayes and Logistic Regression**
- **Objective:** Train and evaluate classification models on the Breast Cancer Wisconsin dataset.
- **Highlights:**
  - Implemented Gaussian Naive Bayes for probabilistic classification.
  - Built a Logistic Regression model with regularization for binary classification.
  - Compared model accuracies and decision boundaries.

---

### **Assignment 2: Advanced Classification and Optimization**

#### **Perceptron Algorithm**
- **Objective:** Implement and test the Perceptron algorithm for binary classification.
- **Highlights:**
  - Developed a perceptron model with a focus on convergence for linearly separable data.
  - Visualized classification boundaries and training progress.

#### **Support Vector Machines (SVMs)**
- **Objective:** Explore SVMs with different kernels for classification tasks.
- **Highlights:**
  - Implemented linear and RBF kernel SVMs to classify synthetic datasets (moons and circles).
  - Analyzed model performance under varying noise levels.
  - Evaluated SVM training time and computational cost.

#### **Complementary Slackness and Kernel SVM**
- **Objective:** Explore the complementary slackness condition in SVMs using a moons dataset.
- **Highlights:**
  - Visualized the decision boundary under different slack penalties.
  - Analyzed kernel effects using polynomial and RBF kernels.

---

### **Assignment 3: Neural Networks and Time-Series Analysis**

#### **Multilayer Perceptron (MLP)**
- **Objective:** Implement a simple MLP for classification.
- **Highlights:**
  - Experimented with hidden layer sizes and activation functions.
  - Used cross-validation to evaluate model robustness.
  - Compared MLP performance to SVM for synthetic datasets.

#### **Gaussian Discriminant Analysis**
- **Objective:** Apply GDA for classification on the Breast Cancer Wisconsin dataset.
- **Highlights:**
  - Derived and implemented GDA equations for model training.
  - Compared GDA with Gaussian Naive Bayes in terms of accuracy and computation.

#### **RNNs for Time-Series Forecasting**
- **Objective:** Design a synthetic time-series signal and apply RNNs to learn dependencies.
- **Highlights:**
  - Created time-series signals with long-term dependencies using trends and noise.
  - Trained RNNs with varying memory sizes to explore their limitations.
  - Visualized RNN predictions to evaluate the model’s capacity for long-term memory.

---

### **Assignment 4: Dimensionality Reduction and Data Visualization**

#### **Dimensionality Reduction on MNIST**
- **Objective:** Apply PCA, t-SNE, and Autoencoders to visualize high-dimensional data.
- **Highlights:**
  - **PCA:** Linear approach capturing maximum variance for efficient global structure visualization.
  - **t-SNE:** Non-linear technique addressing the crowding problem for local neighborhood preservation.
  - **Autoencoders:** Neural network-based reduction for capturing complex patterns.
  - Compared and contrasted these techniques for clarity, efficiency, and applicability.

#### **Research Summary on t-SNE**
- **Objective:** Summarize **"Visualizing Data using t-SNE"** by Laurens van der Maaten and Geoffrey Hinton.
- **Highlights:**
  - Explained the t-SNE algorithm and its solution to the crowding problem.
  - Discussed applications of t-SNE in diverse fields such as drug discovery, single-cell analysis, and word embeddings.
  - Connected theoretical insights to practical applications from Task 1.

#### **Generative Models**
- **Objective:** Implement Variational Autoencoders (VAEs) and compare them to traditional Autoencoders.
- **Highlights:**
  - Trained VAEs on MNIST to learn probabilistic representations.
  - Evaluated the quality of generated samples and latent space embeddings.

---


## **Technologies Used**
- **Python Libraries:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, `tensorflow`, `pypi` and more.
- **Frameworks:** `PyTorch`, `Keras`.
- **Datasets:** Boston Housing, Breast Cancer Wisconsin, MNIST, and synthetic datasets.

---

## **How to Use**
Clone the repository:

   ```bash
   git clone https://github.com/salehalkhalifa/cs6140-machine-learning.git
   
   cd cs6140-machine-learning
   ```



Save to PDF:

  ```bash
  jupyter nbconvert --to pdf YourNotebook.ipynb
  ```
