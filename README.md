# Hello, I'm Omid Torkan 👋

## About
Hello! I'm Omid Torkan, a passionate software developer and data scientist with a strong academic background and hands-on experience in machine learning, deep learning, and natural language processing (NLP). I hold a Master's degree in Computer Science from the University of Milan, where I gained comprehensive knowledge in advanced AI techniques and their applications.

Throughout my academic journey and professional experience, I have developed a solid foundation in programming languages such as Python and SQL, and I'm proficient in using various data analysis and machine learning libraries, including Pandas, NumPy, SciPy, Scikit-Learn, TensorFlow, and Keras. I have also worked extensively with deep learning architectures, particularly LSTM networks and bidirectional LSTMs, for NLP tasks.

My portfolio showcases a range of projects, from sentiment analysis and data augmentation to custom machine learning model development. I am particularly interested in the practical applications of AI in business information systems and information retrieval, which were key focus areas in my coursework and thesis.

In addition to my technical skills, I am proficient in data visualization with Matplotlib and have experience in data pipeline automation and web scraping. My projects reflect a blend of theoretical knowledge and practical implementation, aiming to solve real-world problems with innovative solutions.

I am always eager to learn new technologies and take on challenging projects that push the boundaries of what is possible with AI and data science. Feel free to explore my repositories and get in touch if you'd like to collaborate or discuss interesting ideas!

## Projects Lists:

- [Thesis](https://github.com/OmidTorkan-msc/Thesis-Project): Description of Project 1. *(University Course Project)*
- [Business Information Systems](https://github.com/OmidTorkan-msc/BIS-Project): Description of Project 2. *(University Course Project)*
- [Information Retrieval](https://github.com/OmidTorkan-msc/Causal-relations-in-argumentation-): Description of Project 3. *(University Course Project)*
- [Audio Pattern Recognition](https://github.com/OmidTorkan-msc/Audio-Pattern-Recognition.git): Description of Project 4. *(University Course Project)*
- [Sound In Interaction](https://github.com/OmidTorkan-msc/Sound-Interaction.git): Description of Project 5. *(University Course Project)*
- [Statistical Methods for Machine Learning](https://github.com/OmidTorkan-msc/Statistical-Methods-for-Machine-Learning.git): Description of Project 6. *(University Course Project)*
- [Algorithm For Massive Dataset](https://github.com/OmidTorkan-msc/Algorithm-For-Massive-Dataset): Description of Project 7. *(University Course Project)*
- [Bioinformatics and Genomics](https://github.com/OmidTorkan-msc/Bioinformatics-and-Genomics.git): Description of Project 8. *(University Course Project)*

## My Skills

- **Programming Languages**: Python, SQL  
- **Data Analysis Tools**: Pandas, NumPy, SciPy, Power BI, Excel  
- **Machine Learning Libraries**: Scikit-Learn, TensorFlow, Keras, PyTorch  
- **Deep Learning Techniques**: Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, Bidirectional LSTMs  
- **Natural Language Processing (NLP)**:  
  - Text Preprocessing and Tokenization  
  - Sentiment Analysis (VADER, custom models)  
  - Data Augmentation Techniques (Synonym Replacement)  
  - Text Cleaning (using regex and BeautifulSoup)  
- **Audio Processing and Feature Extraction**:  
  - pyAudioAnalysis (Audio file processing, feature extraction)  
  - pywt (Wavelet Transforms)  
  - ShortTermFeatures (Short-term audio features extraction)  
- **Unsupervised Learning**:  
  - KMeans Clustering  
- **Data Collection and Web Scraping**: `requests`, `BeautifulSoup`  
- **Model Evaluation**: Accuracy, Precision, Recall, F1 Score, ROC Curve, AUC  
- **Model Optimization**: Hyperparameter Tuning with `RandomizedSearchCV`  
- **Deep Learning Frameworks**: TensorFlow, Keras  
- **Visualization Tools**: Matplotlib  
- **Data Pipeline Automation**: Custom model wrappers and class weights handling for imbalanced data  
- **Other Tools**: Git, Jupyter Notebooks, AWS, QlikView  
- **Machine Learning Techniques**:  
  - Model Building and Training (Sequential models, embedding layers, dropout layers)  
  - Custom Classifier Implementations (`BaseEstimator`, `ClassifierMixin`)  
- **Information Management**:  
  - **Data Modeling**: Skilled in designing and structuring data models for efficient storage and retrieval.  
  - **Database Management**: Proficient in working with relational (SQL) and non-relational (NoSQL) databases.  
  - **Data Governance**: Understanding of frameworks for ensuring data integrity, compliance, and quality.  
  - **Data Integration**: Expertise in consolidating data from multiple sources into unified systems.  
  - **Distributed Systems**: Knowledge of data-centric and client-centric consistency models and their trade-offs in distributed environments.  
  - **Information Systems Design**: Capable of designing systems to support organizational operations and decision-making.  
  - **OLTP and OLAP Systems**: Understanding of transaction processing and analytical systems for business data.
## Projects Details:

### Thesis: Enhancing Sentiment Analysis
##### Investigating the Impact of Injecting Polarized Terms in NLP
###### This project demonstrates a robust approach to sentiment analysis across a diverse set of contracts, including business contracts and 10 other types such as Construction Contracts, Contracts for Deed, Daycare Contracts, and more. The methodology leverages VADER Sentiment Analysis to evaluate the tone of contracts and assess their sentiment polarity.
- **Key Features**:

✅1.Contract Types Analyzed:
Business Contracts: Standard corporate agreements.
Construction Contracts: Legal agreements for construction projects.
Contract for Deed: Real estate purchase agreements.
Contract Termination Letters: Letters to terminate agreements.
Daycare Contracts: Agreements for childcare services.
Independent Contractor Agreements: Contracts for freelance work.
Instagram Contracts: Influencer collaboration agreements.
Lawn Service Contracts: Landscaping service agreements.
Musical Performance Contracts: Artist and venue agreements.
Photography Contracts: Agreements for professional photography services.
Trucking Contracts: Freight and logistics contracts.

✅2.Sentiment Analysis:
Uses VADER SentimentIntensityAnalyzer to evaluate:
Negative Sentiment: Indicates dissatisfaction or concerns.
Neutral Sentiment: Represents objective or balanced tone.
Positive Sentiment: Reflects favorable and optimistic language.
Compound Sentiment: Aggregated sentiment score for overall tone.

✅3.Data Augmentation:
Injects synthetic sentences with predefined sentiment polarity (positive or negative) into training datasets to test sentiment shifts.
Simulates real-world sentiment variations to enhance model robustness.

✅4.Visualization:
Plots trends in sentiment polarity across datasets before and after data augmentation.
Provides clear insights into how sentiment shifts impact overall contract tone.
- **Key Techniques and Tools**:
Libraries:
VADER SentimentIntensityAnalyzer: For sentence-level sentiment scoring.
Matplotlib: For visualizing sentiment trends.
Scikit-learn: For preprocessing and model evaluation.
BeautifulSoup: For text cleaning and parsing.
TensorFlow/Keras: For potential model extension.
Data Processing:
Tokenization, text cleaning, and preparation for analysis.
Aggregation of sentiment scores for training and test datasets.
Data Augmentation:
Injects realistic sentences with specific sentiment tones to simulate edge cases.
Model Optimization:
Prepares data pipelines for future integration with machine learning models.
### Applications and Benefits

**1. Broad Applicability**:
- The workflow can be applied to any type of contract or legal document to determine its tone and sentiment.

- Supports businesses, legal teams, and analysts in assessing the emotional undertones of agreements.

**2. Enhanced Insight:**
- By analyzing multiple types of contracts, this project highlights the versatility of sentiment analysis in NLP.
- Augmentation experiments simulate real-world scenarios, helping improve interpretability and robustness.

**3. Future Extensions:**
- Integrate custom machine learning models for deeper analysis.

- Extend the methodology to detect complex emotional cues in large-scale datasets.

**Key Results**
Across all 11 contract types, the sentiment analysis workflow:
- Identified key emotional tones with high accuracy.
- Showcased the adaptability of the approach to various legal and business contexts.
- Demonstrated the impact of data augmentation on sentiment distribution.

- **Business Process Analysis**: Conducted a comprehensive analysis of the car-sharing service business, utilizing **value models** and identifying critical success factors for business growth and efficiency.
- **Process Automation and Execution**: Worked with **BPEL4People** to automate business processes and manage workflow execution, ensuring operational efficiency and compliance.
- **Performance Metrics and KPIs**: Developed and tracked **Key Performance Indicators (KPIs)** to evaluate business performance, focusing on areas like user growth, car bookings, demand forecasting, and customer satisfaction.
- **Critical Success Factors Identification**: Identified and defined **Critical Success Factors (CSFs)** crucial for business success, including inventory management, driver-patient communication, and revenue management.

### Business Process Engineering
- **Business Process Modeling**: Expertise in modeling business processes using **BPMN** (Business Process Model and Notation), including process flow diagrams for various business operations (e.g., login, registration, booking, payment).
- **Business Process Analysis**: Conducted a comprehensive analysis of the car-sharing service business, utilizing **value models** and identifying critical success factors for business growth and efficiency.
- **Process Automation and Execution**: Worked with **BPEL4People** to automate business processes and manage workflow execution, ensuring operational efficiency and compliance.
- **Performance Metrics and KPIs**: Developed and tracked **Key Performance Indicators (KPIs)** to evaluate business performance, focusing on areas like user growth, car bookings, demand forecasting, and customer satisfaction.
- **Critical Success Factors Identification**: Identified and defined **Critical Success Factors (CSFs)** crucial for business success, including inventory management, driver-patient communication, and revenue management.
- **Web-based Application Design**: Developed a conceptual framework for an **online car-sharing platform**, incorporating features like user registration, payment processing, and service management.
- **Stakeholder Communication**: Utilized **BPMN** to create clear, accessible visual documentation for various stakeholders, enabling efficient communication between business analysts, developers, and non-technical participants.

### Image Processing and Computer Vision
- **Image Representation**: Sampling, Quantization  
- **Histogram Analysis**: Equalization, Matching, and Contrast Adjustment  
- **Filtering Techniques**: Smoothing (Gaussian, Median), Sharpening  
- **Edge Detection**: Canny, Sobel, Laplacian  
- **Image Segmentation**: Otsu's Thresholding, Region Growing, Watershed Segmentation  
- **Morphological Operations**: Erosion, Dilation  
- **Frequency Domain Analysis**: Fourier Transform  

### Image Processing Libraries
- **OpenCV**: Image Manipulation, Feature Extraction, Segmentation  
- **Scikit-Image**: Filtering, Segmentation, Feature Extraction  
- **PIL/Pillow**: Image Loading and Manipulation  

### Intelligent Systems and Applications
- **Human Tracking and Detection**:  
  - Vision-Based Tracking Systems (e.g., high-visibility garments, stereo cameras)  
  - Real-Time Tracking with RGB-D Sensors  
  - Integration of Detection and Tracking Algorithms  
- **Safety Monitoring and Industrial Applications**:  
  - Wearable Technology for Worker Safety (e.g., RFID, GPS, Sensor Networks)  
  - Proximity Detection Systems and Collision Avoidance in Hazardous Environments  
  - Monitoring Systems for Worker Productivity and Safety Metrics  
- **Data Analysis and System Design**:  
  - Evaluation of High-Visibility Safety Garments (e.g., compliance with safety standards, degradation analysis)  
  - Filtering and Classification Techniques (e.g., SVM, Histogram-Based Segmentation)  
- **Technologies and Tools**:  
  - `ROS-Industrial`: Open Source Tools for Robotics and Automation  
  - Visual C#: Development of Vision-Based Safety Systems  
  - Spectrophotometry: Analysis of High-Visibility Garments  

### Statistical Methods for Machine Learning
- **Classification Techniques**:  
  - Support Vector Machines (SVM) with Kernel Functions (Linear, Polynomial, RBF, Sigmoid)  
  - Backpropagation Neural Networks (BPN) for Classification and Pattern Recognition  
  - K-Nearest Neighbors (k-NN) for Classification with Optimal k Selection  
- **Inventory Management Models**:  
  - Multi-Criteria Inventory Classification (ABC Analysis, DEA, AHP, TOPSIS)  
  - Optimization Techniques for Inventory Scoring  
- **Machine Learning Tools**:  
  - LIBSVM for SVM Implementation  
  - MATLAB Neural Network Toolbox for BPN and k-NN Development  
- **Applications**:  
  - Inventory Classification in Supply Chain Management  
  - Comparative Analysis of Machine Learning Models for Classification
### Algorithms for Massive Datasets
- **Scalable Data Processing**: Implementing algorithms capable of processing large datasets efficiently.  
- **Data Preprocessing for Large Datasets**: Techniques for data normalization, resizing, and handling high-dimensional data.  
- **Distributed Data Processing**: Familiarity with methods like data partitioning and batch processing to scale computations.  
- **Algorithmic Techniques**:  
  - Stream processing algorithms for single-pass data handling.  
  - Approximation algorithms for large-scale problems.  
  - External memory algorithms for out-of-core processing.  
- **Data Generators**: Using tools like Image Data Generators for managing large-scale training data dynamically.  
- **Neural Network Optimization**:  
  - Designing convolutional architectures (CNNs) for feature extraction.  
  - Utilizing advanced activation functions (Leaky ReLU) and regularization techniques (dropout layers).  
  - Employing optimizers like Adam for scalable and efficient training.  
- **Performance Scaling**: Strategies to manage memory constraints and improve model scalability on massive datasets.  
- **Model Evaluation at Scale**: Techniques for validating and testing models with large datasets using metrics like accuracy and loss trends.  
### Bioinformatics and Genomics
- **Cis-Regulatory Region Analysis**: Prediction of active promoters and enhancers using deep learning models.  
- **Deep Learning Models for Genomic Data**:  
  - Perceptron for linear feature combinations.  
  - Feedforward Neural Networks for extracting nonlinear patterns.  
  - Convolutional Neural Networks (CNNs) for feature extraction and sequence pattern recognition.  
  - Multi-Model Neural Networks for combining CNN and feedforward features.  
- **Epigenomics Data Handling**: Experience with datasets like HG38 and tools like FANTOM and UCSC Genome Browser for genomic analysis.  
- **Data Preprocessing Techniques**:  
  - KNN Imputation for handling missing values.  
  - Robust scaling for normalization of genomic data.  
  - Pearson and Spearman correlation for identifying and removing redundant features.  
- **Feature Selection**: Filtering highly correlated features to optimize model performance.  
- **Data Visualization**: Applied techniques like t-SNE for visualizing high-dimensional genomic data in reduced dimensions.  
- **Experimental Setup for Genomic Prediction**:  
  - Stratified Holdout for ensuring balanced training and test sets.  
  - Metrics like Mean Squared Error (MSE) and loss for model evaluation.  
- **Programming and Tools**: Python, TensorFlow, Keras, SciPy, and sklearn.  


  


## 🔭 I’m currently focusing on

- Expanding my expertise in NLP and deep learning through personal projects.
- Exploring cutting-edge AI techniques for real-world applications.
- Contributing to open-source projects in machine learning and data science.
- Enhancing my skills in scalable data pipelines and cloud-based analytics.


## 👯 I’m looking to collaborate on

- Open-source machine learning projects.
- NLP applications.
- Data science challenges.
- AI for social good.
- Innovative solutions in business information systems.

## 🤔 I’m looking for help with

- Collaborating on open-source projects related to machine learning.
- Finding mentors in the AI industry.
- Improving my knowledge in data visualization techniques.
- Getting feedback on my thesis project.

## 💬 Ask me about

- Machine learning algorithms.
- Deep learning models.
- NLP techniques.
- Data science workflows.
- Python programming.
- Open-source contributions.

## ⚡ Fun fact

- I love to solve complex puzzles and often participate in competitive coding events.
- In my free time, I enjoy hiking and exploring nature trails.

## 📫 How to reach me

- **Email**: [omid.torkan@outlook.com]
- **LinkedIn**: [www.linkedin.com/in/omid-torkan-2b30b01b4]
- **Twitter**: [https://x.com/OmidTorkan86]
