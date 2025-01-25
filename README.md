# Hello, I'm Omid Torkan 👋

## About
Hello! I'm Omid Torkan, a passionate software developer and data scientist with a strong academic background and hands-on experience in machine learning, deep learning, and natural language processing (NLP). I hold a Master's degree in Computer Science from the University of Milan, where I gained comprehensive knowledge in advanced AI techniques and their applications.

Throughout my academic journey and professional experience, I have developed a solid foundation in programming languages such as Python and SQL, and I'm proficient in using various data analysis and machine learning libraries, including Pandas, NumPy, SciPy, Scikit-Learn, TensorFlow, and Keras. I have also worked extensively with deep learning architectures, particularly LSTM networks and bidirectional LSTMs, for NLP tasks.

My portfolio showcases a range of projects, from sentiment analysis and data augmentation to custom machine learning model development. I am particularly interested in the practical applications of AI in business information systems and information retrieval, which were key focus areas in my coursework and thesis.

In addition to my technical skills, I am proficient in data visualization with Matplotlib and have experience in data pipeline automation and web scraping. My projects reflect a blend of theoretical knowledge and practical implementation, aiming to solve real-world problems with innovative solutions.

I am always eager to learn new technologies and take on challenging projects that push the boundaries of what is possible with AI and data science. Feel free to explore my repositories and get in touch if you'd like to collaborate or discuss interesting ideas!

## Projects :

### [Thesis : Enhancing Sentiment Analysis](https://github.com/OmidTorkan-msc/Thesis-Project)
##### Investigating the Impact of Injecting Polarized Terms in NLP
###### This project demonstrates a robust approach to sentiment analysis across a diverse set of contracts, including business contracts and 10 other types such as Construction Contracts, Contracts for Deed, Daycare Contracts, and more. The methodology leverages VADER Sentiment Analysis to evaluate the tone of contracts and assess their sentiment polarity.
- **Key Features**:

🔴Contract Types Analyzed:
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

🔴Sentiment Analysis:
Uses VADER SentimentIntensityAnalyzer to evaluate:
Negative Sentiment: Indicates dissatisfaction or concerns.
Neutral Sentiment: Represents objective or balanced tone.
Positive Sentiment: Reflects favorable and optimistic language.
Compound Sentiment: Aggregated sentiment score for overall tone.

🔴Data Augmentation:
Injects synthetic sentences with predefined sentiment polarity (positive or negative) into training datasets to test sentiment shifts.
Simulates real-world sentiment variations to enhance model robustness.

🔴Visualization:
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

### Key Results
**Across all 11 contract types, the sentiment analysis workflow:**

- **Identified key emotional tones with high accuracy.**
- **Showcased the adaptability of the approach to various legal and business contexts.**
- **Demonstrated the impact of data augmentation on sentiment distribution.**
### [Business Information Systems(University Course Project):](https://github.com/OmidTorkan-msc/BIS-Project)
This project, conducted as part of my Master's degree at the University of Milan, leverages process mining techniques to analyze and optimize business processes using event log data from a multinational Dutch corporation involved in coatings and paints.

**Overview**
The project focuses on preprocessing event logs, identifying common process behavior, and applying various process discovery algorithms. Key steps include data cleaning, noise removal, segmenting logs, and evaluating process mining algorithms.

- **Key Components:**

 -**Data Preparation:**

In-depth exploration of the event log file, including activity frequencies, start and end activities, case duration analysis, and category distribution.
Filtering and removing noise to improve data quality and ensure relevant insights.

 -**Process Discovery:**

**Application of three process mining algorithms:**

Alpha Miner: Identifies parallel activities but struggles with complex process models.

Inductive Miner: Bottom-up approach for creating more accurate process models.

Heuristic Miner: Handles noisy data and captures more complex behavior.
Comparison of results using process trees and Petri-nets for model visualization.

Segmenting and Filtering Data:

Segmenting logs and applying filters to remove anomalies, rework activities, and infrequent variants.
Focus on detecting and addressing bottlenecks to improve process efficiency.

  -**Goals:**

Knowledge Uplift: Identifying actionable interventions to enhance process completion, reduce management costs, and detect dysfunctional executions.
Performance Evaluation: Using heatmap charts to compare the effectiveness of different process discovery algorithms.
Tools Used:

Python (with libraries like PM4Py for process mining and Disco for performance analytics)
Petri-nets, Process Trees for visual representation
This project provides a comprehensive analysis of process mining techniques and their application to real-world business log data.




### [Information Retrieval(University Course Project):](https://github.com/OmidTorkan-msc/Causal-relations-in-argumentation-)

**1. Course Overview:**
Briefly describe the core principles and objectives of the information retrieval course. Mention topics like search engines, ranking algorithms, indexing, and query processing. Highlight how these elements are crucial for retrieving relevant information from large datasets, an essential task in many NLP applications.

**2. Linking Information Retrieval to NLP:**
Discuss how information retrieval techniques are applied to NLP tasks. For instance, text-based search and retrieval involve understanding query intent, ranking relevant results, and processing text data efficiently, which aligns with how NLP models analyze and understand language.
Mention how methods like tokenization, term frequency, and document indexing are foundational for models that interpret large corpora of text, as seen in my causal relations in argumentation project.

**3. Relevance to Causal Relation Detection:**
Relate the knowledge gained from information retrieval to my work on causal relations in argumentation. Point out how the identification of causal relationships between entities in texts can be seen as a form of information retrieval, where the “cause” is retrieved as relevant information from a sentence or document and connected to the “effect.”
Mention how I used deep learning and NLP libraries (like NLTK) to process and retrieve relevant data for my cause-effect relationship project, drawing parallels to information retrieval's focus on efficient data access and retrieval.

**4. Application of Techniques in my Project:**
Talk about the specific techniques I learned in the course (like tokenization and vectorization) and how they were essential for preprocessing the text data in my deep learning models. For example, tokenizing text into sentences or words, stemming, and lemmatization directly impact how the network identifies patterns, similar to how an information retrieval system would process and index a document.
Explain how the bidirectional LSTM with attention mechanism helped I retrieve important context from the text, akin to a retrieval system that selects and ranks relevant pieces of data based on a query’s needs.

**5. Challenges and Insights:**
Reflect on any challenges faced during the project that overlapped with information retrieval tasks, such as handling large datasets, ensuring the model retrieves relevant features (words/sentences), or dealing with noisy data.
Share insights gained from applying information retrieval techniques in the real-world NLP tasks I worked on, such as how ranking and retrieval systems could be enhanced with deep learning models for more sophisticated language processing.

**6. Future Directions:**
Conclude by considering how the concepts from the information retrieval course could inform future improvements in my NLP projects, particularly around optimizing models for better retrieval and classification of causal relationships, or even enhancing search algorithms within NLP applications.


### [Speech Emotion Recognition using MFCC and PWP Features(University Course Project)](https://github.com/OmidTorkan-msc/Audio-Pattern-Recognition.git)
 
This project focuses on recognizing emotions from speech signals by employing Mel-Frequency Cepstral Coefficients (MFCCs) and Perceptual Wavelet Packets (PWP) as acoustic features, and classifying emotions using K-Nearest Neighbors (KNN) and Support Vector Machines (SVM). The primary goal is to enhance emotion detection accuracy by leveraging different feature extraction methods and classification schemes.

**Key Features:**
Mel-Frequency Cepstral Coefficients (MFCCs): Widely used in speech processing, MFCCs model the human auditory system and are effective for emotion detection in speech.

**Perceptual Wavelet Packets (PWP):** These features capture perceptual properties of speech, offering a more detailed spectral analysis using wavelet transforms.

**Methodology:**
Feature Extraction: MFCCs and PWP are extracted from speech signals to capture their emotional content.

Clustering: K-means clustering is applied to the feature spaces of MFCC and PWP to group similar patterns.
Classification: KNN and SVM classifiers are used to categorize emotions such as sadness, anger, happiness, and neutrality.

**Dataset:**
The Berlin Emotional Speech Database (Emo-dB), which contains German-language recordings of emotional speech, was used for training and testing. The dataset includes recordings of 535 emotional samples spoken by 5 male and 5 female speakers, representing emotions such as anger, boredom, disgust, fear, sadness, happiness, and neutrality.

**Results:**
The proposed system demonstrates robustness in recognizing basic emotions.
SVM with a linear kernel outperformed other classifiers, achieving high accuracy in emotion classification.

### [Spatial Sound Rendering in Python and Pyo(University Course Project)](https://github.com/OmidTorkan-msc/Sound-Interaction.git)

**Overview**
This project explores the application of reverberation techniques for spatial sound processing using Python and the Pyo library. Reverberation, a key factor in spatial audio perception, is simulated using various artificial methods to replicate the acoustic properties of real-world environments.

**Key Techniques**

**1.	Freeverb:** Implements Schroeder's reverberation model using parallel comb filters and series all-pass filters to replicate natural reflections.

**2.	WGverb:** Simulates reverberation via a Feedback Delay Network (FDN) for high-density echo approximation.

**3.	Convolve:** Utilizes circular convolution for reverberation based on impulse response measurements.

**4.	HRTF (Head-Related Transfer Function):** Processes binaural audio signals to simulate 3D spatial sound for enhanced listener experience.

**5.	Binaural Rendering:** Dynamically applies spatial transformations for virtual audio sources using directional and positional filters.

**Applications**
The project demonstrates the integration of these methods for use in virtual reality, video games, music production, and immersive audio simulations. It includes detailed analysis of audio effects generated by the methods and their parameter sensitivity.


 
### [Statistical Methods for Machine Learning](https://github.com/OmidTorkan-msc/Statistical-Methods-for-Machine-Learning.git)

 This project focused on leveraging Support Vector Machines (SVM) to analyze and classify large-scale kernel machines, a critical method for inventory management and optimization. The study utilized advanced statistical and machine learning techniques, including k-Nearest Neighbors (k-NN) and Backpropagation Neural Networks (BPN), to evaluate and enhance traditional inventory classification methods.
**Key Highlights:**

**Problem Addressed:**
The project tackled the challenge of inventory classification in supply chain management, emphasizing multi-criteria evaluation over traditional methods relying solely on annual dollar usage.

**Methods and Techniques:**
SVM was used for its robustness in creating optimal hyperplanes for classification in high-dimensional spaces.
k-NN and BPNs were integrated for comparative analysis, focusing on accuracy and predictive power.
Genetic algorithms and multi-criteria approaches, including techniques like AHP and TOPSIS, were reviewed for broader applicability in inventory classification.

**Contributions:**

Demonstrated the superior accuracy of AI-based techniques like SVM and BPN over classical methods (e.g., multiple discriminant analysis).
Evaluated the trade-offs of various models, identifying SVM as a versatile solution for applications ranging from inventory management to financial analysis and bioinformatics.
Highlighted the potential of neural networks and genetic algorithms for dynamic inventory classification in real-world datasets.

**Applications:**
This project provides a framework for improving supply chain operations by integrating machine learning techniques, with implications for industries requiring efficient inventory management strategies.

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
## Projects :


### Business Process Engineering and Automation in Car-Sharing Service
  - Conducted a comprehensive analysis of the car-sharing service business, utilizing value models to identify critical success factors (CSFs) for business growth and efficiency.
  - Expertise in **Business Process Modeling** using **BPMN** (Business Process Model and Notation), including creating process flow diagrams for key operations such as login, registration, booking, and payment.
  - Identified and defined **Critical Success Factors (CSFs)**, focusing on areas like inventory management, driver-patient communication, and revenue management.
  
- **Process Automation and Execution**
  - Worked with **BPEL4People** to automate business processes, manage workflow execution, and ensure operational efficiency and compliance.
  
- **Performance Metrics and KPIs**
  - Developed and tracked **Key Performance Indicators (KPIs)** to evaluate business performance, focusing on metrics like user growth, car bookings, demand forecasting, and customer satisfaction.
  
- **Web-based Application Design**
  - Developed a **conceptual framework** for an online car-sharing platform, incorporating features like user registration, payment processing, and service management.
  
- **Stakeholder Communication**
  - Created clear, accessible visual documentation using **BPMN** to facilitate efficient communication between business analysts, developers, and non-technical stakeholders.


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
