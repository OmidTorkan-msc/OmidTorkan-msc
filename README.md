# Hello, I'm Omid Torkan 👋

## About
Hello! I'm Omid Torkan, a passionate **Data Analyst** with a strong background in Data Engineering and Machine Learning. I hold a Master's degree in Computer Science from the University of Milan, where I developed expertise in data analysis, statistical modeling, and machine learning.

As a Data Analyst, I specialize in transforming raw data into actionable insights by applying advanced techniques in data cleaning, exploratory data analysis (EDA), and visualization using tools like Pandas, NumPy, SciPy, and Power BI. I am skilled at analyzing large datasets, identifying trends and patterns, and making data-driven decisions to drive business success.

I have hands-on experience with a variety of machine learning algorithms, including regression models, classification, clustering (e.g., KMeans), and time-series analysis. I'm proficient in building end-to-end data pipelines, automating data processes, and integrating data from multiple sources. I also have experience in SQL for data manipulation and query optimization.

I enjoy exploring real-world data problems, applying statistical methods, and developing predictive models that provide value to businesses. My projects reflect my ability to deliver insights and solutions through data analysis, whether it's in business performance, sentiment analysis, or other domains.

I'm always eager to expand my knowledge and explore new data analysis techniques. Feel free to explore my repositories, and feel free to reach out if you'd like to collaborate or discuss innovative data solutions!

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
### [Business Information Systems(University Course Project)](https://github.com/OmidTorkan-msc/BIS-Project)
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




### [Information Retrieval(University Course Project)](https://github.com/OmidTorkan-msc/Causal-relations-in-argumentation-)

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


 
### [Statistical Methods for Machine Learning(University Course Project)](https://github.com/OmidTorkan-msc/Statistical-Methods-for-Machine-Learning.git)

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

### [Image Classification via Neural Network(University Course Project)](https://github.com/OmidTorkan-msc/Algorithm-For-Massive-Dataset)

**Project Summary:**

In this project, I developed a convolutional neural network (CNN) for classifying Turkish Lira banknotes into six denominations: 5, 10, 20, 50, 100, and 200. The dataset contained 6,000 balanced images with dimensions 1280x720x3. I applied preprocessing techniques, including resizing images to reduce computational costs and normalizing RGB values to accelerate convergence.
The CNN architecture consisted of convolutional layers for feature extraction, max-pooling layers for scalability, LeakyReLU activation to avoid the "dying ReLU" problem, and dropout layers to prevent overfitting. The output layer used a softmax activation function for class prediction. The model was trained on 5,550 samples and validated on 450 images, achieving 98% test accuracy and 97% training accuracy after 12 epochs.
To manage the large dataset, I implemented an image data generator to load batches into memory dynamically, enabling scalability and efficient training. The results demonstrate the model's robustness and adaptability to new data, showcasing its potential for real-world applications in image classification.

**Key Details:**

•	Dataset: 6,000 images (balanced across six classes).

•	Preprocessing: Image resizing, RGB normalization.

•	Architecture:

o	Convolutional layers with LeakyReLU activation.

o	Max-pooling and dropout layers for scalability and regularization.

o	Softmax activation for classification.

•	Optimizer: Adam with categorical cross-entropy loss.

•	Accuracy: 98% (test), 97% (training).

•	Tools: TensorFlow and Keras.

•	Scalability: Utilized an image data generator to handle large datasets efficiently.

**Files:**

1.	preprocess-dataset.ipynb: Handles image resizing and dataset organization.

2.	neural-net.ipynb: Defines the CNN model, trains it using the processed dataset, and evaluates performance.
This project highlights the practical application of deep learning for real-world image classification tasks and addresses challenges like handling large datasets and optimizing model performance.


### [Prediction of Regulatory Regions Active in K562 Using Deep Learning(University Course Project)](https://github.com/OmidTorkan-msc/Bioinformatics-and-Genomics.git)

This project focuses on predicting active regulatory regions in the K562 cell line using advanced deep learning techniques. Regulatory regions, including promoters and enhancers, are critical for gene expression. The task involves identifying active promoters versus inactive promoters (APvsIP) and active enhancers versus inactive enhancers (AEvsIE). Various deep learning models were developed and analyzed to address this problem, including perceptrons, feed-forward neural networks, convolutional neural networks (CNNs), and multi-model architectures.

**Key Contributions:**

**1.	Deep Learning Models:**

o	Perceptron: Linear modeling used as a baseline.

o	Feed-Forward Neural Network: A multi-layer architecture designed to extract complex nonlinear patterns.

o	Convolutional Neural Network: Applied to sequence data to capture localized patterns through filters and max-pooling layers.

o	Multi-Model Neural Network: Combined features from feed-forward and convolutional networks for enhanced prediction accuracy.

**2.	Data Sources:**

o	Epigenomic Data: Active/inactive regulatory regions annotated.

o	FANTOM Database: Provided labels for promoter and enhancer activity.

o	UCSC Genome Browser: Retrieved genome sequences and annotations.

**3.	Data Preprocessing:**

o	KNN Imputation: Addressed missing values by averaging nearest neighbors.

o	Normalization: Used robust scaling to manage outliers and standardize data.

o	Correlation Analysis: Applied Pearson and Spearman tests to identify and drop uncorrelated or redundant features, ensuring cleaner datasets.

**4.	Experimental Setup:**

o	Tasks: APvsIP and AEvsIE classification.

o	Evaluation: Utilized stratified holdout to balance classes and mitigate biases during training and testing.

Achievements:

•	Implemented and compared the effectiveness of different neural network architectures.

•	Identified and removed irrelevant or highly correlated features, improving model performance.

•	Leveraged epigenomic datasets to provide insights into active regulatory regions critical for gene regulation.

This project demonstrates the utility of deep learning in bioinformatics, particularly for genomic regulatory region prediction. By combining rigorous preprocessing, innovative modeling techniques, and comprehensive evaluation, the models provide a strong foundation for further research in understanding gene regulation mechanisms.

## Other Projects

**- Business Process Engineering and Automation in Car-Sharing Service:** Designed BPMN models and automated workflows using BPEL4People.

**- Image Processing and Computer Vision:** Implemented image segmentation, edge detection, and Fourier Transform techniques.

**- Intelligent Systems and Applications:** Developed safety monitoring systems and real-time human tracking with RGB-D sensors.

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

- **Business Process Modeling and Automation**:  
  - BPMN (Business Process Model and Notation)  
  - BPEL4People (Business Process Execution Language for People)  
  - Process Automation & Workflow Management  
  - KPI Development and Tracking (User Growth, Car Bookings, Demand Forecasting)

- **Computer Vision and Image Processing**:  
  - OpenCV (Image Manipulation, Feature Extraction)  
  - Scikit-Image (Filtering, Segmentation)  
  - Image Segmentation and Edge Detection (e.g., Sobel, Canny)  
  - Morphological Operations (Erosion, Dilation)  
  - Histogram Equalization and Matching  
  - Fourier Transform (Frequency Domain Analysis)

- **Human Tracking and Safety Systems**:  
  - Vision-Based Tracking Systems (Stereo Cameras, High-Visibility Garments)  
  - Real-Time Tracking with RGB-D Sensors  
  - Wearable Technology for Safety (RFID, GPS, Sensor Networks)  
  - Collision Avoidance in Hazardous Environments

- **Stakeholder Communication**:  
  - Visual Documentation for Stakeholders (BPMN for communication)  
  - Collaboration between technical and non-technical teams


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
