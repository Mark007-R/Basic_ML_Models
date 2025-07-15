🔭 Gamma vs. Hadron Particle Classification
Hey there! 👋
This project is all about classifying high-energy particles detected by the MAGIC gamma-ray telescope. Basically, we’re trying to figure out if a particle is a gamma ray or a hadron based on a bunch of numeric measurements.

🧠 What’s This All About?
Telescope data gives us 10 features per event — like size, width, asymmetry — and our job is to predict whether that event was caused by a gamma ray or a hadron.
Why? Because identifying gamma rays helps astrophysicists explore crazy cosmic phenomena like black holes and pulsars 🌌.

🧰 Tools & Libraries I Used
Python + Pandas, NumPy – for data loading and wrangling

Matplotlib & Seaborn – for visualization

Scikit-learn – for training ML models

imblearn – to handle class imbalance

Models used:

K-Nearest Neighbors

Logistic Regression

Naive Bayes

Support Vector Machine (SVM)

🔍 Quick Look at the Data
The dataset has ~19,000 rows.

Each row is a particle event with 10 features.

The class column is the target:

0 = Gamma

1 = Hadron

Note: Hadron events are fewer than gamma events, so we used oversampling to balance them out.

📊 EDA Highlights
Used a heatmap to check feature correlation.

Plotted histograms to see how each feature differs between gamma and hadron.

Found that some features like fAlpha, fDist, and fConc1 vary noticeably between classes.

🏋️ Model Training
I split the dataset into:

60% for training

20% for validation

20% for testing

Before training:

Scaled all the features

Applied oversampling only on the training set to fix class imbalance

🧪 Results
Here's how each model did on the validation set:

Model	Accuracy	F1 Score (Hadron)	Comments
KNN (k=3)	81%	0.73	Solid, but slower and not great for large data
Naive Bayes	74%	0.52	Not great with this data – poor recall on Hadron
Logistic Regression	80%	0.72	Balanced and easy to interpret
SVM	87%	0.81	Best overall performance ⭐️

🎯 Key Takeaways
Support Vector Machine (SVM) was the best model in this case.

Naive Bayes struggled, especially with correctly identifying hadrons.

Balancing the data with oversampling made a big difference.

Good preprocessing (scaling, encoding, splitting right) is half the battle in ML.

