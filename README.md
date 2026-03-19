# Semiconductor Defect Detection using XGBoost and LSTM

## 📌 Overview
This project focuses on predicting defective outcomes in semiconductor manufacturing using the SECOM dataset. The objective is to detect rare failure events from high-dimensional sensor data.

A major challenge in this task is the **severe class imbalance**, where defective samples represent only a small portion of the dataset. This project compares traditional machine learning and deep learning approaches to understand their effectiveness under such conditions.

---

## 🎯 Objectives
- Build a robust baseline model using XGBoost  
- Handle class imbalance effectively  
- Apply threshold tuning for better defect detection  
- Explore time-series modeling using LSTM  
- Compare the performance of machine learning vs deep learning  

---

## 📊 Dataset
- Source: SECOM dataset  
- Samples: ~1500  
- Features: 590 sensor measurements  
- Target:
  - `0` → Normal (Pass)  
  - `1` → Defect (Fail)  

### ⚠️ Challenges
- High-dimensional feature space (590 features)  
- Many noisy and irrelevant features  
- Severe class imbalance  
- Limited dataset size  
- Weak temporal structure  

---

## ⚙️ Data Preprocessing
- Missing values handled using forward-fill and backward-fill  
- Feature scaling applied using MinMaxScaler  
- Feature selection applied (top important features from XGBoost)  
- Target converted to binary classification  
- Train-test split:
  - Stratified split for XGBoost  
  - Time-based split for LSTM  

---

## 🤖 Model 1: XGBoost (Baseline)

### Approach
- Tree-based model optimized for tabular data  
- Class imbalance handled using `scale_pos_weight`  
- Probability-based predictions  
- Threshold tuning applied to optimize F1-score  

### 📈 Results
- Accuracy: ~0.93  
- F1-score: **0.28**  

### 💡 Key Insight
Lowering the classification threshold significantly improves the detection of rare defect events.

---

## 🔄 Model 2: LSTM (Time-Series Approach)

### Approach
- Data transformed into sequences using sliding window  
- LSTM used to capture temporal dependencies  
- Threshold tuning applied  

### 📈 Results
- Accuracy: ~0.05  
- F1-score: ~0.10  

### ⚠️ Observations
The LSTM model collapsed into predicting all samples as defective:
Confusion Matrix:
[[ 0 293]
[ 0 16]]


- All defect samples were detected (recall = 1.0)  
- However, all normal samples were misclassified  
- Resulted in extremely low precision  

### 💡 Key Insight
The model exhibited **model collapse**, a condition where it predicts a single class due to imbalance and data limitations.

---

## ⚖️ Model Comparison

| Model     | Accuracy | F1-score | Behavior |
|----------|---------|---------|----------|
| XGBoost  | ~0.93   | **0.28** | Stable and balanced |
| LSTM     | ~0.05   | ~0.10    | Collapsed (predicts all defects) |

---

## 🧠 Key Findings

- Accuracy is misleading for imbalanced datasets  
- F1-score is more appropriate for defect detection  
- Threshold tuning significantly improves performance  
- XGBoost performs well on small, high-dimensional tabular data  
- LSTM struggles with limited data and noisy features  
- Deep learning models are highly sensitive to class imbalance  

---

## 🏭 Industrial Relevance

In semiconductor manufacturing, detecting defects is critical. Missing a defect can lead to significant financial loss.

- LSTM provides high sensitivity (recall = 1.0)  
- However, excessive false positives reduce practicality  
- XGBoost offers a better balance between detection and efficiency  

---

## 🚀 Final Conclusion

XGBoost outperforms LSTM in this study, providing a more stable and reliable solution for defect prediction.

While LSTM has the theoretical advantage of modeling temporal dependencies, it requires larger and cleaner datasets to perform effectively. In this case, the model failed to generalize and collapsed into predicting a single class.

---

## 💡 Key Takeaway

Model selection should be based on data characteristics, not model complexity.  
For small, high-dimensional industrial datasets, tree-based models are often more effective than deep learning approaches.

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- TensorFlow / Keras  

---

## 📁 Project Structure
├── data/
├── notebook/
│ └── defect_detection.ipynb
├── models/
├── README.md

---

## 📌 Future Work
- Advanced feature selection techniques  
- Dimensionality reduction (PCA)  
- Hyperparameter tuning  
- Testing alternative models (LightGBM, Transformer)  
- Collecting larger and cleaner time-series data  

---

## ✨ Author
Sayid Mufaqih
