# Named Entity Recognition (NER) for Recipe Data

## Project Overview

This project focuses on developing a Named Entity Recognition (NER) model using Conditional Random Fields (CRF) to extract key entities from raw recipe ingredient data. The goal is to classify words within ingredient lists into specific categories: `quantity`, `unit`, and `ingredient`. This structured output can then power various applications in recipe management, dietary tracking, and e-commerce.

## Business Objective

The primary business objective is to create a structured database of recipes and ingredients from unstructured text. This enables advanced features such as:
* Automated recipe parsing.
* Ingredient inventory management.
* Dietary analysis and nutritional tracking.
* E-commerce integration for ingredient shopping.

## Data Description

The dataset is provided in JSON format, with each entry containing:
* `input`: The raw ingredient string (e.g., "2 cups all-purpose flour").
* `pos`: Space-separated NER labels corresponding to each token in the `input` (e.g., "quantity unit ingredient ingredient").

The dataset contains 285 raw entries. After data validation and cleaning (removing 5 inconsistent rows), 280 entries were used for modeling.

## Methodology

### 1. Data Ingestion and Preparation
* Loaded JSON data into a pandas DataFrame.
* Tokenized `input` strings and `pos` labels into separate lists.
* Performed data validation to ensure alignment between input tokens and their corresponding POS tags, dropping inconsistent rows.
* Split the cleaned dataset into training (70%) and validation (30%) sets.

### 2. Exploratory Data Analysis (EDA)
* Analyzed the distribution of `ingredient`, `unit`, and `quantity` labels in the training data.
* Identified the top 10 most frequent ingredients (e.g., "powder", "Salt", "seeds") and units (e.g., "teaspoon", "cup", "tablespoon").

### 3. Feature Engineering
A comprehensive set of word-level and contextual features were engineered for each token to aid the CRF model:
* **Core Features:** Token itself (lowercase), lemma, SpaCy POS tag, dependency relation, shape, stop word status, digit presence, alphanumeric presence, hyphenation, slash presence, title case, uppercase, punctuation.
* **Improved Quantity & Unit Detection:** Features indicating if a token is a potential quantity (numeric patterns, keywords) or a unit (keywords).
* **Contextual Features:** Features of the previous and next tokens (token, POS tag) and indicators for beginning/end of sentence.

### 4. Model Training
* A Conditional Random Field (CRF) model from `sklearn_crfsuite` was used.
* The model was trained on the engineered features from the training dataset.

## Results and Evaluation

The model's performance was evaluated using precision, recall, F1-score, and confusion matrices on both training and validation datasets.

### Training Data Performance
* **Overall Accuracy:** 0.99
* **F1-scores:**
    * `ingredient`: 1.00
    * `quantity`: 0.99
    * `unit`: 0.99
* The model showed near-perfect learning on the training data.

### Validation Data Performance
* **Overall Accuracy:** 0.99
* **F1-scores:**
    * `ingredient`: 1.00
    * `quantity`: 0.98
    * `unit`: 0.98
* The model generalized very well to unseen data.

### Confusion Matrix Insights (Validation Data)
A detailed error analysis revealed minor misclassifications primarily between `quantity` and `unit` labels:
* 10 `quantity` tokens were misclassified as `unit`.
* 6 `unit` tokens were misclassified as `quantity`.
`ingredient` labels were perfectly classified on the validation set.

## Insights and Outcomes

The CRF model successfully identifies key entities in recipe data with high accuracy. The primary challenge lies in distinguishing between quantities and units due to their frequent adjacency and similar structural patterns. The robust feature engineering contributed significantly to the model's strong performance.

## Suggestions for Future Improvement

* **Advanced Feature Engineering:**
    * Implement more sophisticated regex for complex numeric patterns (e.g., ranges, mixed fractions).
    * Explore character-level embeddings to capture sub-word nuances.
* **Data Augmentation:** Expand the dataset with more diverse examples, especially for ambiguous quantity/unit scenarios and rare units.
* **Post-processing Rules:** Introduce rule-based corrections to resolve common quantity/unit misclassifications based on observed patterns (e.g., if a numeric token is followed by a known unit, enforce `quantity` tag).
* **Deep Learning Models:** Investigate more advanced models like Bi-LSTM-CRF for potentially higher accuracy and better handling of sequential dependencies.

## How to Run (High-Level)

1.  **Prerequisites:** Ensure you have Python and necessary libraries (`pandas`, `spacy`, `sklearn-crfsuite`, `joblib`) installed.
2.  **Data:** Place `ingredient_and_quantity.json` in the appropriate directory.
3.  **Execution:** Run the Jupyter Notebook (`Identifying_Key_Entities_Recipe_Data_Starter_NitishNarayanan.ipynb`) to perform data loading, preprocessing, feature engineering, model training, and evaluation.
