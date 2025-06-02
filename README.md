# Natural Language Processing with Disaster Tweets

**Objective:**  
Build a machine learning model to predict whether a given tweet describes a real disaster (target = 1) or not (target = 0). We’ll walk through data exploration, preprocessing, model building (using an LSTM-based neural network), evaluation, and submission generation.

---

## Cell 1: Problem & Data Description

```markdown
# Problem & Data Description

**Challenge:** In times of crisis, Twitter can be an important communication channel.  
Disaster relief organizations want to automatically detect tweets reporting real disasters (e.g., “Forest fire in California”) versus non-disaster tweets (e.g., “Fireworks tonight!”).

**Objective:** Given a tweet’s text, predict a binary label:
- `1` if the tweet is about a real disaster  
- `0` otherwise  

**Data Files (provided by Kaggle):**  
- `train.csv` (~7.8 MB, 7 613 rows):  
  - `id`: unique identifier for each tweet  
  - `text`: the tweet’s content  
  - `location`: (may be blank) location from where the tweet was sent  
  - `keyword`: (may be blank) a highlighted keyword from the tweet  
  - `target`: binary label (1 = real disaster, 0 = not disaster)  

- `test.csv` (~1.0 MB, 3 263 rows): same columns except `target`  
- `sample_submission.csv`: template for submission (columns: `id,target`)  

**Key Challenges:**  
1. **Text Preprocessing:** Tweets often contain punctuation, emojis, URLs, and inconsistent casing.  
2. **Class Imbalance:** In this dataset, approximately 62% of tweets are non‐disaster (target=0), 38% are disaster (target=1).  
3. **Modeling Choice:** We’ll use a sequence‐based neural network (LSTM) to capture word order and context.  
4. **Evaluation Metric:** F1‐score (harmonic mean of precision and recall) is used on the test set for leaderboard scoring.

---
