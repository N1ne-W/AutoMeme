import joblib
import numpy as np

model = joblib.load(r"C:\Users\ecm22\Desktop\AutoMeme\first_model.pkl")

def predict_action(feature):
    # feature: (10,)  →  (1,10)
    feature = np.array(feature).reshape(1, -1)

    pred = model.predict(feature)[0]
    prob = model.predict_proba(feature)[0]

    return pred, prob
