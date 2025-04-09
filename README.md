---
title: Mockml - House Price Predictor
emoji: 🏠
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: "5.23.3"
app_file: main.py
pinned: false
---
# 🏠 House Price Predictor

### This is **find the model which uses ML challenge!**
1. A **rule-based model** that uses a logical formula to estimate house prices.
2. A **machine learning model** trained on synthetic housing data.

Both models predict house prices based on three input features:
- **Square meters** of the house.
- **Number of bedrooms** in the house.
- **Distance** from the city center (in kilometers).

Along with **Price (in €)**, the app also provides a **confidence score** for each prediction, indicating how reliable the prediction is.

---

## 📊 Formula for Calculating Price

The **rule-based model** uses the following formula to calculate house prices:
```
price = 400 + 0.15 * sqmt + 200 * bedrooms - 5.5 * distance + bonuses - penalties
```

Where:
- `sqmt`: Square meters of the house (clipped between 15 and 200).
- `bedrooms`: Number of bedrooms (clipped between 1 and 6).
- `distance`: Distance from the city center in kilometers (clipped between 0.1 and 50).
- **Bonuses**: 
  - `+400` if `sqmt > 140` and `bedrooms > 3`.
- **Penalties**:
  - `-100` if `sqmt < 30` and `bedrooms < 2`.

The final price is clipped between **€400** and **€3000**.

---

## 🚀 How to Access the App Demo?

You can access the app demo on **Hugging Face Spaces**:

[👉 Try the House Price Predictor](https://huggingface.co/spaces/vinayin/mockml)


## 📚 Dependencies

This app uses the following Python libraries:
- `gradio`
- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`

Make sure to install them before running the app locally.

---

## 🧑‍💻 Run Locally

To run the app locally:
1. Clone this repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the app with `python main.py`.
4. Open the app in your browser at `http://127.0.0.1:7860`.

Enjoy a tiny challenge!
