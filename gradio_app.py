import gradio as gr
import pickle

# Load the vectorizer and model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_news(text):
    vect_text = vectorizer.transform([text])
    prediction = model.predict(vect_text)[0]
    return "Fake" if prediction == 1 else "Real"

iface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=5, label="Enter News Text"),
    outputs=gr.Label(label="Prediction"),
    title="Fake News Detection",
    description="Enter a news article to check if it is Fake or Real."
)

if __name__ == "__main__":
    iface.launch()
