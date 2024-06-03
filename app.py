from flask import Flask, request, jsonify, send_from_directory
import torch
from torchvision import transforms
from PIL import Image
import librosa
import numpy as np
from transformers import ViTForImageClassification
import os

# Initialize Flask app
app = Flask(__name__, static_url_path='')

# Define the label mapping used in your model
label_mapping = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fear': 5,
    'disgust': 6,
    'surprise': 7
}
label_mapping_inv = {v: k for k, v in label_mapping.items()}

# Function to extract mel spectrogram
def extract_mel_spectrogram(file_path, n_mels=128, hop_length=512):
    y, sr = librosa.load(file_path)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db, sr

# Function to preprocess a single audio file
def preprocess_single_audio(file_path, transform):
    mel_spec, sr = extract_mel_spectrogram(file_path)
    mel_spec = np.stack((mel_spec,) * 3, axis=-1)  # create 3-channel image
    mel_spec = Image.fromarray(mel_spec.astype(np.uint8))  # convert to PIL image
    mel_spec = transform(mel_spec)
    return mel_spec

# Load your trained model
model_path = 'models/VIT_model.pth'
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.num_labels = len(label_mapping)
model.classifier = torch.nn.Linear(model.classifier.in_features, model.num_labels)
model.load_state_dict(torch.load(model_path))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['file']
    audio_file_path = 'temp_audio.wav'
    audio_file.save(audio_file_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_image = preprocess_single_audio(audio_file_path, transform)
    test_image = test_image.unsqueeze(0)

    with torch.no_grad():
        test_image = test_image.to(device)
        outputs = model(test_image)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        predicted_emotion = label_mapping_inv[predicted_idx]
        predicted_probability = probabilities[predicted_idx].item()
        all_probabilities = {label_mapping_inv[i]: prob.item() for i, prob in enumerate(probabilities)}
    
    return jsonify({
        'emotion': predicted_emotion,
        'probability': predicted_probability,
        'all_probabilities': all_probabilities
    })

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
