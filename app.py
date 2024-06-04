from flask import Flask, request, jsonify, send_from_directory
import torch
from torchvision import transforms
from PIL import Image
import librosa
import numpy as np
from transformers import ViTForImageClassification
import os
import boto3

app = Flask(__name__, static_url_path='')

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

model_path = 'models/VIT_model.pth'

# Ensure model directory exists
if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))

# Check for AWS credentials in environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

if not aws_access_key_id or not aws_secret_access_key:
    raise ValueError("AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")

# Download model from S3
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
s3.download_file('vit-model123', 'VIT_model.pth', model_path)

# Load your trained model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.num_labels = len(label_mapping)
model.classifier = torch.nn.Linear(model.classifier.in_features, model.num_labels)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
device = torch.device('cpu')
model.to(device)

@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['file']
    audio_file_path = 'temp_audio.wav'
    audio_file.save(audio_file_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def preprocess_single_audio(file_path, transform):
        y, sr = librosa.load(file_path)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = np.stack((mel_spec_db,) * 3, axis=-1)  # create 3-channel image
        mel_spec = Image.fromarray(mel_spec.astype(np.uint8))  # convert to PIL image
        mel_spec = transform(mel_spec)
        return mel_spec

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
