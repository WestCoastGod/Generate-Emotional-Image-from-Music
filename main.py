import numpy as np
import librosa
import pandas as pd
from joblib import load
import torch
from GAN.Models.generator import Generator
import matplotlib.pyplot as plt

# Load the saved model
trained_model = load(
    r"C:\Users\cxoox\Desktop\AIST3110_Project\Music\music_model.joblib"
)
print("Model loaded successfully!")

# Path to the new music file
new_music_file = (
    r"C:\Users\cxoox\Desktop\AIST3110_Project\Music\music_data\audios\1456.mp3"
)


def mir(music_file):
    # Load the audio file
    y, sr = librosa.load(music_file, sr=None)

    # Extract features
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13
    )  # 13 DCT coefficients are usually enough for main features representation
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    def aggregate(feature_matrix):
        return np.concatenate(
            [np.mean(feature_matrix, axis=1), np.std(feature_matrix, axis=1)]
        )

    features = np.concatenate(
        [
            aggregate(mfccs),
            aggregate(chroma_stft),
            aggregate(spectral_contrast),
            aggregate(zero_crossing_rate),
            aggregate(
                np.array([tempo]).reshape(-1, 1)
            ),  # Reshape tempo to be 2D for aggregation
        ]
    )

    return features


# Step 1: Extract features from the new music file
new_features = mir(new_music_file)
features_array = np.array(new_features)

columns = []
for i in range(13):
    columns.append(f"mfcc_dct{i}_mean")
    columns.append(f"mfcc_dct{i}_std")
for i in range(12):
    columns.append(f"chroma_stft_chord{i}_mean")
    columns.append(f"chroma_stft_chord{i}_std")
for i in range(7):
    columns.append(f"spectral_contrast_frequency{i}_mean")
    columns.append(f"spectral_contrast_frequency{i}_std")
for i in range(1):
    columns.append(f"zero_crossing_rate_frame{i}_mean")
    columns.append(f"zero_crossing_rate_frame{i}_std")
columns.append("tempo_mean")
columns.append("tempo_std")

# Ensure new_features is reshaped to a 2D array with one row
new_features = np.array(new_features).reshape(1, -1)

# Create the DataFrame with the correct shape
new_features_df = pd.DataFrame(new_features, columns=columns)

# Step 3: Predict valence and arousal values using the trained model
predicted_va = trained_model.predict(new_features_df)


# Step 4: Display the results
predicted_valence = predicted_va[0][0]
predicted_arousal = predicted_va[0][1]

print(f"Predicted Valence: {predicted_valence}")
print(f"Predicted Arousal: {predicted_arousal}")


def generate_emotion_image(v, a, save_path=r"generated_image.jpg"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator().to(device)
    G.load_state_dict(
        torch.load(
            r"C:\Users\cxoox\Desktop\AIST3110_Project\GAN\Weights\generator_final.pth"
        )
    )
    G.eval()

    z = torch.randn(1, 100).to(device)
    v_tensor = torch.tensor([v]).float().to(device)
    a_tensor = torch.tensor([a]).float().to(device)

    with torch.no_grad():
        img = G(z, v_tensor, a_tensor, stage=5)
        img = img.squeeze().permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5 + 0.5) * 255  # [-1,1] → [0,255]

    # Show the image in a separate window
    plt.imshow(img.astype("uint8"))
    plt.axis("off")  # hide axes for a cleaner display
    plt.show()
    plt.imsave(save_path, img.astype("uint8"))


generate_emotion_image(
    v=predicted_valence,
    a=predicted_arousal,
    save_path=r"C:\Users\cxoox\Desktop\AIST3110_Project\generated_image.jpg",
)
