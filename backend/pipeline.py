import numpy as np
import librosa
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

SAMPLE_RATE = 16000
N_MELS = 64
MAX_TIME_FRAMES = 128

class AudioFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that takes a list of file paths (or raw audio arrays) 
    and returns a flattened array of features for each.
    
    Actually, to use ColumnTransformer effectively with a Neural Network (RNN),
    we usually normalize per feature dimension (e.g., each mel bin).
    
    This extractor will output an array of shape (N_samples, N_MELS * MAX_TIME_FRAMES)
    or just (N_samples, N_MELS, MAX_TIME_FRAMES) but Sklearn expects 2D.
    
    To keep it compatible with Sklearn pipelines, we will flatten, apply scaler, 
    and then reshape in the Dataset/Loader class.
    """
    def __init__(self, sr=SAMPLE_RATE, n_mels=N_MELS, max_time=MAX_TIME_FRAMES):
        self.sr = sr
        self.n_mels = n_mels
        self.max_time = max_time

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        X: List or Array of file paths.
        Returns: Numpy array (Batch, Vectors)
        """
        features = []
        for path in X:
            if isinstance(path, str):
                wav, _ = librosa.load(path, sr=self.sr, mono=True)
            else:
                wav = path # Assume raw audio
                
            # Compute Mel
            mel = librosa.feature.melspectrogram(y=wav, sr=self.sr, n_mels=self.n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # Pad or Cut
            if mel_db.shape[1] < self.max_time:
                pad = np.zeros((self.n_mels, self.max_time - mel_db.shape[1]))
                mel_db = np.concatenate([mel_db, pad], axis=1)
            else:
                mel_db = mel_db[:, :self.max_time]
            
            # Flatten for StandardScaler (Time * Mels) or just Mean/Std over dataset?
            # Standard way: Flatten -> (N_Mels * Time)
            features.append(mel_db.flatten())
            
        return np.array(features)

def create_pipeline():
    """
    Creates a sklearn pipeline that:
    1. Extracts Spectrograms
    2. scales them
    """
    # Since our input is just a list of paths (1 column conceptually),
    # we can just use a Pipeline directly or ColumnTransformer if we had metadata features.
    # User specifically asked for ColumnTransformer.
    # We will assume input is a DataFrame-like structure where one column is 'path'.
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('mel_spectrogram', 
             AudioFeatureExtractor(sr=SAMPLE_RATE, n_mels=N_MELS, max_time=MAX_TIME_FRAMES), 
             0) # Apply to column 0 (paths)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    # We add a generic StandardScaler to normalize the extracted features.
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('feature_extraction', preprocessor),
        ('scaler', StandardScaler())
    ])
    
    return pipeline

if __name__ == "__main__":
    # Internal Test
    test_paths = ["test.wav"] # Dummy
    print("Pipeline defined.")
