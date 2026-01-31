import os
import glob
from typing import List, Tuple, Dict
import pandas as pd

# Original Paths (Configure these as needed or env vars)
RAVDESS_PATH = "content/RAVDESS DATASET"
IEMOCAP_PATH = "content/IEMCAP DATABASE"
MAD_PATH = "content/M.A.D. (MILITARY AUDIO DATASET)"

def make_ravdess_manifest(root_folder: str) -> pd.DataFrame:
    """
    RAVDESS Filename Example: 03-01-06-01-02-01-12.wav
    7th part is Actor (Speaker)
    3rd part is Emotion
    """
    data = []
    # 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    emotion_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', 
                   '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
    
    if not os.path.exists(root_folder):
        return pd.DataFrame(data, columns=['path', 'label', 'speaker_id', 'dataset'])

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.wav'):
                parts = file.split('-')
                if len(parts) >= 7:
                    emotion_code = parts[2]
                    speaker_code = parts[6].replace('.wav', '')
                    
                    if emotion_code in emotion_map:
                        data.append({
                            'path': os.path.join(root, file),
                            'label': emotion_map[emotion_code],
                            'speaker_id': f"RAVDESS_{speaker_code}",
                            'dataset': 'RAVDESS'
                        })
    return pd.DataFrame(data)

def make_iemocap_manifest(root_folder: str) -> pd.DataFrame:
    """
    IEMOCAP speakers are defined by Session (Ses01 contains specific actors).
    We will use the Session ID as a proxy for the Speaker Group if exact speaker ID isn't easily parsed,
    OR we parse the specific actor ID if available in filenames.
    Actually, strict separation by Session is the safest for IEMOCAP.
    There are 5 sessions. Ses01, Ses02, ...
    """
    data = []
    emotion_map = {'neu': 'neutral', 'hap': 'happy', 'sad': 'sad', 'ang': 'angry', 
                   'fea': 'fearful', 'sur': 'surprised', 'dis': 'disgust', 'fru': 'angry'} # Mapping frustration to angry for simplicity or keep separate
    
    if not os.path.exists(root_folder):
        return pd.DataFrame(data, columns=['path', 'label', 'speaker_id', 'dataset'])

    for session in range(1, 6):
        session_name = f'Session{session}'
        emo_eval_path = os.path.join(root_folder, session_name, 'dialog', 'EmoEvaluation')
        if not os.path.isdir(emo_eval_path): continue
        
        for txt_file in glob.glob(os.path.join(emo_eval_path, '*.txt')):
            with open(txt_file, 'r') as f:
                for line in f:
                    if line.startswith('['):
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            wav_file = parts[0].strip('[]')
                            emotion = parts[3]
                            if emotion in emotion_map:
                                # Construct Path
                                folder = wav_file.split('_')[0] + '_' + wav_file.split('_')[1]
                                wav_path = os.path.join(root_folder, session_name, 'sentences', 'wav', folder, f'{wav_file}.wav')
                                if os.path.exists(wav_path):
                                    # Speaker ID: The session is the primary rigorous split unit.
                                    # But technically Ses01F and Ses01M are different speakers.
                                    # Filename example: Ses01F_impro01_F000
                                    # The speaker is the prefix "Ses01F" or "Ses01M" usually roughly implied by the folder/file.
                                    # Safe bet: Use Session ID to ensure we hold out entirely new recording conditions/actors.
                                    data.append({
                                        'path': wav_path,
                                        'label': emotion_map[emotion],
                                        'speaker_id': f"IEMOCAP_{session_name}", 
                                        'dataset': 'IEMOCAP'
                                    })
    return pd.DataFrame(data)

def make_mad_manifest(root_folder: str) -> pd.DataFrame:
    """
    Assuming MAD follows RAVDESS-like structure based on previous code.
    """
    data = []
    emotion_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', 
                   '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
    
    if not os.path.exists(root_folder):
        return pd.DataFrame(data, columns=['path', 'label', 'speaker_id', 'dataset'])

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.wav'):
                parts = file.split('-')
                if len(parts) >= 7: # Assuming same structure as RAVDESS
                    emotion_code = parts[2]
                    speaker_code = parts[6].replace('.wav', '')
                    if emotion_code in emotion_map:
                        data.append({
                            'path': os.path.join(root, file),
                            'label': emotion_map[emotion_code],
                            'speaker_id': f"MAD_{speaker_code}",
                            'dataset': 'MAD'
                        })
                elif len(parts) >= 3: # Fallback from RNN.py logic if structure is looser
                     emotion_code = parts[2]
                     if emotion_code in emotion_map:
                         # If no speaker ID in filename, use file hash or just "unknown"
                         # But to be safe, we might treat each file as independent if we really can't tell,
                         # or treat the whole folder as one speaker.
                         # Let's trust strict RAVDESS format for now, or use Root folder as speaker proxy.
                         speaker_proxy = os.path.basename(root)
                         data.append({
                            'path': os.path.join(root, file),
                            'label': emotion_map[emotion_code],
                            'speaker_id': f"MAD_{speaker_proxy}",
                            'dataset': 'MAD'
                        })

    return pd.DataFrame(data)

def load_combined_dataset() -> pd.DataFrame:
    df1 = make_ravdess_manifest(RAVDESS_PATH)
    df2 = make_iemocap_manifest(IEMOCAP_PATH)
    df3 = make_mad_manifest(MAD_PATH)
    
    df = pd.concat([df1, df2, df3], ignore_index=True)
    return df
