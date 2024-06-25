import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import *
import os
from scipy.signal import butter, sosfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = max(lowcut, 1) / nyq  # Ensure lowcut is at least 1 Hz
    high = min(highcut, fs/2 - 1) / nyq  # Ensure highcut is below Nyquist frequency
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def visualize_audio(audio_file, start_time, end_time, ax1, ax2, fmin, fmax):
    y, sr = librosa.load(audio_file, offset=start_time, duration=end_time-start_time)
    
    # Apply bandpass filter
    y_filtered = bandpass_filter(y, fmin, fmax, sr)
    
    # Plot the filtered waveform
    librosa.display.waveshow(y_filtered, sr=sr, ax=ax1)
    ax1.set_title(f'Filtered Waveform ({start_time:.2f}s - {end_time:.2f}s)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    
    # Compute and plot the spectrogram
    D = librosa.stft(y_filtered)
    DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
    ax2.set_ylim(fmin, fmax)  # Set the frequency range
    ax2.set_title(f'Filtered Spectrogram ({start_time:.2f}s - {end_time:.2f}s)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Frequency (Hz)')
    return img
st.set_page_config(layout="wide")
# Streamlit app
st.title('MP4 Audio Visualizer - Compare Two Ranges')

# File uploader
uploaded_file = st.file_uploader("Choose an MP4 file", type="mp4")

if uploaded_file is not None:
    st.video(uploaded_file)
    # Create temp directory if it doesn't exist
    if not os.path.exists("temp"):
        os.makedirs("temp")

    temp_path = "temp/temp.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    video = VideoFileClip(temp_path)
    video.audio.write_audiofile("temp/temp.mp3")

    # Get video duration
    duration = video.duration

    # Add range sliders for time range selection
    st.subheader("Select two ranges to compare:")
    range1 = st.slider("Range 1", 0.0, duration, (0.0, duration/2), 0.1)
    range2 = st.slider("Range 2", 0.0, duration, (duration/2, duration), 0.1)

    # Get the sample rate of the audio
    y, sr = librosa.load("temp/temp.mp3", duration=1)  # Load just 1 second to get the sample rate

    # Add frequency range filter
    st.subheader("Frequency Range Filter:")
    freq_range = st.slider("Frequency Range (Hz)", 20, sr//2, (20, sr//2), 10)
    fmin, fmax = freq_range

    # Create a figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    audio_file = "temp/temp.mp3"
    
    # Visualize Range 1
    img1 = visualize_audio(audio_file, range1[0], range1[1], ax1, ax2, fmin, fmax)
    
    # Visualize Range 2
    img2 = visualize_audio(audio_file, range2[0], range2[1], ax3, ax4, fmin, fmax)

    # Add colorbar
    # fig.colorbar(img1, ax=(ax2, ax4), label='Amplitude (dB)')

    # Adjust layout and display the plot
    plt.tight_layout()
    st.pyplot(fig)

    # Clean up temporary files
    video.close()
    os.remove(temp_path)
    os.remove("temp/temp.mp3")
else:
    st.write("Please upload an MP4 file to visualize its audio.")