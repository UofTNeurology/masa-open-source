import librosa.display
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import io
import speech_recognition
from scipy.io.wavfile import write
import scipy.stats
import pywt
import librosa.filters
from PIL import Image
import jax.numpy as jnp
from .superlet_og import superlet

class ProcessAudio:
    """
    A class used to process audio data and compute various audio features and visual representations.

    ...

    Attributes
    ----------
    data : ndarray
        The audio data to be processed.
    sampling_rate : int
        The sampling rate of the audio data.
    hop_length : int, optional
        The hop length for STFT, by default 512.
    n_fft : int, optional
        The FFT window size, by default 2048.
    n_mels : int, optional
        The number of Mel bands, by default 512.
    # ... (rest of the attributes)

    Methods
    -------
    compute_stft():
        Computes the Short-Time Fourier Transform (STFT) of the audio data.
    compute_cwt():
        Computes the Continuous Wavelet Transform (CWT) of the audio data.
    # ... 
    """

    def __init__(self, data, sampling_rate, hop_length=512,  n_fft=2048, n_mels=512):
        """
        Initializes the ProcessAudio object with the specified parameters.

        Parameters
        ----------
        data : ndarray
            The audio data to be processed.
        sampling_rate : int
            The sampling rate of the audio data.
        hop_length : int, optional
            The hop length for STFT, by default 512.
        n_fft : int, optional
            The FFT window size, by default 2048.
        n_mels : int, optional
            The number of Mel bands, by default 512.
        """
        self.data = data
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.mel_spectogram_db = None
        self.norm_log_mel = None
        self.norm_mel = None
        self.cwt_freqs = None
        self.cwt_power = None
        self.cwt_time = None
        self.cwt_scales = None
        self.cwt_coeffs = None
        self.superlet_normalized = None
        self.log_superlet_data_norm = None
        self.transcription = None

    def compute_stft(self):
        """
        Computes the Short-Time Fourier Transform (STFT) of the audio data and stores the result in the stft attribute.
        """
        self.stft = librosa.amplitude_to_db(np.abs(librosa.stft(self.data)))

    def compute_cwt(self):
        """
        Computes the Continuous Wavelet Transform (CWT) of the audio data.

        The method configures the wavelet transform parameters, computes the CWT, and stores the result in class attributes.
        """
        wavelet = 'morl'  # wavelet type: morlet
        sr = self.sampling_rate
        widths = np.arange(4, 512)  # scales for morlet wavelet
        dt = 1 / sr  # timestep difference

        frequencies = pywt.scale2frequency(wavelet, widths) / dt  # Get frequencies corresponding to scales
        
        # Create a filter to select frequencies between 80Hz and 5KHz
        upper = ([x for x in range(len(widths)) if frequencies[x] > 5000])[-1]
        lower = ([x for x in range(len(widths)) if frequencies[x] < 80])[0]
        widths = widths[upper:lower]  # Select scales in this frequency range
        self.cwt_freqs = pywt.scale2frequency(wavelet, widths) / dt

        # Compute continuous wavelet transform of the audio numpy array
        self.cwt_coeffs, self.cwt_scales = pywt.cwt(self.data, widths, wavelet=wavelet, sampling_period=dt)
        self.cwt_power = (np.abs(self.cwt_coeffs)) ** 2
        p_max = self.cwt_power.max()  # maximum power, used for normalization in all plots to this max value
        self.cwt_time = np.arange(self.data.shape[0]) / sr


    def compute_mel_spectrogram(self):
        """
        Computes the Mel Spectrogram of the audio data.

        The method normalizes the audio data, computes the Mel Spectrogram, and stores the result in class attributes.
        """
        audio_data = self.data
        audio_data = librosa.util.normalize(audio_data) # peak normalize audio signal (for some reason librosa load doesn't do this)



        mel_spectogram = librosa.feature.melspectrogram(y=audio_data, sr=self.sampling_rate, n_fft=self.n_fft,
                                                        hop_length=self.hop_length, n_mels=self.n_mels, fmin=50,
                                                        fmax=self.sampling_rate/2, window='hamming')
                                                        # n_fft not specified - default = 2048, default n_mels=128
                                                            #  fmin=20, fmax=3400 #March 31 - 1024 n_fft, 512 hop_length

        self.mel_spectogram_db = librosa.power_to_db(mel_spectogram, ref=np.max)
        self.mel_spectogram_db_z = self.compute_z_score(self.mel_spectogram_db)

        # Colour normalization
        # Normalize the Mel spectrogram to the range [0, 1]
        self.norm_mel = (mel_spectogram - np.min(mel_spectogram) / np.ptp(mel_spectogram))
        self.norm_log_mel = (self.mel_spectogram_db - np.min(self.mel_spectogram_db)) / \
                     (np.max(self.mel_spectogram_db) - np.min(self.mel_spectogram_db))

    def compute_z_score(self, array):
        """
        Computes the z-score normalization of the given array.

        Parameters
        ----------
        array : ndarray
            The array to be normalized.

        Returns
        -------
        z_score : ndarray
            The z-score normalized array.
        """
        array_min = np.min(array)
        array_max = np.max(array)
        array_mean = np.mean(array)
        array_std = np.std(array)
        z_score = (array - array_mean)/(array_std + np.finfo(float).eps)
        return z_score

    def plot_stft(self):
        """
        Plots the Short-Time Fourier Transform (STFT) of the audio data.

        The method assumes that the STFT has already been computed and is stored in the stft attribute.
        """
        librosa.display.specshow(self.stft, sr=self.sampling_rate, hop_length=self.hop_length,
                                 x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.show()

    def compute_mfcc(self):
        """
        Computes the Mel Frequency Cepstral Coefficients (MFCCs) of the audio data and stores the result in the x_mfccs attribute.
        """
        self.x_mfccs = librosa.feature.mfcc(y=self.data, sr=self.sampling_rate, n_mfcc=20)

    def plot_mfccs(self):
        """
        Plots the Mel Frequency Cepstral Coefficients (MFCCs) of the audio data.

        The method assumes that the MFCCs have already been computed and are stored in the x_mfccs attribute.
        """
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(self.compute_z_score(self.x_mfccs), sr=self.sampling_rate, x_axis="time")
        plt.colorbar()
        plt.show()

    def compute_superlet(self, min_order=1, max_order=30, adaptive = True):
        """
        Computes the Superlet transform of the audio data.

        Parameters
        ----------
        min_order : int, optional
            The minimum order for the Superlet transform, by default 1.
        max_order : int, optional
            The maximum order for the Superlet transform, by default 30.
        adaptive : bool, optional
            Whether or not to use adaptive order selection, by default True.

        The method computes the Superlet transform and stores the result in class attributes.
        """
        freqs =np.linspace(80, 5000, 512)
        superlet_data = superlet(self.data, self.sampling_rate, freqs, order_max=max_order, order_min=min_order,adaptive=adaptive)
        # take amplitude of superlet
        superlet_data = np.abs(superlet_data)
        # Normalize superlet in range 0-1
        superlet_data = (superlet_data - np.min(superlet_data)) / (np.max(superlet_data) - np.min(superlet_data))
        self.superlet_normalized = superlet_data
        # Apply the log normalization, add small constant to avoid taking log of zero
        # Avoid taking log of zero by adding a small constant
        epsilon = 1e-10
        log_superlet_data = np.log(superlet_data + epsilon)

        # Normalize log transformed data in range 0-1
        self.log_superlet_data_norm = (log_superlet_data - np.min(log_superlet_data)) / (np.max(log_superlet_data) - np.min(log_superlet_data))

    def plot_mel_spectogram(self):
        """
        Plots the Mel Spectrogram of the audio data.

        The method assumes that the Mel Spectrogram has already been computed and is stored in the mel_spectogram_db attribute.
        """
        fig, ax = plt.subplots()
        if self.mel_spectogram_db is None:
            self.compute_mel_spectrogram()
        img = librosa.display.specshow(self.mel_spectogram_db, x_axis='time', y_axis='mel',
                                       sr=self.sampling_rate, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        plt.show()

    def save_result_to_file(self, file_dir="", name="audio_file", output_type='mel'):
        """
        Saves the computed audio features to a file based on the specified output type.

        Parameters
        ----------
        file_dir : str, optional
            The directory where the files should be saved, by default an empty string indicating the current directory.
        name : str, optional
            The base name for the saved files, by default "audio_file".
        output_type : str, optional
            The type of audio feature to save, by default 'mel'.
        """
        if output_type == 'mel':
            self.save_mel_spectogram_to_file(target_dir=file_dir, name=name)
        elif output_type == 'cwt':
            self.save_cwt_to_file(target_dir=file_dir, name=name)
        elif output_type == 'bw_mel':
            self.save_bw_mel_spectogram_to_file(target_dir=file_dir, name=name)
        elif output_type == 'audio':
            self.save_audio_clip_to_file(target_dir=file_dir, name=name, normalize=False)
        elif output_type == "audio_norm":
            self.save_audio_clip_to_file(target_dir=file_dir, name=name, normalize=True)
        elif output_type == "superlet":
            self.save_superlet_to_file(target_dir=file_dir, name=name)

    def save_audio_clip_to_file(self, target_dir="", name="audio_clip", normalize=None):
        """
        Saves the audio data to a file.

        Parameters
        ----------
        target_dir : str, optional
            The directory where the file should be saved, by default an empty string indicating the current directory.
        name : str, optional
            The name for the saved file, by default "audio_clip".
        normalize : bool, optional
            Whether to normalize the audio data before saving, by default None.
        """
        audio_data = self.data
        if normalize:
            audio_data = librosa.util.normalize(audio_data)
        scipy.io.wavfile.write(target_dir + name + '.wav', self.sampling_rate, audio_data)

    def save_mel_spectogram_to_file(self, target_dir="", name="audio_file"):
        """
        Saves the Mel Spectrogram to a file.

        Parameters
        ----------
        target_dir : str, optional
            The directory where the file should be saved, by default an empty string indicating the current directory.
        name : str, optional
            The name for the saved file, by default "audio_file".
        """
        if self.mel_spectogram_db is None:
            self.compute_mel_spectrogram()

        fig, ax = plt.subplots(figsize=(4, 4))
        dpi = 56
        plt.axis('off')

        img = librosa.display.specshow(self.norm_log_mel, x_axis='time', y_axis='mel',
                                       sr=self.sampling_rate, cmap='magma')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(target_dir + name + ".png", dpi=dpi)
        plt.close()

    def save_bw_mel_spectogram_to_file(self, target_dir="", name="audio_file"):
        """
        Saves the black and white Mel Spectrogram to a file.

        Parameters
        ----------
        target_dir : str, optional
            The directory where the file should be saved, by default an empty string indicating the current directory.
        name : str, optional
            The name for the saved file, by default "audio_file".
        """
        np.save(target_dir + name, self.norm_log_mel)

    def save_cwt_to_file(self,target_dir="",name="audio_file"):
        """
        Saves the Continuous Wavelet Transform (CWT) data to a file.

        Parameters
        ----------
        target_dir : str, optional
            The directory where the file should be saved, by default an empty string indicating the current directory.
        name : str, optional
            The name for the saved file, by default "audio_file".
        """
        if self.cwt_power is None:
            self.compute_cwt()

        fig, ax = plt.subplots(figsize=(2.24, 2.24))
        dpi = 100
        plt.axis('off')

        plt.imshow(np.flip(self.cwt_power,axis=0), cmap='magma', aspect='auto',
                   extent=[0, len(self.cwt_power[0]), min(self.cwt_scales), max(self.cwt_scales)])

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(target_dir + name + ".png", bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()

    def save_superlet_to_file(self, target_dir="", name="audio_file"):
        """
        Saves the Superlet transform data to a file.

        Parameters
        ----------
        target_dir : str, optional
            The directory where the file should be saved, by default an empty string indicating the current directory.
        name : str, optional
            The name for the saved file, by default "audio_file".
        """
        if self.log_superlet_data_norm is None:
            self.compute_superlet()

        fig, ax = plt.subplots(figsize=(2.24, 2.24))
        dpi = 100
        plt.axis('off')

        plt.imshow(self.log_superlet_data_norm, cmap='magma', aspect='auto',
                   extent=[0, len(self.log_superlet_data_norm[0]), 80, 5000])

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(target_dir + name + ".png", bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()

    def save_mfcc_to_file(self, ):
        """
        Saves the Mel Frequency Cepstral Coefficients (MFCCs) to a file.

        This method is yet to be implemented.
        """
        pass #TODO: implement this

    def transcribe(self):
        """
        Transcribes the audio data to text using the Sphinx library. Not fully implemented.

        Returns
        -------
        transcription : str
            The transcription of the audio data.
        """
        byte_io = io.BytesIO(bytes())
        write(byte_io, self.sampling_rate, (self.data * 32767).astype(np.int16))
        result_bytes = byte_io.read()
        audio_data = speech_recognition.AudioData(result_bytes, self.sampling_rate, 2)
        r = speech_recognition.Recognizer()
        # recognize speech using PocketSphinx - implemention of Google API etc. pending
        try:
            self.transcription = r.recognize_sphinx(audio_data)
            print("Sphinx thinks you said: " + self.transcription)
        except speech_recognition.UnknownValueError:
            print("Sphinx could not understand audio")
        except speech_recognition.RequestError as e:
            print("Sphinx error; {0}".format(e))

        return self.transcription


def save_mels_to_file(file_dir, name,  mel1, mel2, mel3, format='.npy'):
    """
    Saves the provided Mel spectrograms to file. Use for three-dimensional (RGB) mel-sepctrogram.

    Parameters
    ----------
    file_dir : str
        The directory where the files should be saved.
    name : str
        The base name for the saved files.
    mel1 : ndarray
        The first Mel spectrogram.
    mel2 : ndarray
        The second Mel spectrogram.
    mel3 : ndarray
        The third Mel spectrogram.
    format : str, optional
        The file format for saving, by default '.npy'.
    """
    if format == 'npy':
        concat_mel = np.stack((mel1, mel2, mel3), axis=2)
        np.save(file_dir + name, concat_mel)
    elif format == 'png':


        red_image = np.stack([mel1, np.zeros(np.shape(mel1)), np.zeros(np.shape(mel1))], axis=-1)
        green_image = np.stack([np.zeros(np.shape(mel2)), mel2, np.zeros(np.shape(mel2))], axis=-1)
        blue_image = np.stack([np.zeros(np.shape(mel3)), np.zeros(np.shape(mel3)), mel3], axis=-1)
        plt.imsave(file_dir + name + '_red.png', red_image)
        plt.imsave(file_dir + name + '_green.png', green_image)
        plt.imsave(file_dir + name + '_blue.png', blue_image)


        rgb_image = np.stack([mel1, mel2, mel3], axis=-1)

        # Save the RGB image to a file
        plt.imsave(file_dir + name + '.png', rgb_image)

def plot_mel(mel, sampling_rate=48000):
    """
    Plots the given Mel Spectrogram.

    Parameters
    ----------
    mel : ndarray
        The Mel Spectrogram to be plotted.
    sampling_rate : int, optional
        The sampling rate of the audio data, by default 48000.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    dpi = 56
    plt.axis('off')

    img = librosa.display.specshow(mel, x_axis='time', y_axis='mel',
                                   sr=sampling_rate, cmap='magma')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.show()
    plt.close()


def save_superlets_to_file(file_dir, name, superlets):
    """
    Saves the provided Superlet transforms to file.

    Parameters
    ----------
    file_dir : str
        The directory where the files should be saved.
    name : str
        The base name for the saved files.
    superlets : list of ndarray
        The Superlet transforms to be saved.
    """
    if format == 'npy':
        concat_superlet = np.stack((superlets[0], superlets[1], superlets[2]), axis=2)
        np.save(file_dir + name, concat_superlet)
    elif format == 'png':


        red_image = np.stack([superlets[0], np.zeros(np.shape(superlets[0])), np.zeros(np.shape(superlets[0]))], axis=-1)
        green_image = np.stack([np.zeros(np.shape(superlets[1])), superlets[1], np.zeros(np.shape(superlets[1]))], axis=-1)
        blue_image = np.stack([np.zeros(np.shape(superlets[2])), np.zeros(np.shape(superlets[2])), superlets[2]], axis=-1)
        plt.imsave(file_dir + name + '_red.png', red_image)
        plt.imsave(file_dir + name + '_green.png', green_image)
        plt.imsave(file_dir + name + '_blue.png', blue_image)


        rgb_image = np.stack([superlets[0], superlets[1], superlets[2]], axis=-1)

        # Save the RGB image to a file
        plt.imsave(file_dir + name + '.png', rgb_image)