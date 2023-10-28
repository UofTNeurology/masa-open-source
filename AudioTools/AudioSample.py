import numpy as np
import warnings

class AudioSample:
    """
    A class representing an audio sample.

    Attributes
    ----------
    data : ndarray
        The audio data.
    sampling_rate : int
        The sampling rate of the audio data.
    min_power : float, optional
        The minimum power threshold for warnings, by default None.
    name : str, optional
        A name for the audio sample, by default "Audio".
    num_samples : int
        The number of samples in the audio data.
    duration : float
        The duration of the audio sample in seconds.
    num_channels : int
        The number of channels in the audio data.
    power : float
        The power of the audio sample, initially None.

    Methods
    -------
    calculate_power():
        Calculates and returns the power of the audio sample.
    check_power(min_power=None):
        Checks the power of the audio sample against a minimum threshold, and emits a warning if below threshold.
    """

    def __init__(self, data, sampling_rate, min_power=None, name="Audio"):
        """
        Initializes the AudioSample object with the specified parameters.

        Parameters
        ----------
        data : ndarray
            The audio data.
        sampling_rate : int
            The sampling rate of the audio data.
        min_power : float, optional
            The minimum power threshold for warnings, by default None.
        name : str, optional
            A name for the audio sample, by default "Audio".
        """
        self.data = data
        self.sampling_rate = sampling_rate
        self.num_samples = len(self.data)
        self.duration = self.num_samples / self.sampling_rate
        self.num_channels = 2 if len(self.data.shape) == 2 else 1
        self.name = name
        self.min_power = min_power
        self.power = None

    def calculate_power(self):
        """
        Calculates and returns the power of the audio sample.

        The power is calculated as the sum of the squares of the absolute values of the audio data.

        Returns
        -------
        float
            The power of the audio sample.
        """
        self.power = np.sum(np.abs(self.data) ** 2)
        return self.power

    def check_power(self, min_power=None):
        """
        Checks the power of the audio sample against a minimum threshold.

        If a minimum power threshold is provided, it updates the min_power attribute.
        If the power of the audio sample is below the threshold, a warning is emitted.

        Parameters
        ----------
        min_power : float, optional
            The minimum power threshold for warnings, by default None.

        Returns
        -------
        float
            The power of the audio sample.
        """
        if min_power is not None:
            self.min_power = min_power

        if self.power < self.min_power:
            warnings.warn(
                f"POWER WARNING: Audio sample has a power of {self.power}, the minimum power is {self.min_power}"
            )

        return self.power
