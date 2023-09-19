import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import librosa
from librosa.onset import onset_detect
import sounddevice as sd
import warnings
from AudioTools.AudioSample import AudioSample


class LoadAudio:

    def __init__(self, file, name, segment=None, target_duration=None, max_duration=None, min_duration=None,
                 detect_onset=False, verbose=False, epoch_duration=None, min_power=None, overlap=0,
                 target_sampling_rate=None):
        self.file = file
        self.name = name
        self.verbose = verbose
        self.audio_sample = None
        self.interval_times = None
        self.target_duration = target_duration
        self.target_sampling_rate = target_sampling_rate
        self.epoch_duration = epoch_duration
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.detect_onset = detect_onset
        self.min_power = min_power
        self.load_file()
        self.intervals = []
        self.epochs = []
        if epoch_duration is not None:
            self.overlap = overlap
            self.epoch()
        elif segment == "auto":
            self.auto_segment()
        elif type(segment) == int or type(segment) == float:
            self.discrete_segment(segment)

    def check_power(self):
        power = np.sum(np.abs(self.audio_sample.data) ** 2)
        return power

    def check_duration(self):
        """
          Check that the duration of the audio clip being loaded meets the minimum (minimum_duration) if set
          :return:
          True - if sample meets minimum duration requirements
          False - if sample does NOT meet minimum duration requirements
          """
        if (self.min_duration is not None) and (self.audio_sample.duration < self.min_duration):
            warnings.warn("DURATION WARNING: Audio clip has a duration of " + str(self.audio_sample.duration) +
                          " s the minimum duration is " + str(self.min_duration) + " s")
            return False
        return True

    def load_file(self):
        """
        Loads audio file using Librosa. If a target_length class variable is defined the data will be repeated to meet
        the target_length
        :return:
        """
        data, sampling_rate = librosa.load(self.file, sr=None)
        self.audio_sample = AudioSample(data=data, sampling_rate=sampling_rate)
        self.check_duration()
        if self.detect_onset:
            onset_index = onset_detect(y=data, sr=sampling_rate)
            if len(onset_index) > 0:
                data = data[onset_index[0]:-1]

        if (self.target_duration is not None) or (self.max_duration is not None):
            if (self.target_duration is not None):
                if len(data) < self.target_duration*sampling_rate:
                    data = np.resize(data, self.target_duration*sampling_rate)
                elif len(data) > self.target_duration * sampling_rate:
                    data = data[0:self.target_duration*sampling_rate]
            if self.max_duration is not None:
                if len(data) > self.max_duration * sampling_rate:
                    data = data[0:self.max_duration * sampling_rate]
                elif len(data) < self.max_duration * sampling_rate:
                    data = np.concatenate([data, np.zeros(self.max_duration * sampling_rate - len(data))])

            self.audio_sample = AudioSample(data=data, sampling_rate=sampling_rate)

        if self.verbose:
            print(self.audio_sample.num_channels)
            print("File successfully loaded")

    def plot_time_series(self, plot_intervals=False, db=False):
        """
        Plots loaded waveform and intervals corresponding to single words (if computed)
        :return:
        """
        time_seconds = np.linspace(0., self.audio_sample.duration, self.audio_sample.num_samples)
        if db:
            time_series = librosa.amplitude_to_db(self.audio_sample.data)
        else:
            time_series = self.audio_sample.data
        max_time_series_db = max(time_series)
        min_time_series_db = min(time_series)
        plt.figure()
        ax = plt.gca()
        plt.plot(time_seconds, time_series)
        plt.title('Time Series Waveplot')
        if self.interval_times is not None and plot_intervals:
            for row in self.interval_times:
                rect = patches.Rectangle((row[0]/self.audio_sample.sampling_rate, min_time_series_db),
                                         (row[1]/self.audio_sample.sampling_rate-row[0]/self.audio_sample.sampling_rate),
                                         (max_time_series_db - min_time_series_db),
                                         linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()


    def discrete_segment(self, interval_len=1):
        """
        Process a continuous audio recording into intervals (consisting of start and stop indices) of specified length
        (interval_len)

        :param interval_len: length of the interval in seconds
        :return: None
        """
        num_intervals = int(np.ceil(self.audio_sample.duration/interval_len))
        interval_num = 0
        self.interval_times = []
        while interval_num < num_intervals:
            interval_start = interval_num*self.audio_sample.sampling_rate
            interval_stop = min((interval_num+1)*self.audio_sample.sampling_rate, self.audio_sample.num_samples)
            self.intervals.append(AudioSample(data=self.audio_sample.data[interval_start:interval_stop],
                                              sampling_rate=self.audio_sample.sampling_rate,
                                              name="Interval_" + str(interval_num)))
            self.interval_times.append([interval_start, interval_stop])
            interval_num = interval_num+1
        self.interval_times = np.array(self.interval_times)
        if self.verbose:
            print("Segmented into " + str(num_intervals) + " intervals")

    def epoch(self):
        """
        Process a continuous audio recording into fixed length epochs (consisting of start and stop indices).
        Populates self.intervals a Numpy 2-D array of start and stop indices corresponding to each epoch
        in the recording

        :param min_silence_time: minimum amount of silence between consecutive intervals (in seconds)
        :param padding: padding to add to the start and end of interval (in seconds)
        :return: None
        """
        duration = self.audio_sample.duration
        num_epochs = int(duration/self.epoch_duration)
        epochs = []
        epoch_num = 0

        while True:
            interval_start = int(epoch_num*self.epoch_duration*self.audio_sample.sampling_rate -
                                 self.overlap*self.epoch_duration*epoch_num*self.audio_sample.sampling_rate)
            interval_stop = int((epoch_num+1)*self.epoch_duration*self.audio_sample.sampling_rate-
                            self.overlap*self.epoch_duration*epoch_num*self.audio_sample.sampling_rate)
            if interval_stop > self.audio_sample.num_samples:
                break
            epochs.append([interval_start, interval_stop])
            self.epochs.append(AudioSample(data=self.audio_sample.data[interval_start:interval_stop],
                                              sampling_rate=self.audio_sample.sampling_rate,
                                              name="Interval_" + str(epoch_num))) #TODO: Change this to a LoadAudio object
            epoch_num = epoch_num + 1


    def epoch_power_stats(self):
        powers = []
        for epoch in self.epochs:
            powers.append(epoch.calculate_power())
        avg_power = np.median(powers)
        std_power = np.std(powers)
        num_epochs = len(powers)
        print("Number of epochs = " + str(num_epochs))
        print("Average power of epochs = " + str(avg_power))
        print("Std power of epochs = " + str(std_power))
        #plt.hist(powers,bins=len(powers))
        #plt.show()
        return avg_power, std_power, num_epochs


    def auto_segment(self, min_silence_time=0.5, padding=0.05):
        """
        Process a continuous audio recording into intervals (consisting of start and stop indices) that correspond
        to single words. Populates self.intervals a Numpy 2-D array of start and stop indices corresponding to each word
        in the recording

        :param min_silence_time: minimum amount of silence between consecutive intervals (in seconds)
        :param padding: padding to add to the start and end of interval (in seconds)
        :return: None
        """
        raw_intervals = librosa.effects.split(self.audio_sample.data, top_db=40)
        corrected_intervals = []
        num_intervals = raw_intervals.shape[0]
        interval_num = 0
        while interval_num < num_intervals:
            corrected_interval = [0,0]
            if (num_intervals - interval_num) > 1:
                corrected_interval[0] = raw_intervals[interval_num,0]
                silence_time = raw_intervals[interval_num + 1, 0] - raw_intervals[interval_num, 1]
                if self.verbose:
                    print("Silence time (s): " + str(silence_time/self.audio_sample.sampling_rate))
                if silence_time > min_silence_time*self.audio_sample.sampling_rate:  # check for min_silence_time between intervals
                    corrected_interval[1] = raw_intervals[interval_num, 1]
                    interval_num = interval_num + 1
                else:
                    corrected_interval[1] = raw_intervals[interval_num+1, 1]  # combine intervals if time between intervals is less than min_silence_time
                    interval_num = interval_num + 2  # skip next interval
            else:
                corrected_interval = raw_intervals[interval_num, :]
            interval_start = corrected_interval[0] - int(padding*self.audio_sample.sampling_rate)
            interval_stop = corrected_interval[1] + int(padding*self.audio_sample.sampling_rate)
            corrected_intervals.append([interval_start, interval_stop])
            self.intervals.append(AudioSample(data=self.audio_sample.data[interval_start:interval_stop],
                                  sampling_rate=self.audio_sample.sampling_rate,
                                  name="Interval_" + str(interval_num)))
        self.interval_times = np.array(corrected_intervals)

    def play_intervals(self):
        """
        Plays audio from intervals using the sounddevice library. For debug use.
        :return:
        """
        for row in self.interval_times:
            sd.play(self.audio_sample.data[row[0]:row[1]], samplerate=self.audio_sample.sampling_rate, blocking=True)

