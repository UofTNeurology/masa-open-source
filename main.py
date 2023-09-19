from AudioTools.LoadAudio import LoadAudio
from AudioTools.ProcessAudio import ProcessAudio,save_mels_to_file, save_superlets_to_file
# from AudioTools.superlets.superlets.superlets import superlet
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import shutil
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), "AudioTools"))

# Select directory and load all files in the directory #
path = "Audio Data/For Processing"
output_path = "Audio Data/Superlet Outputs/"
files = []
with os.scandir(path) as it:
    for entry in it:
        if entry.name.endswith(".wav") and entry.is_file():
            files.append({"path": entry.path, "name": entry.name})

# Make directory for output if they do not exist
if os.path.exists(output_path + "Fail"):
    shutil.rmtree(output_path + "Fail")
    os.mkdir(output_path + "Fail")
else:
    os.mkdir(output_path + "Fail/")

if os.path.exists(output_path + "Pass"):
    shutil.rmtree(output_path + "Pass")
    os.mkdir(output_path + "Pass")
else:
    os.mkdir(output_path + "Pass")

# Process files and intervals and save output images #
fail_sound_files_lens = []
pass_sound_files_lens = []

epoch_duration = 0.5
min_duration = 0.5 # epoch_duration
overlap = 0.5
min_power = 0
rejected_files_duration = []
rejected_epochs_power = []
power = []
num_epochs = 0
target_sampling_rate = 22050 # set to 'None' to preserve native sampling rate of audio file
output_type = 'superlet' # options: mel, bw_mel, audio, audio_norm, superlet

i=1
for file in files:

    print("Processing " + str(i) + " of " + str(len(files) + 1) + " (Filename = " + (str(file['name'])) + ")")
    audio = LoadAudio(file=file['path'], name=file['name'], segment=None, epoch_duration=epoch_duration,
                      detect_onset=True, overlap=overlap, min_duration=min_duration,
                      target_sampling_rate=target_sampling_rate)

    if audio.audio_sample.duration < audio.min_duration:
        rejected_files_duration.append(file)
        continue
    if audio.intervals:
        for interval in audio.intervals:
            Process1 = ProcessAudio(interval.data, sampling_rate=interval.sampling_rate)
            if (file['name'].find("- F-") > 0) or (file['name'].find("-F-") > 0) or (file['name'].find("- F -") > 0) or \
                    (file['name'].find("- F_") > 0) or (file['name'].find("-F_") > 0) or (file['name'].find("- F _") > 0):
                fail_sound_files_lens.append(interval.duration)
                Process1.save_result_to_file(file_dir=output_path + "Fail/",
                                             name=file['name'][:-4] + "_" + interval.name, output_type=output_type)
            else:
                pass_sound_files_lens.append(interval.duration)
                Process1.save_result_to_file(file_dir=output_path + "Pass/",
                                             name=file['name'][:-4] + "_" + interval.name, output_type=output_type)
    elif audio.epochs:
        avg_power, std_power, _ = audio.epoch_power_stats()
        power_min = max(avg_power - std_power*1.5 - 1, min_power)
        for epoch in audio.epochs:
            num_epochs = num_epochs + 1
            power.append(epoch.check_power(power_min))
            if epoch.power < power_min:
                rejected_epochs_power.append(epoch.name)
                continue
            if (file['name'].find("- F-") > 0) or (file['name'].find("-F-") > 0) or (file['name'].find("- F -") > 0)  or \
                    (file['name'].find("- F_") > 0) or (file['name'].find("-F_") > 0) or (file['name'].find("- F _") > 0):
                fail_sound_files_lens.append(epoch.duration)
                label = "Fail"
            else:
                pass_sound_files_lens.append(epoch.duration)
                label = "Pass"
            
            if output_type == "superlet":
                '''
                superlets = []
                Process1 = ProcessAudio(epoch.data, sampling_rate=epoch.sampling_rate, n_fft=2048,
                                        n_mels=512)
                for (i, (base_cycle, min_order, max_order)) in enumerate(zip([3, 5, 1], [1, 1, 5], [30, 30, 40])):
                    Process1.compute_superlet(base_cycle=base_cycle, min_order=min_order, max_order=max_order)
                    superlets.append(Process1.superlet_normalized)
                    
                save_superlets_to_file(file_dir=output_path + label + "/",
                                             name=file['name'][:-4] + "_" + epoch.name, superlets = superlets, format='png')
                '''
                Process1 = ProcessAudio(epoch.data, sampling_rate=epoch.sampling_rate, n_fft=2048, n_mels=512)
                Process1.save_result_to_file(file_dir=output_path + label + "/",
                                                name=file['name'][:-4] + "_" + epoch.name, output_type=output_type)
            elif output_type != "bw_mel":
                Process1 = ProcessAudio(epoch.data, sampling_rate=epoch.sampling_rate, n_fft=2048,
                                        n_mels=512)
                Process1.save_result_to_file(file_dir=output_path + label + "/",
                                             name=file['name'][:-4] + "_" + epoch.name, output_type=output_type)
            else:
                hop_length = 64
                Mel1 = ProcessAudio(epoch.data, sampling_rate=epoch.sampling_rate, hop_length=hop_length,  n_fft=2048,
                                        n_mels=376)
                Mel1.compute_mel_spectrogram()
                Mel2 = ProcessAudio(epoch.data, sampling_rate=epoch.sampling_rate, hop_length=hop_length, n_fft=1024,
                                        n_mels=376)
                Mel2.compute_mel_spectrogram()
                Mel3 = ProcessAudio(epoch.data, sampling_rate=epoch.sampling_rate, hop_length=hop_length, n_fft=4096,
                                    n_mels=376)
                Mel3.compute_mel_spectrogram()

                save_mels_to_file(file_dir=output_path + label + "/",
                                                     name=file['name'][:-4] + "_" + epoch.name, mel1=Mel1.norm_log_mel,
                                                     mel2=Mel2.norm_log_mel, mel3=Mel3.norm_log_mel, format='png')


    else:
        Process1 = ProcessAudio(audio.audio_sample.data, sampling_rate=audio.audio_sample.sampling_rate)
        if (file['name'].find("- F-") > 0) or (file['name'].find("-F-") > 0) or (file['name'].find("- F -") > 0) or \
                (file['name'].find("- F_") > 0) or (file['name'].find("-F_") > 0) or (file['name'].find("- F _") > 0):
            fail_sound_files_lens.append(audio.audio_sample.duration)
            Process1.save_result_to_file(file_dir=output_path + "Fail/",
                                         name=file['name'][:-4] + "_" + audio.audio_sample.name, output_type=output_type)
        else:
            pass_sound_files_lens.append(audio.audio_sample.duration)
            Process1.save_result_to_file(file_dir=output_path + "Pass/",
                                         name=file['name'][:-4] + "_" + audio.audio_sample.name, output_type=output_type)

    i = i + 1

# Plot histogram of lengths
pass_sound_files_lens = np.around(pass_sound_files_lens, decimals=1)
ax1 = plt.subplot(2, 1, 1)
plt.hist(pass_sound_files_lens, density=True)  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Data')
plt.title("Histogram of Pass Soundfile Lengths")

ax1 = plt.subplot(2, 1, 2)
fail_sound_files_lens = np.around(fail_sound_files_lens, decimals=1)
plt.hist(fail_sound_files_lens, density=True)  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Data')
plt.title("Histogram of Fail Soundfile Lengths")
plt.show()

print("Rejected files: ")
print(rejected_files_duration)
print("Percentage files rejected = " + str((len(rejected_files_duration) / (len(files) + 1))*100))

print(np.mean(power))
print(np.std(power))
print(np.mean(power) - np.std(power))
print("Percentage epochs rejected = " + str((len(rejected_epochs_power) / (num_epochs))*100))
plt.hist(np.sort(power),bins=120)
plt.show()
print("Done")