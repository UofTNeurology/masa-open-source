import os
import pandas as pd
import numpy as np
import shutil
import time
import pyfastcopy

# Folder select
drive_letter = "C:/"
input_folder = drive_letter + "Users/MASA/Documents/Image Outputs (Local)/Spectrograms (500 ms) June 30/"
output_folder = drive_letter + "Users/MASA/Documents/Image Outputs (Local)/Random Fold Spectrograms (500 ms) June 30/"

# Make output folder if it doesn't exist
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
    os.mkdir(output_folder)
else:
    os.mkdir(output_folder)

# Open study log
xls = pd.ExcelFile('study_master_log12.xlsx')
study_log = xls.parse(xls.sheet_names[0])
#mrn_to_study_id = study_log.to_dict()
#mrn_to_study_id = study_log[1]
test_id = study_log[study_log["MRN"]==12345]["Study_ID"].values[0]


# Select directory and load all files in the directory #
fail_dir = input_folder + "Fail"
fails = []

# Find fails
with os.scandir(fail_dir) as it:
    for entry in it:
        if (entry.name.endswith(".png") or entry.name.endswith(".npy")) and entry.is_file():
            fails.append({"path": entry.path, "name": entry.name, "TOR-BSST Status": "Fail"})

# Find passes
pass_dir = input_folder + "Pass"
passes = []
with os.scandir(pass_dir) as it:
    for entry in it:
        if (entry.name.endswith(".png") or entry.name.endswith(".npy")) and entry.is_file():
            passes.append({"path": entry.path, "name": entry.name, "TOR-BSST Status": "Pass"})
all_files = np.concatenate((fails, passes), axis=0)

# All
subject_dict = {}
for entry in all_files:
    name = entry["name"]
    mrn = name[name.find(',')+1:name.find('-',name.find(','))].strip()
    date = name[0:name.find(',') ].strip()
    if mrn not in subject_dict:
        subject_dict.update({mrn: {"files": [entry["path"]], "Study_ID": study_log[study_log["MRN"]==(int(mrn) if mrn != "KL" else mrn)]["Study_ID"].values[0],
                                   "TOR-BSST Status": entry["TOR-BSST Status"]}})
    else:
        subject_dict[mrn]["files"].append(entry["path"])

for mrn in subject_dict:
    subject_dict[mrn] = subject_dict[mrn] | {"num_files": str(len(subject_dict[mrn]["files"]))}

df = pd.DataFrame.from_dict(subject_dict, orient='index')
df = df.rename_axis('MRN').reset_index()
df = pd.merge(df, study_log, how='left', on=['Study_ID'])
df = df.drop(df[df["Reject"]==True].index)
df.to_csv("df.csv", encoding='utf-8', index=True)


print("Done")
print(df.iloc[0]["files"])
print(len(df))
df = df.sample(frac=1)
df = df.reset_index(drop=True)


print("Processing")
tic = time.time()
folder_name = os.path.join(output_folder, "Fold 0")
if os.path.exists(folder_name):
    shutil.rmtree(folder_name)
    os.mkdir(folder_name)
else:
    os.mkdir(folder_name)

os.mkdir(folder_name + "/Train")
os.mkdir(folder_name + "/Test")

os.mkdir(folder_name + "/Train/Fail")
os.mkdir(folder_name + "/Train/Pass")

toc = time.time()
print("Mkdir duration = " + str(toc-tic))

split_participant = 40
tic = time.time()
for i in range(0,split_participant):
    for file in df.iloc[i]["files"]:
        shutil.copy(file, os.path.join(folder_name, "Train/" + df.iloc[i]["TOR-BSST Status"]))
    toc = time.time()
    print("Train file move duration = " + str(toc - tic))

tic = time.time()

for i in range(split_participant, df.shape[0]):
    os.mkdir(folder_name + "/Test/Test Participant " + str(i - split_participant))
    os.mkdir(folder_name + "/Test/Test Participant " + str(i - split_participant) + "/" + "Fail")
    os.mkdir(folder_name + "/Test/Test Participant " + str(i - split_participant) + "/" + "Pass")
    for file in df.iloc[i]["files"]:
        shutil.copy(file, os.path.join(folder_name, "Test/Test Participant " + str(i - split_participant) + "/" + df.iloc[i]["TOR-BSST Status"]))
    toc = time.time()
    print("Test file move duration = " + str(toc - tic))

print("Number of train participants = " + str(split_participant))
print("Number of test participants = " + str(i - split_participant + 1))




