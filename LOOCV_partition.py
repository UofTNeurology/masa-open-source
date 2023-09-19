import os
import pandas as pd
import numpy as np
import shutil
import time
import pyfastcopy

# Folder select
drive_letter = "C:/"
input_folder = drive_letter + "Users/MASA/Documents/Image Outputs (Local)/April 4/"
output_folder = drive_letter + "Users/MASA/Documents/Image Outputs (Local)/Fold (auto) April 4/"

# Make output folder if it doesn't exist
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
    os.mkdir(output_folder)
else:
    os.mkdir(output_folder)

# Open study log
xls = pd.ExcelFile('study_master_log7.xlsx')
study_log = xls.parse(xls.sheet_names[0], header=None, index_col=0)
study_log = study_log.to_dict()
study_log = study_log[1]

# Select directory and load all files in the directory #
fail_dir = input_folder + "Fail"
fails = []

# Find fails
with os.scandir(fail_dir) as it:
    for entry in it:
        if entry.name.endswith(".png") and entry.is_file():
            fails.append({"path": entry.path, "name": entry.name, "TOR-BSST Status": "Fail"})

# Find passes
pass_dir = input_folder + "Pass"
passes = []
with os.scandir(pass_dir) as it:
    for entry in it:
        if entry.name.endswith(".png") and entry.is_file():
            passes.append({"path": entry.path, "name": entry.name, "TOR-BSST Status": "Pass"})
all_files = np.concatenate((fails, passes), axis=0)

# All
subject_dict = {}
for entry in all_files:
    name = entry["name"]
    mrn = name[name.find(',')+1:name.find('-',name.find(','))].strip()
    date = name[0:name.find(',') ].strip()
    if mrn not in subject_dict:
        subject_dict.update({mrn: {"files": [entry["path"]], "study_id": study_log[(int(mrn) if mrn != "KL" else mrn)],
                                   "TOR-BSST Status": entry["TOR-BSST Status"], "date": date}})
    else:
        subject_dict[mrn]["files"].append(entry["path"])

for mrn in subject_dict:
    subject_dict[mrn] = subject_dict[mrn] | {"num_files": str(len(subject_dict[mrn]["files"]))}

df = pd.DataFrame.from_dict(subject_dict, orient='index')
df['date'] = pd.to_datetime(df['date'],  errors = 'coerce')
df.to_csv("df.csv", encoding='utf-8', index=True, columns=["TOR-BSST Status", "study_id", 'num_files' ])


print("Done")
print(df.iloc[0]["files"])
print(len(df))

folds_dir = output_folder
row_indices = range(0, len(df))
for i in row_indices:
    print("Processing fold " + str(i+1) + " of " + str(row_indices[-1]+1))
    tic = time.time()
    folder_name = os.path.join(folds_dir, "Fold " + (str(i) if i >= 10 else ("0" + str(i))))
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
        os.mkdir(folder_name)
    else:
        os.mkdir(folder_name)

    os.mkdir(folder_name + "/Train")
    os.mkdir(folder_name + "/Validation")

    os.mkdir(folder_name + "/Train/Fail")
    os.mkdir(folder_name + "/Train/Pass")
    os.mkdir(folder_name + "/Validation/Fail")
    os.mkdir(folder_name + "/Validation/Pass")
    toc = time.time()
    print("Mkdir duration = " + str(toc-tic))

    tic = time.time()
    for file in df.iloc[i]["files"]:
        shutil.copy(file, os.path.join(folder_name, "Validation/" + df.iloc[i]["TOR-BSST Status"]))
    toc = time.time()
    print("Validation file move duration = " + str(toc - tic))


    tic = time.time()
    for k in row_indices:
        dest = os.path.join(folder_name, "Train/" + df.iloc[k]["TOR-BSST Status"])
        if k == i:
            continue
        for file in df.iloc[k]["files"]:
            shutil.copy(file, os.path.join(folder_name, "Train/" + df.iloc[k]["TOR-BSST Status"]))

    toc = time.time()
    print("Train file move duration = " + str(toc - tic))
