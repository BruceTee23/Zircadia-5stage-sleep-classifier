import mne
import staging
import pandas as pd

# Get EDF file path
edf_path = "../database/EDF Score tech 1 6-9-26.edf"

# Load the EDF file
raw = mne.io.read_raw_edf(edf_path, preload=True)

sls = staging.InEarSleepStaging(raw, eeg_name = "LUEER-RUEER")

# Extract features
features = sls.get_features()

# Print the extracted features to csv
# features.to_csv("features.csv", index=False)
print(features)

