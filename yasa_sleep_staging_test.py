# Prerequisites
import yasa
# edf_path = yasa.fetch_sample("night_young.edf")
# hypno_path = yasa.fetch_sample("night_young_hypno.csv")
edf_path = "database/EDF Score tech 1 6-9-26.edf"

# Data loading and preprocessing
import mne
raw = mne.io.read_raw_edf(edf_path, preload=True)
print(raw.ch_names)

# # remove EOG, EMG, and EKG channels
# raw.drop_channels(["ROC-A1", "LOC-A2", "EMG1-EMG2", "EKG-R-EKG-L"])
# chan = raw.ch_names
# print(chan)

# # Downsample and filtering
print(raw.info["sfreq"]) #1024.0Hz
# raw.resample(100)
# sf = raw.info["sfreq"]
# print(sf) # 100Hz
# raw.filter(0.3, 45)
data = raw.get_data(units="uV")
print(data.shape)


# Automatic sleep staging
import matplotlib.pyplot as plt
sls = yasa.SleepStaging(raw, eeg_name="LUEER-RUEER")
hypno_pred = sls.predict()  # Returns a yasa.Hypnogram
yasa.plot_hypnogram(hypno_pred);  # Plot
plt.show()