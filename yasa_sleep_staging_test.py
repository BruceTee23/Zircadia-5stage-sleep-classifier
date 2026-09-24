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
psg_data = yasa.SleepStaging(raw, eeg_name="C3-M2")
psg_hypno = psg_data.predict()  # Returns a yasa.Hypnogram
zircadia_data = yasa.SleepStaging(raw, eeg_name="LUEER-RUEER")
zircadia_hypno = zircadia_data.predict()  # Returns a yasa.Hypnogram
# yasa.plot_hypnogram(hypno_pred);  # Plot
# plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_actual = psg_hypno.hypno
y_pred = zircadia_hypno.hypno

# Make sure both arrays have the same length
min_length = min(len(y_actual), len(y_pred))
y_actual = y_actual[:min_length]
y_pred = y_pred[:min_length]

labels = ["Wake", "N1", "N2", "N3", "REM"]
cm = confusion_matrix(y_actual, y_pred, labels=labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()

plt.title("Confusion Matrix: YASA vs Zircadia Sleep Staging")
plt.xlabel("Zircadia Predicted Labels")
plt.ylabel("PSG Actual Labels")
plt.show()