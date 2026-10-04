import mne

# Get EDF file path
edf_path = "database/EDF Score tech 1 6-9-26.edf"

# Load the EDF file
raw = mne.io.read_raw_edf(edf_path, preload=True)

