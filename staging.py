"""Automatic sleep staging of in ear EEG data."""

import antropy as ant
import logging
import numpy as np
import pandas as pd
import mne
from mne.filter import filter_data
import scipy.signal as sp_sig
import scipy.stats as sp_stats
from scipy.integrate import trapezoid
from sklearn.preprocessing import robust_scale
from yasa import sliding_window  
from yasa import bandpower_from_psd_ndarray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yasa")

class InEarSleepStaging:
    def __init__(self, raw, eeg_name):
        # Type check
        assert isinstance(eeg_name, str), "`eeg_name` must be a string."

        # Validate Raw instance and load data
        assert isinstance(raw, mne.io.BaseRaw), "`raw` must be a MNE Raw object."
        sf = raw.info["sfreq"]
        ch_names = [eeg_name]
        ch_types = ["eeg"]
        # Keep only selected channels (creating a copy of Raw)
        raw_pick = raw.copy().pick(ch_names)

        # Downsample if sf != 100
        assert sf > 80, "Sampling frequency must be at least 80 Hz."
        if sf != 100:
            raw_pick.resample(100, npad="auto")
            sf = raw_pick.info["sfreq"]

        # Get data and convert to microVolts
        data = raw_pick.get_data(units=dict(eeg="uV"))

        # Extract duration of recording in minutes
        duration_minutes = data.shape[1] / sf / 60
        if duration_minutes < 5:
            msg = (
                "Insufficient data. A minimum of 5 minutes of data is recommended "
                "otherwise results may be unreliable."
            )
            logger.warning(msg)

        # Add to self
        self.sf = sf
        self.ch_names = ch_names
        self.ch_types = ch_types
        self.data = data

    # Representation methods (for developers)
    def __repr__(self):
            n_samples = self.data.shape[-1]
            duration = (n_samples / self.sf) / 60
            return (
                f"<SleepStaging | {len(self.ch_names)} x {n_samples} samples ({duration:.1f} minutes), "
                f"{self.sf} Hz>"
            )

    # String representation (for users)
    def __str__(self):
        return self.__repr__()

    def fit(self):
        """Extract features from data.
        Returns
        -------
        self : returns an instance of self.
        """
        #######################################################################
        # MAIN PARAMETERS
        #######################################################################

        # Bandpass filter
        freq_broad = (0.4, 30)
        
        # FFT & bandpower parameters
        win_sec = 5  # = 2 / freq_broad[0]
        sf = self.sf
        win = int(win_sec * sf) # Number of samples in window = 5 * sf
        kwargs_welch = dict(window="hamming", nperseg=win, average="median") # Welch’s method
        bands = [
            (0.4, 1, "sdelta"),
            (1, 4, "fdelta"),
            (4, 8, "theta"),
            (8, 12, "alpha"),
            (12, 16, "sigma"),
            (16, 30, "beta"),
        ]


        #######################################################################
        # CALCULATE FEATURES
        #######################################################################

        features = []

        # Filter the data
        dt_filt = filter_data(
            self.data[0, :],
            sf,
            l_freq=freq_broad[0],
            h_freq=freq_broad[1],
            verbose=False,
        )

        # Divide the data into epochs of 30 seconds
        times, epochs = sliding_window(dt_filt, sf=sf, window=30)

        # Describe the shape of each epoch
        hmob, hcomp = ant.hjorth_params(epochs, axis=1)
        feat = {
            "std": np.std(epochs, ddof=1, axis=1), # How much the signal deviates from the mean
            "iqr": sp_stats.iqr(epochs, rng=(25, 75), axis=1), # How much the signal deviates from the median
            "skew": sp_stats.skew(epochs, axis=1), # Where the signal is concentrated (left or right)
            "kurt": sp_stats.kurtosis(epochs, axis=1), # How strong extreme values are in the signal
            "nzc": ant.num_zerocross(epochs, axis=1), # How often the signal crosses zero
            "hmob": hmob, # How quickly the signal changes relative to its size (mobility)
            "hcomp": hcomp, # How complicated those changes are (complexity)
        }

        # How much activity exists at each frequency 
        # Describe each 30-second epoch in terms of its frequency content
        # Groups the frequency measurements into slow delta, fast delta, theta, alpha, sigma, and beta
        freqs, psd = sp_sig.welch(epochs, sf, **kwargs_welch)

        # Group these activities into frequency bands
        bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands)
        for j, (_, _, b) in enumerate(bands):
            feat[b] = bp[j]

        # Compare the bands
        delta = feat["sdelta"] + feat["fdelta"]
        feat["dt"] = delta / feat["theta"]
        feat["ds"] = delta / feat["sigma"]
        feat["db"] = delta / feat["beta"]
        feat["at"] = feat["alpha"] / feat["theta"]

        # Measure the overal signal power
        idx_broad = np.logical_and(
            freqs >= freq_broad[0],
            freqs <= freq_broad[1],
        )
        dx = freqs[1] - freqs[0]
        feat["abspow"] = trapezoid(psd[:, idx_broad], dx=dx)

        # Describe how regular or complex the signal is
        # Calculate entropy and fractal dimension features
        feat["perm"] = np.apply_along_axis(ant.perm_entropy, axis=1, arr=epochs, normalize=True)
        feat["higuchi"] = np.apply_along_axis(ant.higuchi_fd, axis=1, arr=epochs)
        feat["petrosian"] = ant.petrosian_fd(epochs, axis=1)

        # Convert to dataframe
        feat = pd.DataFrame(feat).add_prefix("eeg_")
        features.append(feat)


        #######################################################################
        # SMOOTHING & NORMALIZATION
        #######################################################################

        # Save features to dataframe
        features = pd.concat(features, axis=1)
        features.index.name = "epoch"

        # Apply centered rolling average (15 epochs = 7 min 30) - 7 previous, 7 next, 1 current
        # Triang: [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.,
        #          0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
        rollc = features.rolling(window=15, center=True, min_periods=1, win_type="triang").mean()
        rollc[rollc.columns] = robust_scale(rollc, quantile_range=(5, 95))
        rollc = rollc.add_suffix("_c7min_norm")

        # Now look at the past 2 minutes
        rollp = features.rolling(window=4, min_periods=1).mean()
        rollp[rollp.columns] = robust_scale(rollp, quantile_range=(5, 95))
        rollp = rollp.add_suffix("_p2min_norm")