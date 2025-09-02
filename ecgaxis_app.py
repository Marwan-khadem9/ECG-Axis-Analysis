import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import scipy.signal as sg
from scipy.signal import butter, filtfilt
import io
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="ECG Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === UTILITY FUNCTIONS ===

def butter_highpass_filter(data, cutoff, fs, order=4):
    """High-pass Butterworth filter"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered = filtfilt(b, a, data)
    return filtered

def padded_filter_reflect(signal, cutoff, fs, pad_len=100):
    """Pads the signal by reflection instead of flat value."""
    pad_start = signal[1:pad_len+1][::-1]
    pad_end = signal[-pad_len-1:-1][::-1]
    padded = np.concatenate([pad_start, signal, pad_end])
    filtered = butter_highpass_filter(padded, cutoff, fs)
    return filtered[pad_len:-pad_len]

def circular_median_deg(angles_deg):
    """Calculate circular median of angles in degrees"""
    angles_rad = np.deg2rad(angles_deg)
    # Convert to unit circle points
    x = np.cos(angles_rad)
    y = np.sin(angles_rad)
    # Compute mean angle
    mean_angle_rad = np.arctan2(np.median(y), np.median(x))
    # Convert back to degrees
    median_angle_deg = np.rad2deg(mean_angle_rad) % 360
    return median_angle_deg

# === PAN-TOMPKINS QRS DETECTION CLASS ===

class Pan_Tompkins_QRS:
    def __init__(self, fs, lowcut, highcut):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
    
    def butter_bandpass_filter(self, data):
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = sg.butter(5, [low, high], btype='band')
        y = sg.filtfilt(b, a, data)
        return y
    
    def band_pass_filter(self, signal):
        sig = signal.copy()
        for index in range(len(signal)):
            sig[index] = signal[index]
            if index >= 1:
                sig[index] += 2 * sig[index - 1]
            if index >= 2:
                sig[index] -= sig[index - 2]
            if index >= 6:
                sig[index] -= 2 * signal[index - 6]
            if index >= 12:
                sig[index] += signal[index - 12]
        result = sig.copy()
        for index in range(len(signal)):
            result[index] = -1 * sig[index]
            if index >= 1:
                result[index] -= result[index - 1]
            if index >= 16:
                result[index] += 32 * sig[index - 16]
            if index >= 32:
                result[index] += sig[index - 32]
        max_val = max(max(result), -min(result))
        result = result / max_val
        return result
    
    def derivative(self, signal):
        result = signal.copy()
        for index in range(len(signal)):
            result[index] = 0
            if index >= 1:
                result[index] -= 2 * signal[index - 1]
            if index >= 2:
                result[index] -= signal[index - 2]
            if index >= 2 and index <= len(signal) - 2:
                result[index] += 2 * signal[index + 1]
            if index >= 2 and index <= len(signal) - 3:
                result[index] += signal[index + 2]
            result[index] = (result[index] * self.fs) / 8
        return result
    
    def squaring(self, signal):
        result = signal.copy()
        for index in range(len(signal)):
            result[index] = signal[index] ** 2
        return result
    
    def moving_window_integration(self, signal):
        result = signal.copy()
        win_size = round(0.12 * self.fs)
        sum_val = 0
        for j in range(win_size):
            sum_val += signal[j] / win_size
            result[j] = sum_val
        for index in range(win_size, len(signal)):
            sum_val += signal[index] / win_size
            sum_val -= signal[index - win_size] / win_size
            result[index] = sum_val
        return result
    
    def solve(self, signal_df):
        input_signal = signal_df
        bpass = self.butter_bandpass_filter(input_signal.copy())
        der = self.derivative(bpass.copy())
        sqr = self.squaring(der.copy())
        mwin = self.moving_window_integration(sqr.copy())
        return bpass, der, sqr, mwin

# === HEART RATE DETECTION CLASS ===

class HeartRate:
    def __init__(self, bandpass_signal, mwi_signal, samp_freq):
        self.signal = bandpass_signal
        self.mwi = mwi_signal
        self.fs = samp_freq

        self.peaks = []
        self.r_locs = []
        self.probable_peaks = []

        self.SPKI = self.NPKI = self.SPKF = self.NPKF = 0
        self.Threshold_I1 = self.Threshold_I2 = 0
        self.Threshold_F1 = self.Threshold_F2 = 0
        self.T_wave = False

        self.RR1 = []
        self.RR2 = []
        self.RR_Average1 = 0
        self.RR_Low_Limit = self.RR_High_Limit = self.RR_Missed_Limit = 0

        self.win_150ms = round(0.15 * self.fs)
        self.result = []

    def approx_peaks(self):
        slopes = sg.fftconvolve(self.mwi, np.ones(25) / 25, mode='same')
        for i in range(round(0.5 * self.fs) + 1, len(slopes) - 1):
            if slopes[i - 1] < slopes[i] > slopes[i + 1]:
                self.peaks.append(i)

    def update_thresholds(self):
        self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
        self.Threshold_I2 = 0.5 * self.Threshold_I1
        self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
        self.Threshold_F2 = 0.5 * self.Threshold_F1
        self.T_wave = False

    def adjust_rr_limits(self, i):
        self.RR1 = np.diff(self.peaks[max(0, i - 8):i + 1]) / self.fs
        self.RR_Average1 = np.mean(self.RR1)
        rr_filtered = [rr for rr in self.RR1 if self.RR_Low_Limit < rr < self.RR_High_Limit]
        if i >= 8 and rr_filtered:
            self.RR2 = rr_filtered[-8:]
        RR_avg2 = np.mean(self.RR2) if self.RR2 else self.RR_Average1
        self.RR_Low_Limit = 0.92 * RR_avg2
        self.RR_High_Limit = 1.16 * RR_avg2
        self.RR_Missed_Limit = 1.66 * RR_avg2

    def detect(self, update_only=False):
        self.approx_peaks()

        for i, peak in enumerate(self.peaks):
            win = np.arange(max(0, peak - self.win_150ms), min(peak + self.win_150ms, len(self.signal) - 1))
            max_val = np.max(self.signal[win])
            max_idx = np.argmax(self.signal[win]) + win[0]
            self.probable_peaks.append(max_idx)

            if i == 0 or i >= len(self.probable_peaks):
                self._adjust_learning(i, max_idx)
            else:
                self.adjust_rr_limits(i)
                if self.RR_Average1 < self.RR_Low_Limit or self.RR_Average1 > self.RR_Missed_Limit:
                    self.Threshold_I1 /= 2
                    self.Threshold_F1 /= 2
                RRn = self.RR1[-1]
                self._searchback(i, peak, RRn)
                self._check_t_wave(i, peak, RRn)

            self.update_thresholds()

        self._refine_r_locs()
        if not update_only:
            return np.unique(self.result)
        if update_only:
            return []

    def _adjust_learning(self, i, idx):
        if self.mwi[idx] >= self.Threshold_I1:
            self.SPKI = 0.125 * self.mwi[idx] + 0.875 * self.SPKI
            if self.signal[i] > self.Threshold_F1:
                self.SPKF = 0.125 * self.signal[i] + 0.875 * self.SPKF
                self.r_locs.append(idx)
            else:
                self.NPKF = 0.125 * self.signal[i] + 0.875 * self.NPKF
        else:
            self.NPKI = 0.125 * self.mwi[idx] + 0.875 * self.NPKI
            self.NPKF = 0.125 * self.signal[i] + 0.875 * self.NPKF

    def _check_t_wave(self, i, peak, RRn):
        if self.mwi[peak] >= self.Threshold_I1:
            if 0.20 < RRn < 0.36 and i > 0:
                curr_slope = np.max(np.diff(self.mwi[peak - self.win_150ms//2: peak + 1]))
                prev_peak = self.peaks[i - 1]
                prev_slope = np.max(np.diff(self.mwi[prev_peak - self.win_150ms//2: prev_peak + 1]))
                if curr_slope < 0.5 * prev_slope:
                    self.T_wave = True
                    self.NPKI = 0.125 * self.mwi[peak] + 0.875 * self.NPKI
            if not self.T_wave:
                if self.signal[i] > self.Threshold_F1:
                    self.SPKI = 0.125 * self.mwi[peak] + 0.875 * self.SPKI
                    self.SPKF = 0.125 * self.signal[i] + 0.875 * self.SPKF
                    self.r_locs.append(self.probable_peaks[i])
                else:
                    self.SPKI = 0.125 * self.mwi[peak] + 0.875 * self.SPKI
                    self.NPKF = 0.125 * self.signal[i] + 0.875 * self.NPKF
        else:
            self.NPKI = 0.125 * self.mwi[peak] + 0.875 * self.NPKI
            self.NPKF = 0.125 * self.signal[i] + 0.875 * self.NPKF

    def _searchback(self, i, peak, RRn):
        if RRn > self.RR_Missed_Limit:
            win = self.mwi[peak - int(RRn * self.fs) + 1: peak + 1]
            coord = np.where(win > self.Threshold_I1)[0]
            if len(coord):
                x_max = coord[np.argmax(win[coord])]
                self.SPKI = 0.25 * self.mwi[x_max] + 0.75 * self.SPKI
                self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
                self.Threshold_I2 = 0.5 * self.Threshold_I1

                band_win = self.signal[max(0, x_max - self.win_150ms): min(len(self.signal), x_max)]
                coord_band = np.where(band_win > self.Threshold_F1)[0]
                if len(coord_band):
                    r_max = coord_band[np.argmax(band_win[coord_band])]
                    if self.signal[r_max] > self.Threshold_F2:
                        self.SPKF = 0.25 * self.signal[r_max] + 0.75 * self.SPKF
                        self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
                        self.Threshold_F2 = 0.5 * self.Threshold_F1
                        self.r_locs.append(r_max)

    def _refine_r_locs(self):
        win_200ms = round(0.2 * self.fs)
        for r in self.r_locs:
            coord = np.arange(max(0, r - win_200ms), min(r + win_200ms + 1, len(self.signal)))
            if len(coord):
                max_idx = coord[np.argmax(self.signal[coord])]
                self.result.append(max_idx)

    def warmup(self, passes=2):
        """Run multiple passes over the signal to stabilize internal thresholds."""
        for _ in range(passes):
            self.detect(update_only=True)

# === DATA LOADING FUNCTION ===

def load_dat_file(uploaded_file):
    """Load and parse .dat file"""
    try:
        # Read the file content
        content = uploaded_file.read().decode('utf-8')
        lines = content.split('\n')
        
        # Skip header and parse data
        data_lines = lines[1:]  # Skip first line (header)
        
        time_index = []
        lead1 = []
        lead2 = []
        lead3 = []
        
        for line in data_lines:
            if line.strip():  # Skip empty lines
                parts = line.strip().split()
                if len(parts) == 4:
                    t, l1, l2, l3 = parts
                    time_index.append(int(float(t)))
                    lead1.append(float(l1))
                    lead2.append(float(l2))
                    lead3.append(float(l3))
        
        return np.array(time_index), np.array(lead1), np.array(lead2), np.array(lead3)
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None, None, None

# === ANALYSIS FUNCTIONS ===

def process_ecg_signal(lead1, lead2, lead3, fs=200):
    """Process ECG signals through complete pipeline"""
    
    # Baseline wander removal
    cutoff = 1  # Hz
    filtered_lead1 = padded_filter_reflect(lead1, cutoff, fs)
    filtered_lead2 = padded_filter_reflect(lead2, cutoff, fs)
    filtered_lead3 = padded_filter_reflect(lead3, cutoff, fs)
    
    # Pan-Tompkins QRS detection
    qrs = Pan_Tompkins_QRS(fs, lowcut=5, highcut=35)
    bpass_1, der_1, sqr_1, mwin_1 = qrs.solve(filtered_lead1)
    bpass_2, der_2, sqr_2, mwin_2 = qrs.solve(filtered_lead2)
    bpass_3, der_3, sqr_3, mwin_3 = qrs.solve(filtered_lead3)
    
    # Heart Rate Detection
    hr1 = HeartRate(bpass_1, mwin_1, fs)
    hr2 = HeartRate(bpass_2, mwin_2, fs)
    hr3 = HeartRate(bpass_3, mwin_3, fs)
    
    # Warmup and detect
    hr1.warmup(passes=4)
    hr2.warmup(passes=4)
    hr3.warmup(passes=4)
    
    r1 = hr1.detect()
    r2 = hr2.detect()
    r3 = hr3.detect()
    
    return {
        'filtered': (filtered_lead1, filtered_lead2, filtered_lead3),
        'processed': (bpass_1, bpass_2, bpass_3),
        'r_peaks': (r1, r2, r3),
        'mwi': (mwin_1, mwin_2, mwin_3)
    }

def calculate_heart_rate(r_peaks, fs):
    """Calculate heart rate from R-peaks"""
    if len(r_peaks) < 2:
        return 0
    
    rr_intervals = np.diff(r_peaks) / fs  # Convert to seconds
    mean_rr = np.mean(rr_intervals)
    hr_bpm = 60 / mean_rr
    return hr_bpm

def calculate_ecg_axis(lead1_peaks, lead2_peaks, lead3_peaks, filtered_lead1, filtered_lead2, filtered_lead3):
    """Calculate ECG electrical axis"""
    
    def get_qrs_amplitude(signal, peaks):
        if len(peaks) == 0:
            return 0
        amplitudes = [signal[peak] for peak in peaks]
        return np.mean(amplitudes)
    
    # Get average QRS amplitudes for each lead
    amp1 = get_qrs_amplitude(filtered_lead1, lead1_peaks)
    amp2 = get_qrs_amplitude(filtered_lead2, lead2_peaks)
    amp3 = get_qrs_amplitude(filtered_lead3, lead3_peaks)
    
    # Calculate electrical axis using standard formulas
    # Simplified calculation - in practice, you might want more sophisticated methods
    if amp1 != 0 and amp2 != 0:
        axis_angle = np.rad2deg(np.arctan2(amp2, amp1))
        if axis_angle < 0:
            axis_angle += 360
        return axis_angle
    
    return 0

def plot_ecg_with_peaks(time_index, original_signals, filtered_signals, r_peaks_list, fs):
    """Plot ECG signals with detected R-peaks"""
    
    fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True, dpi=100)
    
    lead_colors = ['tab:blue', 'tab:orange', 'tab:green']
    lead_names = ['Lead I', 'Lead II', 'Lead III']
    
    time_index_in_secs = time_index / fs
    
    for i, (ax, original, filtered, rpeaks, name, color) in enumerate(zip(
        axs,
        original_signals,
        filtered_signals,
        r_peaks_list,
        lead_names,
        lead_colors
    )):
        ax.plot(time_index_in_secs, original, label='Original ' + name, color=color, alpha=0.4)
        ax.plot(time_index_in_secs, filtered, label='Filtered ' + name, color=color, linewidth=1.5)
        
        if len(rpeaks) > 0:
            ax.scatter(time_index_in_secs[rpeaks], filtered[rpeaks], 
                      color='red', marker='*', s=50, label='R-peaks')
        
        ax.set_ylabel("Amplitude (pixels)")
        ax.set_title(f"{name} - Original & Baseline Corrected with R-peaks")
        ax.grid(True)
        ax.legend(loc='upper right')
    
    axs[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    return fig

def plot_ecg_axis(axis_angle):
    """Plot ECG electrical axis visualization"""
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title('ECG Electrical Axis\n(0° = +X axis, Clockwise Positive)', pad=20)
    
    # Region definitions
    regions = {
        'Normal':  ('#a8e6cf', -30, 90),
        'LAD':     ('#ffd3b6', -90, -30),
        'RAD':     ('#ff8b94', 90, 180),
        'Extreme': ('#dcedc1', 180, 270)
    }
    
    # Draw axis regions
    for label, (color, start, end) in regions.items():
        theta1 = (-end) % 360
        theta2 = (-start) % 360
        wedge = Wedge((0, 0), 1, theta1, theta2, facecolor=color, edgecolor='k', alpha=0.3, label=label)
        ax.add_patch(wedge)
    
    # Draw unit circle
    circle = Circle((0, 0), 1, edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(circle)
    
    # Plot ECG axis arrow
    angle_rad = np.deg2rad(-axis_angle % 360)
    x = np.cos(angle_rad)
    y = np.sin(angle_rad)
    ax.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1,
              color='r', width=0.008, headwidth=3, headlength=5)
    ax.text(1.15 * x, 1.15 * y, f'{axis_angle:.1f}°', color='r', fontsize=12,
            ha='center', va='center', weight='bold')
    
    # Label cardinal angles
    label_pos = {
        '0°':   (1.15, 0),
        '90°':  (0, -1.15),
        '180°': (-1.15, 0),
        '-90°': (0, 1.15)
    }
    for text, (lx, ly) in label_pos.items():
        ax.text(lx, ly, text, ha='center', va='center', fontsize=10)
    
    # Clean up plot
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.legend(loc='upper left', fontsize=10, frameon=False)
    
    return fig

def classify_axis_deviation(axis_angle):
    """Classify ECG axis deviation"""
    # Normalize angle to 0-360 range
    angle = axis_angle % 360
    
    if -30 <= angle <= 90 or (angle >= 330):  # Normal axis
        return "Normal Axis", "#a8e6cf"
    elif -90 <= angle < -30 or (270 <= angle < 330):  # Left axis deviation
        return "Left Axis Deviation (LAD)", "#ffd3b6"
    elif 90 < angle <= 180:  # Right axis deviation
        return "Right Axis Deviation (RAD)", "#ff8b94"
    else:  # Extreme axis deviation
        return "Extreme Axis Deviation", "#dcedc1"

# === STREAMLIT APP ===

def main():
    st.title("ECG Axis Analysis Web App")
    st.markdown("---")
    
    # Sidebar for file upload and parameters
    st.sidebar.header("📁 File Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a .dat file", 
        type=['dat'],
        help="Upload an ECG .dat file for analysis"
    )
    
    st.sidebar.header("⚙️ Parameters")
    fs = st.sidebar.number_input("Sampling Frequency (Hz)", value=200, min_value=50, max_value=1000)
    
    if uploaded_file is not None:
        try:
            # Load data
            with st.spinner("Loading ECG data..."):
                time_index, lead1, lead2, lead3 = load_dat_file(uploaded_file)
            
            if time_index is not None:
                st.success(f"✅ Successfully loaded {len(time_index)} samples")
                
                # Display basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(time_index))
                with col2:
                    st.metric("Duration (sec)", f"{len(time_index)/fs:.2f}")
                with col3:
                    st.metric("Sampling Rate", f"{fs} Hz")
                
                # Process ECG signals
                with st.spinner("Processing ECG signals..."):
                    results = process_ecg_signal(lead1, lead2, lead3, fs)
                
                filtered_signals = results['filtered']
                r_peaks = results['r_peaks']
                
                # Calculate heart rate (using Lead II)
                heart_rate = calculate_heart_rate(r_peaks[1], fs)
                
                # Calculate ECG axis
                ecg_axis = calculate_ecg_axis(r_peaks[0], r_peaks[1], r_peaks[2], 
                                            filtered_signals[0], filtered_signals[1], filtered_signals[2])
                
                axis_classification, axis_color = classify_axis_deviation(ecg_axis)
                
                # Display Results
                st.markdown("## 📊 Analysis Results")
                
                # Metrics row
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Heart Rate (BPM)", f"{heart_rate:.1f}")
                with col2:
                    st.metric("ECG Electrical Axis", f"{ecg_axis:.1f}°")
                with col3:
                    st.markdown(f"**Axis Classification:**")
                    st.markdown(f'<div style="background-color: {axis_color}; padding: 10px; border-radius: 5px; text-align: center;">{axis_classification}</div>', 
                               unsafe_allow_html=True)
                
                # Tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["📈 ECG Signals with R-peaks", "🎯 Electrical Axis Visualization", "📋 Detection Details"])
                
                with tab1:
                    st.markdown("### ECG Signals with Detected R-peaks")
                    fig_ecg = plot_ecg_with_peaks(time_index, 
                                                 [lead1, lead2, lead3], 
                                                 filtered_signals, 
                                                 r_peaks, fs)
                    st.pyplot(fig_ecg)
                    
                    # R-peak counts
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Lead I R-peaks", len(r_peaks[0]))
                    with col2:
                        st.metric("Lead II R-peaks", len(r_peaks[1]))
                    with col3:
                        st.metric("Lead III R-peaks", len(r_peaks[2]))
                
                with tab2:
                    st.markdown("### ECG Electrical Axis Visualization")
                    fig_axis = plot_ecg_axis(ecg_axis)
                    st.pyplot(fig_axis)
                    
                    # Axis interpretation
                    st.markdown("#### Axis Interpretation:")
                    interpretations = {
                        "Normal Axis": "The electrical axis is within normal limits (-30° to +90°). This indicates normal ventricular depolarization.",
                        "Left Axis Deviation (LAD)": "The electrical axis is deviated to the left (-30° to -90°). This may indicate left ventricular hypertrophy, left bundle branch block, or other cardiac conditions.",
                        "Right Axis Deviation (RAD)": "The electrical axis is deviated to the right (+90° to +180°). This may indicate right ventricular hypertrophy, right bundle branch block, or pulmonary conditions.",
                        "Extreme Axis Deviation": "The electrical axis is in the extreme range (+180° to -90°). This is abnormal and may indicate severe cardiac pathology."
                    }
                    
                    st.info(interpretations.get(axis_classification, "Unknown axis classification."))
                
                with tab3:
                    st.markdown("### Detection Details")
                    
                    # Heart Rate Details
                    if len(r_peaks[1]) >= 2:
                        rr_intervals = np.diff(r_peaks[1]) / fs
                        st.markdown("#### Heart Rate Analysis (Lead II):")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Mean RR Interval", f"{np.mean(rr_intervals):.3f} sec")
                            st.metric("Min RR Interval", f"{np.min(rr_intervals):.3f} sec")
                        with col2:
                            st.metric("Max RR Interval", f"{np.max(rr_intervals):.3f} sec")
                            st.metric("RR Std Deviation", f"{np.std(rr_intervals):.3f} sec")
                    
                    # R-peak positions
                    st.markdown("#### R-Peak Positions:")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**Lead I R-peaks (samples):**")
                        if len(r_peaks[0]) > 0:
                            st.write(f"{list(r_peaks[0][:10])}{'...' if len(r_peaks[0]) > 10 else ''}")
                        else:
                            st.write("No R-peaks detected")
                    
                    with col2:
                        st.markdown("**Lead II R-peaks (samples):**")
                        if len(r_peaks[1]) > 0:
                            st.write(f"{list(r_peaks[1][:10])}{'...' if len(r_peaks[1]) > 10 else ''}")
                        else:
                            st.write("No R-peaks detected")
                    
                    with col3:
                        st.markdown("**Lead III R-peaks (samples):**")
                        if len(r_peaks[2]) > 0:
                            st.write(f"{list(r_peaks[2][:10])}{'...' if len(r_peaks[2]) > 10 else ''}")
                        else:
                            st.write("No R-peaks detected")
                    
                    # Processing pipeline visualization
                    if st.checkbox("Show Processing Pipeline Details"):
                        st.markdown("#### Signal Processing Pipeline:")
                        
                        # Show processing stages for Lead II
                        qrs_temp = Pan_Tompkins_QRS(fs, lowcut=5, highcut=35)
                        bpass_temp, der_temp, sqr_temp, mwin_temp = qrs_temp.solve(filtered_signals[1])
                        
                        fig_pipeline, axs_pipeline = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
                        
                        time_samples = np.arange(len(bpass_temp))
                        
                        axs_pipeline[0].plot(time_samples, bpass_temp, color='blue', label='Bandpass Filtered')
                        axs_pipeline[0].set_title("1. Bandpass Filtered Signal (5-35 Hz)")
                        axs_pipeline[0].set_ylabel("Amplitude")
                        axs_pipeline[0].grid(True)
                        axs_pipeline[0].legend()
                        
                        axs_pipeline[1].plot(time_samples, der_temp, color='orange', label='Derivative')
                        axs_pipeline[1].set_title("2. Derivative Signal")
                        axs_pipeline[1].set_ylabel("Amplitude")
                        axs_pipeline[1].grid(True)
                        axs_pipeline[1].legend()
                        
                        axs_pipeline[2].plot(time_samples, sqr_temp, color='red', label='Squared')
                        axs_pipeline[2].set_title("3. Squared Signal")
                        axs_pipeline[2].set_ylabel("Amplitude")
                        axs_pipeline[2].grid(True)
                        axs_pipeline[2].legend()
                        
                        axs_pipeline[3].plot(time_samples, mwin_temp, color='green', label='Moving Window Integrated')
                        axs_pipeline[3].set_title("4. Moving Window Integrated Signal")
                        axs_pipeline[3].set_ylabel("Amplitude")
                        axs_pipeline[3].set_xlabel("Sample Number")
                        axs_pipeline[3].grid(True)
                        axs_pipeline[3].legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig_pipeline)
                
                # Download results
                st.markdown("## 💾 Download Results")
                
                # Prepare results summary
                results_text = f"""ECG Analysis Results
===================
File: {uploaded_file.name}
Sampling Frequency: {fs} Hz
Total Samples: {len(time_index)}
Duration: {len(time_index)/fs:.2f} seconds

Heart Rate Analysis:
- Heart Rate (Lead II): {heart_rate:.2f} BPM
- R-peaks detected in Lead I: {len(r_peaks[0])}
- R-peaks detected in Lead II: {len(r_peaks[1])}
- R-peaks detected in Lead III: {len(r_peaks[2])}

ECG Electrical Axis:
- Axis Angle: {ecg_axis:.1f}°
- Classification: {axis_classification}

R-peak Positions (samples):
Lead I: {list(r_peaks[0])}
Lead II: {list(r_peaks[1])}
Lead III: {list(r_peaks[2])}
"""
                
                st.download_button(
                    label="📄 Download Analysis Report",
                    data=results_text,
                    file_name=f"ecg_analysis_{uploaded_file.name.split('.')[0]}.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.error("Please ensure the .dat file is properly formatted with time_index, lead1, lead2, lead3 columns.")
    
    else:
        # Welcome message and instructions
        st.markdown("""
        ## Welcome to the Web App! 
        
        This application processes ECG signals using the **Pan-Tompkins QRS detection algorithm** and provides:
        
        ### 📋 Features:
        - **Baseline Wander Removal**: Using reflective padding and Butterworth high-pass filtering
        - **QRS Complex Detection**: Pan-Tompkins algorithm with adaptive thresholds
        - **Heart Rate Calculation**: Based on RR intervals from Lead II
        - **ECG Electrical Axis**: Calculation and visualization with axis deviation classification
        - **Interactive Visualizations**: ECG signals with detected R-peaks and electrical axis plots
        
        ### Getting Started:
        1. Upload your `.dat` file using the sidebar
        2. Adjust sampling frequency if needed (default: 200 Hz)
        3. View the analysis results and visualizations
        4. Download the analysis report
        
        ---
        **Note**: This app implements the Pan-Tompkins QRS detection algorithm with baseline wander removal as described in our research documentation.
        """)
        # Show example of expected file format
        with st.expander("📄 View Expected File Format"):
            st.code("""
time_index  lead1       lead2       lead3
0           -0.025      0.125       0.150
1           -0.020      0.130       0.145
2           -0.015      0.135       0.140
3           -0.010      0.140       0.135
4           -0.005      0.145       0.130
...         ...         ...         ...
            """, language="text")

if __name__ == "__main__":
    main()