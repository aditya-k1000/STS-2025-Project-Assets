from astropy.timeseries import LombScargle
from ChandraPy import Utilities as utils
from ChandraPy import Download as d
from ChandraPy import Lightcurves as lc
from ciao_contrib.runtool import dmcopy, dmkeypar, dmlist
import numpy as np
import os
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d
import shutil
import time

def generate_simulation_data(binsize, gti_list):
    """Function to generate time-series data simulating an eclipsing X-ray Binary system.

    Args:
        binsize (int): Size of bins to use for binning the data (s).
        gti_list (list): List of tuples of GTI start and end times.

    Returns:
        tuple: A tuple containing the orbital period used to generate simulation, event list, binned times array, count rates array, and count rate errors array.
    """

    # Defining parameters
    rng = np.random.default_rng(1)
    mean_rate = np.random.uniform(0.8, 1.4) 
    eclipse_depth = np.random.uniform(0.6, 0.99)
    k_noise = 0.001
    eclipse_duration = np.random.randint(2000, 6000)
    P_sim = np.random.uniform(max(12000, 3 * eclipse_duration), 80000)

    if gti_list is not None and len(gti_list) > 0:
        t_blocks = [np.arange(tstart, tstop, binsize) for tstart, tstop in gti_list]
        binned_times = np.concatenate(t_blocks)
        binned_times.sort()
    else:
        binned_times = np.arange(0, P_sim * 5, binsize)

    # Creating flat-line light curve for steady source
    phase = binned_times % P_sim
    baseline = mean_rate * (1 + k_noise * rng.normal(0, 1, size = binned_times.size))
    flatline_count_rates = baseline.copy()

    # Creating eclipsing XRB model
    ingress_egress_frac = rng.uniform(0.15, 0.3)
    slope_width = ingress_egress_frac * eclipse_duration
    full_eclipse_width = eclipse_duration - 2 * slope_width
    center = 0.5 * P_sim
    start_ingress = center - 0.5 * eclipse_duration
    phase_rel = (phase - start_ingress) % P_sim
    attenuation = np.ones_like(binned_times)

    ingress_mask = (phase_rel >= 0) & (phase_rel < slope_width)
    attenuation[ingress_mask] = 1 - 0.5 * eclipse_depth * (1 - np.cos(np.pi * phase_rel[ingress_mask] / slope_width))

    full_mask = (phase_rel >= slope_width) & (phase_rel < slope_width + full_eclipse_width)
    attenuation[full_mask] = 1 - eclipse_depth

    egress_mask = (phase_rel >= slope_width + full_eclipse_width) & (phase_rel < eclipse_duration)
    x = phase_rel[egress_mask] - (slope_width + full_eclipse_width)
    attenuation[egress_mask] = 1 - 0.5 * eclipse_depth * (1 + np.cos(np.pi * x / slope_width))

    # Creating binned data by superposing eclipsing XRB model on flat-line light curve to simulate an eclipsing XRB
    flatline_count_rates *= attenuation

    counts = rng.poisson(flatline_count_rates * binsize)
    count_rates = counts / binsize
    count_rate_errors = np.sqrt(counts) / binsize
    count_rate_errors = np.maximum(count_rate_errors, 0.01 * np.mean(count_rates))
    if np.any(count_rate_errors <= 0):
        positive_median = np.median(count_rate_errors[count_rate_errors > 0]) if np.any(count_rate_errors > 0) else 1.0 / binsize
        count_rate_errors[count_rate_errors <= 0] = positive_median * 0.1

    # Creating event list
    photon_times = [ti + rng.uniform(0, binsize, size = ci) for ti, ci in zip(binned_times, counts) if ci > 0]
    event_list = np.concatenate(photon_times) if photon_times else np.array([], dtype = float)
    event_list.sort()

    return P_sim, event_list, binned_times, count_rates, count_rate_errors

def run_lombscargle_periodogram(binned_times, 
           count_rates,
           count_rate_errors, 
           min_period = 10000, 
           max_period = 100000, 
           num_periods = 20000, 
           sg_window_frac = 0.01, 
           sg_polyorder = 4, 
           harm_threshold = 0.85):
    """Function to run the binned data through a Lomb-Scargle Periodogram.

    Args:
        binned_times (list): Binned times array.
        count_rates (list): Count rates array.
        count_rate_errors (list): Count rate errors array.
        min_period (int, optional): Minimum period used for Lomb-Scargle search. Defaults to 10000.
        max_period (int, optional): Maximum period used for Lomb-Scargle search. Defaults to 100000.
        num_periods (int, optional): Number of periods searched. Defaults to 20000.
        sg_window_frac (float, optional): Window fraction used for Savitzky-Golay smoothing. Defaults to 0.01.
        sg_polyorder (int, optional): Order of polynomial used for Savitzky-Golay smoothing. Defaults to 4.
        harm_threshold (float, optional): Power threshold for harmonics. Defaults to 0.85.

    Returns:
        tuple: A tuple containing P_LS, P_min, and P_max.
    """
    
    # Ensuring proper datatypes and creating period grid
    count_rates = np.asarray(count_rates, dtype = float)
    count_rate_errors = np.asarray(count_rate_errors, dtype = float)
    min_freq, max_freq = 1 / max_period, 1 / min_period
    frequency = np.linspace(min_freq, max_freq, num_periods)
    period = 1 / frequency

    # Running raw Lomb-Scargle Periodogram
    ls = LombScargle(binned_times, count_rates, count_rate_errors, nterms = 2)
    power = ls.power(frequency)

    # Function to smooth periodogram
    def smooth_periodogram(power, frac = 0.01, polyorder = 3):
        n = len(power)
        win = int(frac * n)

        if win <= polyorder:
            win = polyorder + 1
        if win % 2 == 0: 
            win += 1
        if win > n:       
            win = n - 1 if n % 2 == 0 else n
            if win <= polyorder:
                win = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2

        return savgol_filter(power, window_length = win, polyorder = polyorder)

    power_smooth = smooth_periodogram(power, frac = sg_window_frac, polyorder = sg_polyorder)

    # Finding peak in smoothed periodogram
    base_idx_smooth = np.argmax(power_smooth)
    base_period_smooth = period[base_idx_smooth]
    base_power_smooth = power_smooth[base_idx_smooth]

    # Checking for harmonics
    harmonics = []
    for mult in range(2, 10):
        target_period = base_period_smooth * mult
        mask = (period > target_period * 0.95) & (period < target_period * 1.05)
        if np.any(mask):
            idx_local = np.argmax(power_smooth[mask])
            idx = np.where(mask)[0][idx_local]
            p = power_smooth[idx]
            harmonics.append((mult, idx, p))

    harmonics = [(m, i, p) for m, i, p in harmonics if p >= harm_threshold * base_power_smooth]
    best_idx = base_idx_smooth
    P_LS = base_period_smooth
    n_mask = int(0.05 * len(period))

    if harmonics:
        harmonics.sort(key = lambda x: x[2], reverse = True)
        for m, i, p in harmonics:
            rel_power = p / base_power_smooth
            if rel_power >= harm_threshold and (n_mask <= i <= len(period) - n_mask):
                if m < 6:
                    best_idx = i
                    P_LS = period[i]
                    break

    # Calculating period range for Step 2 of period recovery algorithm
    pmax = power_smooth[best_idx]
    threshold = pmax / np.sqrt(2)

    left_candidates = np.where(power_smooth[:best_idx + 1] <= threshold)[0]
    period_left = period[left_candidates[-1]] if len(left_candidates) > 0 else period[0]

    right_candidates = np.where(power_smooth[best_idx:] <= threshold)[0]
    period_right = period[right_candidates[0] + best_idx] if len(right_candidates) > 0 else period[-1]

    P_min, P_max = sorted([period_left, period_right])

    if P_max - P_min < 100:  
        P_min, P_max = P_LS - 300, P_LS + 300

    return P_LS, P_min, P_max

def novel_chi2_periodogram(event_list, 
                           min_period, 
                           max_period, 
                           period_step, 
                           num_bins,
                           smoothing = True,
                           sg_window_frac = 0.1, 
                           sg_polyorder = 3,
                           centrality_power = 5.0, 
                           peak_prominence_frac = 0.01,
                           plateau_grad_thresh = 0.02, 
                           harmonic_factor = 0.6,
                           edge_trim_frac = 0.05, 
                           gaussian_smooth_sigma_frac = 0.01, 
                           ratio_threshold = 0.03):
    """Function to run event list through the novel chi-squared calculation and periodogram search.

    Args:
        event_list (list): Array of photon arrival times.
        min_period (int): Minimum period used for Lomb-Scargle search.
        max_period (int): Maximum period used for Lomb-Scargle search.
        period_step (float): Period step value used to generate period range to search.
        num_bins (int, optional): Number of bins used for phase folding.
        smoothing (bool, optional): Option to use Smoothing or not to determing P_best. Defaults to True.
        sg_window_frac (float, optional): Window fraction used for Savitzky-Golay smoothing. Defaults to 0.1.
        sg_polyorder (int, optional): Order of polynomial used for Savitzky-Golay smoothing. Defaults to 3.
        centrality_power (float, optional): Power used for centrality calculation. Defaults to 5.0.
        peak_prominence_frac (float, optional): Prominence fraction used for peak detection. Defaults to 0.01.
        plateau_grad_thresh (float, optional): Threshold for a region to be determined to be a plateau. Defaults to 0.02.
        harmonic_factor (float, optional): Factor for harmonics, multiplied each time harmonic detected. Defaults to 0.6.
        edge_trim_frac (float, optional): Percentage of periods to excise from each side of period range. Defaults to 0.05.
        gaussian_smooth_sigma_frac (float, optional): Gaussian smoothing fraction. Defaults to 0.01.
        ratio_threshold (float, optional): Ratio threshold for harmonic testing. Defaults to 0.03.

    Returns:
        float: Value for P_best, the time period best fitting the data.
    """

    eps = 1e-12
    periods = np.arange(min_period, max_period, period_step)
    n_periods = len(periods)

    # Creating raw χ² Periodogram
    chi2 = np.empty(n_periods)
    for i, p in enumerate(periods):
        phases = (event_list % p) / p
        counts, _ = np.histogram(phases, bins = num_bins, range = (0.0, 1.0))
        expected = np.mean(counts)
        sigma = 1 + np.sqrt(0.75 + counts)
        chi2[i] = np.sum((counts - expected) ** 2 / (sigma) ** 2)

    if smoothing:
        # Define range of valid periods
        n_trim = max(1, int(edge_trim_frac * n_periods))
        valid_slice = slice(n_trim, n_periods - n_trim)
        periods_valid = periods[valid_slice]
        chi2_valid = chi2[valid_slice]

        # Applying Gaussian and S-G filter to smooth χ² Periodogram
        sigma = max(1.0, gaussian_smooth_sigma_frac * n_periods)
        chi2_gauss = gaussian_filter1d(chi2_valid, sigma = sigma, mode = "reflect")

        n = len(chi2_gauss)
        if n <= sg_polyorder + 2:
            chi2_smooth = chi2_gauss.copy()
            P_best = periods_valid[np.argmax(chi2_smooth)]
        else:
            win = int(n * sg_window_frac)
            win = max(win, sg_polyorder + 2)    
            if win % 2 == 0:
                win += 1                         
            if win > n:
                win = n - 1 if (n - 1) % 2 == 1 else n - 2
            win = max(win, sg_polyorder + 2)     

            chi2_smooth = savgol_filter(chi2_gauss, window_length = win, polyorder = sg_polyorder)
            chi2_smooth_full = np.full_like(chi2, np.nan)
            chi2_smooth_full[valid_slice] = chi2_smooth

            # Using a prominence detection method to determine the peaks in the smoothed periodogram
            chi2_range = np.nanmax(chi2_smooth_full) - np.nanmin(chi2_smooth_full)
            min_prominence = max(eps, chi2_range * peak_prominence_frac)
            peaks, props = find_peaks(chi2_smooth, prominence = min_prominence)

            chi2_range = np.nanmax(chi2_smooth) - np.nanmin(chi2_smooth)
            min_prominence = max(eps, chi2_range * peak_prominence_frac * 0.5)  
            min_height = np.nanmin(chi2_smooth) + 0.05 * chi2_range            

            peaks, props = find_peaks(chi2_smooth,
                                      prominence = min_prominence,
                                      height = min_height,
                                      distance = max(1, int(0.01 * len(chi2_smooth))))

            if len(peaks) == 0:
                peak_idx_local = int(np.argmax(chi2_smooth))
                peaks = np.array([peak_idx_local])
                props = {"prominences": np.array([chi2_smooth[peak_idx_local] - np.min(chi2_smooth)])}

            # Scoring peaks
            center_idx_local = (len(periods_valid) - 1) / 2.0
            prominences = props.get("prominences", np.ones_like(peaks))
            scores = []

            for j, pidx in enumerate(peaks):
                # Defining centrality
                c = (1.0 - abs(pidx - center_idx_local) / center_idx_local) ** centrality_power

                # Defining sharpness
                left = max(0, pidx - 1)
                right = min(len(chi2_smooth) - 1, pidx + 1)
                slope_change = abs(chi2_smooth[right] - 2 * chi2_smooth[pidx] + chi2_smooth[left]) + eps
                s = (prominences[j] ** 2) / (slope_change + eps)

                # Defining harmonic penalty
                h = 1.0
                for other_pidx in peaks:
                    if other_pidx == pidx:
                        continue
                    ratio = periods_valid[pidx] / periods_valid[other_pidx]
                    k = np.round(ratio)
                    if k >= 1 and abs(ratio - k) / k < ratio_threshold:
                        h *= harmonic_factor

                # Calculating scores
                score = (s ** 1.5) * (c ** 3.0) * h * (chi2_smooth[pidx] ** 0.2)
                scores.append(score)

            best_local_idx = peaks[int(np.argmax(scores))]
            P_score = periods_valid[best_local_idx]

            # Inflection point analysis
            grad = np.gradient(chi2_smooth)
            abs_grad_peak = abs(grad[best_local_idx])
            if abs_grad_peak < plateau_grad_thresh:
                curv = np.gradient(grad)
                inflections = np.where(np.sign(curv[:-1]) != np.sign(curv[1:]))[0]
                if inflections.size:
                    diffs = np.abs(inflections - best_local_idx)
                    inf_idx = inflections[np.argmin(diffs)]
                    if inf_idx < best_local_idx:
                        best_local_idx = inf_idx
                        P_score = periods_valid[best_local_idx]

            # Parabolic refinement
            global_idx = best_local_idx + n_trim
            if 1 <= global_idx < (len(periods) - 1):
                x0, x1, x2 = periods[global_idx - 1:global_idx + 2]
                y0, y1, y2 = chi2[global_idx - 1:global_idx + 2]
                denom = (y0 - 2 * y1 + y2)
                if abs(denom) > eps:
                    dp = 0.5 * (y0 - y2) / denom * period_step
                    refined = x1 + dp
                    if x0 <= refined <= x2:
                        P_best = refined
                    else:
                        P_best = x1
                else:
                    P_best = x1
            else:
                P_best = P_score

    else:
        chi2_smooth = chi2.copy()
        P_best = periods[np.argmax(chi2_smooth)]

    return P_best

def extract_47_tucanae_gtis(data_dir):
    """Function that extracts GTIs for all Chandra observations of 47 Tucanae.

    Args:
        data_dir (str): Absolute path to directory where 47 Tucanae Chandra data can be downloaded and accessed.

    Returns:
        list: List of tuples of GTI start and end times.
    """

    utils.retrieve_obs_ids(data_dir, "NGC104")
    df = pd.read_csv(os.path.join(data_dir, "NGC104.csv"), dtype = str)

    for obs_id in df["Observation ID"]:
        d.download_and_reprocess_obsid(data_dir, obs_id)

    gti_intervals = []

    for obs_id in os.listdir(data_dir):
        if not obs_id.endswith("csv"):
            event_file = os.path.join(data_dir, obs_id, f"{obs_id}_evt2.fits")
            if os.path.exists(event_file):
                tstart = float(dmkeypar(infile = event_file, keyword = "TSTART", echo = True))
                tstop = float(dmkeypar(infile = event_file, keyword = "TSTOP", echo = True))
                if tstart is not None and tstop is not None:
                    gti_intervals.append((tstart, tstop))

    return gti_intervals

def period_recovery_algorithm(event_list, binned_times, count_rates, count_rate_errors, start_period, end_period, num_periods, num_bins):
    """Function running the Period Recovery Algorithm.

    Args:
        event_list (list): Array of photon arrival times.
        binned_times (list): Binned times array.
        count_rates (list): Count rates array.
        count_rate_errors (list): Count rate errors array.
        start_period (int): Period to start Step 1 at.
        end_period (int): Period to end Step 1 at.
        num_periods (int): Number of periods to test using Step 1.
        num_bins (int): Number of bins used for phase folding in Step 2.

    Returns:
        float: Value for P_final, the time period best fitting the data.
    """

    # Step 1
    P_LS, P_min, P_max = run_lombscargle_periodogram(binned_times, count_rates, count_rate_errors, start_period, end_period, num_periods)
    
    # Step 2 Iteraton 1
    P_1 = novel_chi2_periodogram(event_list, P_min, P_max, 5, num_bins, smoothing = True)

    # Step 2 Iteration 2
    P_final = novel_chi2_periodogram(event_list, P_1 - 50, P_1 + 50, 0.5, num_bins, smoothing = False)

    return P_final

def test_simulations(data_dir, num_simulations):
    """Function to test Period Recovery Algorithm on simulations.

    Args:
        data_dir (str): Absolute path to directory where 47 Tucanae Chandra data can be downloaded and accessed.
        num_simulations (int): The number of simulations to generate.

    Returns:
        tuple: A tuple containing the accuracies and computation time per simulation for all 3 methods.
    """

    gtis_for_simulations = extract_47_tucanae_gtis(data_dir)

    accuracies = np.array([])
    accuracies_ls = np.array([])
    accuracies_chi2 = np.array([])

    times_running = np.array([])
    times_running_ls = np.array([])
    times_running_chi2 = np.array([])

    for i in range(num_simulations):
        # Generating simulation
        P_sim, event_list, binned_times, count_rates, count_rate_errors = generate_simulation_data(500, gtis_for_simulations)

        # Testing Period Recovery Algorithm on simulation
        start_time = time.time()
        P_final = period_recovery_algorithm(event_list, binned_times, count_rates, count_rate_errors, 10000, 100000, 20000, 100)
        end_time = time.time()

        accuracy = 100 - ((np.abs(P_final - P_sim) / P_sim) * 100)
        time_elapsed = end_time - start_time

        accuracies = np.append(accuracies, accuracy)
        times_running = np.append(times_running, time_elapsed)

        # Also testing raw L-S Periodogram and raw χ² Periodogram on first 100 simulations
        if i < 100:
            start_time_ls = time.time()
            P_LS, _, _ = run_lombscargle_periodogram(binned_times, count_rates, count_rate_errors, 10000, 100000, 20000)
            end_time_ls = time.time()

            accuracy_ls = 100 - ((np.abs(P_LS - P_sim) / P_sim) * 100)
            time_elapsed_ls = end_time_ls - start_time_ls

            accuracies_ls = np.append(accuracies_ls, accuracy_ls)
            times_running_ls = np.append(times_running_ls, time_elapsed_ls)


            start_time_chi2 = time.time()
            P_chi2 = novel_chi2_periodogram(event_list, 10000, 100000, 1, 100, smoothing = True)
            end_time_chi2 = time.time()

            accuracy_chi2 = 100 - ((np.abs(P_chi2 - P_sim) / P_sim) * 100)
            time_elapsed_chi2 = end_time_chi2 - start_time_chi2

            accuracies_chi2 = np.append(accuracies_chi2, accuracy_chi2)
            times_running_chi2 = np.append(times_running_chi2, time_elapsed_chi2)

    return accuracies, accuracies_ls, accuracies_chi2, times_running, times_running_ls, times_running_chi2

def test_archival_data(source, sources_dir, data_dir):
    """Function to test Period Recovery Algorithm on real sources.

    Args:
        source (str): J2000 sexagecimal source name.
        sources_dir (str): Absolute path to directory where source data can be saved.
        data_dir (str): Absolute path to directory where Chandra data can be downloaded and accessed.

    Returns:
        tuple: A tuple containing the value of P_final and the computation time.
    """

    # Extracting archival Chandra time-series data for the source
    not_processed = 0
    to_remove = []
    event_list = []
    binned_times = []
    count_rates = []
    count_rate_errors = []

    utils.retrieve_obs_ids(data_dir, source)
    df = pd.read_csv(os.path.join(data_dir, f"{source}.csv"), dtype = str)
    for obs_id in df["Observation ID"]:
        obs_dir = os.path.join(sources_dir, obs_id)
        obs_data_dir = os.path.join(data_dir, obs_id)
        os.makedirs(obs_dir, exist_ok = True)

        if not os.path.exists(os.path.join(obs_data_dir, f"{obs_id}_evt2.fits")):
            d.download_and_reprocess_obsid(obs_data_dir, obs_id)

        try:
            utils.save_source_region(obs_dir, obs_data_dir, source)
            region_file = os.path.join(obs_dir, f"{source}_{obs_id}.reg")
            event_file = os.path.join(obs_data_dir, f"{obs_id}_evt2.fits")
            processed = lc.lightcurve_generation(obs_dir, obs_data_dir, source, 200, 5, np.log(1e-4))
            if not processed:
                not_processed += 1
                to_remove.append(obs_dir)
            else:
                instrument = utils.instrument_checker(os.path.join(obs_data_dir, f"{obs_id}_evt2.fits"))
                outfile = os.path.join(obs_dir, f"{source}_{obs_id}.fits")
                if instrument == "ACIS":
                    dmcopy(infile = f"{event_file}[sky=region({region_file})][energy=200:8000]", outfile = outfile, clobber = "yes")
                    dmlist(infile = f"{outfile}[cols time]", outfile = f"{outfile}.txt", opt = "data,raw")
                else:
                    dmcopy(infile = f"{event_file}[sky=region({region_file})][samp=10:300]", outfile = outfile, clobber = "yes")
                    dmlist(infile = f"{outfile}[cols time]", outfile = f"{outfile}.txt", opt = "data,raw")

                os.remove(outfile)
                binned_data_df = pd.read_csv(os.path.join(obs_dir, f"{source}_{obs_id}.csv"))
                events_df = pd.read_csv(f"{outfile}.txt")
                event_list.extend(list(events_df["#  time"].astype(float)))
                binned_times.extend(list(binned_data_df["Time"].astype(float)))
                count_rates.extend(list(binned_data_df["Broadband Count Rate"].astype(float)))
                count_rate_errors.extend(list(binned_data_df["Count Rate Error"].astype(float)))

        except Exception:
            not_processed += 1
            processed = False
            to_remove.append(obs_dir)

    for obs_dir in to_remove:
        shutil.rmtree(obs_dir, ignore_errors = True)

    # Testing Period Recovery Algorithm on archival Chandra time-series data
    start_time = time.time()
    P_final = period_recovery_algorithm(event_list, binned_times, count_rates, count_rate_errors, 10000, 100000, 100000, 200)
    end_time = time.time()

    time_elapsed = end_time - start_time

    return P_final, time_elapsed