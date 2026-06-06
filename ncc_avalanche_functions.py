import os
import sys
import warnings
import numpy as np
import common_functions as cf
import neuron_data_functions as ndf


def _get_ncc_toolbox():
    """Load ncc_toolbox and point failed imports to ncc_toolbox_path.txt."""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    path_file = os.path.join(module_dir, "ncc_toolbox_path.txt")
    if os.path.exists(path_file):
        with open(path_file, "r") as f:
            ncc_path = f.read().strip()
        if ncc_path and ncc_path not in sys.path:
            sys.path.append(ncc_path)
    else:
        warnings.warn("ncc_toolbox_path.txt is missing; please check it if ncc_toolbox cannot be imported.", RuntimeWarning)
    try:
        import ncc_toolbox as ncc
    except ModuleNotFoundError as exc:
        if exc.name != "ncc_toolbox":
            raise
        raise ModuleNotFoundError("Cannot import ncc_toolbox; please check ncc_toolbox_path.txt in common_func.") from exc
    return ncc


def _get_matlab_args(kwargs):
    """Convert keyword options to MATLAB-style name/value arguments."""
    if kwargs is None:
        return []
    args = []
    for key, value in kwargs.items():
        if value is not None:
            args.extend([key, value])
    return args


def _get_selected_spike_times(spike_times, neuron_idx=None):
    """Normalize and validate spike times before selecting neurons."""
    if isinstance(spike_times, np.ndarray) and spike_times.dtype == object:
        spike_times = spike_times.tolist()
    if not isinstance(spike_times, (list, tuple)):
        raise ValueError("spike_times must be a list of spike-time arrays/lists.")
    if neuron_idx is None:
        selected_idx = np.arange(len(spike_times), dtype=int)
    else:
        selected_idx = np.asarray(neuron_idx, dtype=int).reshape(-1)
    selected = []
    for idx in selected_idx:
        if idx < 0 or idx >= len(spike_times):
            raise ValueError("neuron_idx contains an index outside spike_times.")
        arr = np.asarray(spike_times[int(idx)], dtype=float).reshape(-1)
        if np.any(~np.isfinite(arr)):
            raise ValueError("spike_times contains non-finite values.")
        selected.append(arr)
    return selected, selected_idx


def _get_flat_spike_times(selected_spike_times):
    """Merge selected spike arrays for population-level timing estimates."""
    arrays = [arr for arr in selected_spike_times if arr.size > 0]
    if len(arrays) == 0:
        raise ValueError("spike_times contains no spikes in selected neurons.")
    return np.concatenate(arrays).astype(float)


def get_time_bin_duration_from_spike_times(spike_times, start_time=None, end_time=None, neuron_idx=None, method="mean_population_isi"):
    """Choose an avalanche bin width from selected-neuron ISI statistics."""
    selected, _ = _get_selected_spike_times(spike_times, neuron_idx=neuron_idx)
    if method in ["mean_population_isi", "median_population_isi"]:
        flat_times = np.sort(_get_flat_spike_times(selected))
        if start_time is not None:
            flat_times = flat_times[flat_times >= float(start_time)]
        if end_time is not None:
            flat_times = flat_times[flat_times < float(end_time)]
        diffs = np.diff(flat_times)
        diffs = diffs[diffs > 0]
    elif method in ["mean_neuron_isi", "median_neuron_isi"]:
        diff_list = []
        for arr in selected:
            local = np.sort(arr)
            if start_time is not None:
                local = local[local >= float(start_time)]
            if end_time is not None:
                local = local[local < float(end_time)]
            local_diff = np.diff(local)
            local_diff = local_diff[local_diff > 0]
            if local_diff.size > 0:
                diff_list.append(local_diff)
        diffs = np.concatenate(diff_list) if len(diff_list) > 0 else np.array([], dtype=float)
    else:
        raise ValueError("Unknown time-bin estimation method.")
    if diffs.size == 0:
        raise ValueError("Cannot estimate time_bin_duration because no positive ISI is available.")
    if method.startswith("mean"):
        time_bin_duration = float(np.mean(diffs))
    else:
        time_bin_duration = float(np.median(diffs))
    if not np.isfinite(time_bin_duration) or time_bin_duration <= 0:
        raise ValueError("Estimated time_bin_duration is not positive.")
    return time_bin_duration


def _call_plparams(values, args, seed=None):
    """Call NCC plparams while making its implicit sampling reproducible."""
    ncc = _get_ncc_toolbox()
    if seed is None:
        return ncc.plparams(values, *args)
    import ncc_toolbox.powerlaw as ncc_powerlaw
    old_get_rng = ncc_powerlaw.get_rng
    rng = np.random.default_rng(seed)

    def get_rng_for_call(rng_arg=None, seed_arg=None):
        """Use the local RNG only when NCC did not receive one explicitly."""
        if rng_arg is not None or seed_arg is not None:
            return old_get_rng(rng_arg, seed_arg)
        return rng

    try:
        ncc_powerlaw.get_rng = get_rng_for_call
        return ncc.plparams(values, *args)
    finally:
        ncc_powerlaw.get_rng = old_get_rng


def _get_plot_data(values, fit, bin_density=50, unique_bins="on"):
    """Ask NCC for binned empirical and fitted curves without drawing."""
    ncc = _get_ncc_toolbox()
    args = ["plot", "off", "binDensity", int(bin_density), "uniqueBins", unique_bins]
    if np.isfinite(fit.get("tau", np.nan)) and np.isfinite(fit.get("xmin", np.nan)) and np.isfinite(fit.get("xmax", np.nan)):
        fit_params = ncc.Struct()
        fit_params.tau = np.asarray([fit["tau"]], dtype=float)
        fit_params.xmin = np.asarray([fit["xmin"]], dtype=float)
        fit_params.xmax = np.asarray([fit["xmax"]], dtype=float)
        fit_params.x2fit = np.asarray([1], dtype=int)
        args.extend(["fitParams", fit_params])
    data = ncc.plplottool(values, *args)
    return {
        "x": [np.asarray(item, dtype=float) for item in data.x],
        "fit": [np.asarray(item, dtype=float) for item in data.fit],
    }


def get_ncc_powerlaw_fit(values, plparams_kwargs=None, bin_density=50, seed=None, unique_bins="on"):
    """Fit a power law and keep status plus plotting data for summaries."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    fit = {
        "tau": np.nan,
        "xmin": np.nan,
        "xmax": np.nan,
        "sigma_tau": np.nan,
        "p": np.nan,
        "p_crit": np.nan,
        "ks": None,
        "plot_data": None,
        "runtime_warnings": [],
        "status": "not_run",
        "error": None,
    }
    if arr.size < 3 or np.unique(arr).size < 3:
        fit["status"] = "insufficient_unique_values"
    else:
        try:
            args = _get_matlab_args(plparams_kwargs)
            with warnings.catch_warnings(record=True) as warning_records:
                warnings.simplefilter("always", RuntimeWarning)
                tau, xmin, xmax, sigma_tau, p, p_crit, ks = _call_plparams(arr, args, seed=seed)
            fit.update({
                "tau": float(tau),
                "xmin": float(xmin),
                "xmax": float(xmax),
                "sigma_tau": float(sigma_tau),
                "p": float(p),
                "p_crit": float(p_crit),
                "ks": ks,
                "runtime_warnings": [str(item.message) for item in warning_records],
                "status": "ok",
            })
        except Exception as exc:
            fit["status"] = "failed"
            fit["error"] = str(exc)
    try:
        fit["plot_data"] = _get_plot_data(arr, fit, bin_density=bin_density, unique_bins=unique_bins)
    except Exception as exc:
        if fit["error"] is None:
            fit["error"] = "plot_data: " + str(exc)
        else:
            fit["error"] = fit["error"] + "; plot_data: " + str(exc)
    return fit


def get_ncc_size_duration_fit(sizes, durations, size_duration_kwargs=None):
    """Estimate the <size>|duration scaling curve with WLS (weighted least squares)."""
    sizes = np.asarray(sizes, dtype=float).reshape(-1)
    durations = np.asarray(durations, dtype=float).reshape(-1)
    keep = np.isfinite(sizes) & np.isfinite(durations) & (sizes > 0) & (durations > 0)
    sizes = sizes[keep]
    durations = durations[keep]
    fit = {
        "gamma": np.nan,
        "sigma_gamma": np.nan,
        "log_coeff": np.nan,
        "coeff": np.nan,
        "duration_values": np.array([], dtype=float),
        "mean_size": np.array([], dtype=float),
        "counts": np.array([], dtype=int),
        "fit_duration": np.array([], dtype=float),
        "fit_mean_size": np.array([], dtype=float),
        "durmin": np.nan,
        "durmax": np.nan,
        "status": "not_run",
        "error": None,
    }
    if sizes.size == 0:
        fit["status"] = "empty"
        return fit
    unique_duration, counts = np.unique(durations, return_counts=True)
    mean_size = np.asarray([np.mean(sizes[durations == duration]) for duration in unique_duration], dtype=float)
    fit["duration_values"] = unique_duration
    fit["mean_size"] = mean_size
    fit["counts"] = counts.astype(int)
    if unique_duration.size < 2:
        fit["status"] = "insufficient_unique_values"
        return fit
    try:
        args = _get_matlab_args(size_duration_kwargs)
        ncc = _get_ncc_toolbox()
        gamma, sigma_gamma, log_coeff = ncc.sizegivdurwls(sizes, durations, *args)
        coeff = 10 ** float(log_coeff)
        durmin = unique_duration[0]
        durmax = unique_duration[-1]
        if size_duration_kwargs is not None and "durmin" in size_duration_kwargs and size_duration_kwargs["durmin"] is not None:
            durmin = float(size_duration_kwargs["durmin"])
        if size_duration_kwargs is not None and "durmax" in size_duration_kwargs and size_duration_kwargs["durmax"] is not None:
            durmax = float(size_duration_kwargs["durmax"])
        fit_keep = (unique_duration >= durmin) & (unique_duration <= durmax)
        fit_duration = unique_duration[fit_keep]
        fit.update({
            "gamma": float(gamma),
            "sigma_gamma": float(sigma_gamma),
            "log_coeff": float(log_coeff),
            "coeff": coeff,
            "fit_duration": fit_duration,
            "fit_mean_size": coeff * (fit_duration ** float(gamma)),
            "durmin": float(durmin),
            "durmax": float(durmax),
            "status": "ok",
        })
    except Exception as exc:
        fit["status"] = "failed"
        fit["error"] = str(exc)
    return fit


def get_ncc_criticality(tau, alpha, gamma):
    """Compare fitted exponents against the avalanche scaling relation."""
    if tau <= 1:
        predicted_gamma = np.nan
    else:
        predicted_gamma = (alpha - 1) / (tau - 1)
    difference = abs(predicted_gamma - gamma) if np.isfinite(predicted_gamma) and np.isfinite(gamma) else np.nan
    ratio = gamma / predicted_gamma if np.isfinite(predicted_gamma) and predicted_gamma != 0 else np.nan
    return predicted_gamma, difference, ratio


def _get_duration_xlabel(results):
    """Return the duration label matching the fitted duration unit."""
    return "duration (bins)"


def get_ncc_avalanche_results(spike_times, start_time=None, end_time=None, time_bin_duration=None, neuron_idx=None, time_bin_estimation="mean_population_isi", plparams_kwargs=None, size_duration_kwargs=None, bin_density=50, unique_bins="on", seed=None):
    """Bin spikes, fit NCC avalanche exponents, and package diagnostics.

    NCC duration fits use integer duration bin counts, not duration in time units.
    """
    if time_bin_duration is None:
        time_bin_duration = get_time_bin_duration_from_spike_times(
            spike_times,
            start_time=start_time,
            end_time=end_time,
            neuron_idx=neuron_idx,
            method=time_bin_estimation,
        )
        time_bin_source = time_bin_estimation
    else:
        time_bin_source = "input"
    avalanche_results = ndf.get_avalanche_from_spike_times(
        spike_times,
        start_time=start_time,
        end_time=end_time,
        time_bin_duration=time_bin_duration,
        neuron_idx=neuron_idx,
    )
    duration_fit_values = np.asarray(avalanche_results["avalanche_duration_bin"], dtype=int)
    selected_spike_times, selected_idx = _get_selected_spike_times(spike_times, neuron_idx=neuron_idx)
    flat_spike_times = _get_flat_spike_times(selected_spike_times)
    local_start_time = float(np.min(flat_spike_times)) if start_time is None else float(start_time)
    local_end_time = float(np.max(flat_spike_times[flat_spike_times >= local_start_time]) + time_bin_duration) if end_time is None else float(end_time)
    n_spikes = int(np.sum((flat_spike_times >= local_start_time) & (flat_spike_times < local_end_time)))
    size_seed = seed
    duration_seed = None if seed is None else int(seed) + 1
    size_fit = get_ncc_powerlaw_fit(
        avalanche_results["avalanche_size"],
        plparams_kwargs=plparams_kwargs,
        bin_density=bin_density,
        seed=size_seed,
        unique_bins=unique_bins,
    )
    duration_fit = get_ncc_powerlaw_fit(
        duration_fit_values,
        plparams_kwargs=plparams_kwargs,
        bin_density=bin_density,
        seed=duration_seed,
        unique_bins=unique_bins,
    )
    size_duration_fit = get_ncc_size_duration_fit(
        avalanche_results["avalanche_size"],
        duration_fit_values,
        size_duration_kwargs=size_duration_kwargs,
    )
    tau = size_fit["tau"]
    alpha = duration_fit["tau"]
    gamma = size_duration_fit["gamma"]
    predicted_gamma, difference, ratio = get_ncc_criticality(tau, alpha, gamma)
    warnings = []
    for label, fit in [("size_fit", size_fit), ("duration_fit", duration_fit), ("size_duration_fit", size_duration_fit)]:
        if fit["status"] != "ok":
            warnings.append(label + ": " + fit["status"] + ("" if fit.get("error") is None else " (" + fit["error"] + ")"))

    results = {}
    results.update(avalanche_results)
    results.update({
        "n_spikes": n_spikes,
        "n_neurons": int(len(selected_idx)),
        "neuron_idx": selected_idx,
        "start_time": local_start_time,
        "end_time": local_end_time,
        "time_bin_duration": time_bin_duration,
        "time_bin_source": time_bin_source,
        "duration_bin_count": avalanche_results["avalanche_duration_bin"],
        "duration_fit_values": duration_fit_values,
        "duration_fit_unit": "bin",
        "size_fit": size_fit,
        "duration_fit": duration_fit,
        "size_duration_fit": size_duration_fit,
        "plot_data": {
            "size": size_fit["plot_data"],
            "duration": duration_fit["plot_data"],
            "size_duration": size_duration_fit,
        },
        "size_tau": tau,
        "duration_alpha": alpha,
        "gamma": gamma,
        "predicted_gamma": predicted_gamma,
        "difference": difference,
        "ratio": ratio,
        "scaling": {
            "predicted_gamma": predicted_gamma,
            "difference": difference,
            "ratio": ratio,
            "gamma": gamma,
        },
        "warnings": warnings,
    })
    return results


def _get_first_plot_array(plot_data, key):
    """Extract one NCC plplottool curve, or None when it is unusable."""
    if plot_data is None:
        return None
    values = plot_data.get(key)
    if values is None or len(values) == 0 or values[0] is None:
        return None
    arr = np.asarray(values[0], dtype=float)
    if arr.ndim != 2 or arr.shape[0] != 2 or arr.shape[1] == 0:
        return None
    return arr


def _plot_distribution_fallback(ax, values, color, label):
    """Plot the empirical PMF when NCC plot data is missing or invalid."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values) & (values > 0)]
    unique_values, counts = np.unique(values, return_counts=True)
    if unique_values.size == 0:
        return None
    probability = counts.astype(float) / np.sum(counts)
    return cf.plt_scatter(ax, unique_values, probability, color=color, label=label)


def _add_fit_text(ax, lines, text_x=0.04, text_y=0.04, ha="left", va="bottom", fontsize=None):
    """Add compact fit diagnostics while ignoring absent text lines."""
    lines = [line for line in lines if line is not None]
    if len(lines) == 0:
        return None
    return cf.add_text(
        ax,
        "\n".join(lines),
        x=text_x,
        y=text_y,
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 2},
    )


def plot_ncc_powerlaw_distribution(ax, results, property_name="size", show_info=True, data_color=cf.BLUE, fit_color=cf.RED, data_label=None, fit_label=None, marker_kwargs=None, line_kwargs=None, set_ax_kwargs=None):
    """Plot one fitted distribution from get_ncc_avalanche_results results."""
    marker_kwargs = cf.update_dict({"s": 18}, marker_kwargs)
    line_kwargs = cf.update_dict({}, line_kwargs)
    set_ax_kwargs = cf.update_dict({}, set_ax_kwargs)
    if property_name == "size":
        values = results["avalanche_size"]
        fit = results["size_fit"]
        title = "Size distribution"
        xlabel = "avalanche size"
        exponent_name = r"$\tau$"
    elif property_name == "duration":
        values = results["duration_fit_values"]
        fit = results["duration_fit"]
        title = "Duration distribution"
        xlabel = _get_duration_xlabel(results)
        exponent_name = r"$\alpha$"
    else:
        raise ValueError("property_name must be size or duration.")

    plot_data = fit.get("plot_data")
    data_arr = _get_first_plot_array(plot_data, "x")
    fit_arr = _get_first_plot_array(plot_data, "fit")
    if data_arr is None:
        _plot_distribution_fallback(ax, values, data_color, data_label)
    else:
        cf.plt_scatter(ax, data_arr[0], data_arr[1], color=data_color, label=data_label, **marker_kwargs)
    if fit_arr is not None and fit_arr.shape[1] > 0:
        cf.plt_line(ax, fit_arr[0], fit_arr[1], color=fit_color, label=fit_label, **line_kwargs)
    if show_info:
        lines = [
            exponent_name + " = " + cf.round_float(float(fit.get("tau", np.nan)), digits=3, format_type="general"),
            r"$x_{\min}$ = " + cf.round_float(float(fit.get("xmin", np.nan)), digits=3, format_type="general") + r", $x_{\max}$ = " + cf.round_float(float(fit.get("xmax", np.nan)), digits=3, format_type="general"),
            r"$p$ = " + cf.round_float(float(fit.get("p", np.nan)), digits=3, format_type="general"),
        ]
        _add_fit_text(ax, lines, text_x=0.04, text_y=0.04, ha="left", va="bottom")
    local_set_ax_kwargs = {
        "xlabel": xlabel,
        "ylabel": "probability",
        "title": title,
        "xlog": True,
        "ylog": True,
        "legend": True,
        "legend_loc": "upper right",
    }
    local_set_ax_kwargs = cf.update_dict(local_set_ax_kwargs, set_ax_kwargs)
    cf.set_ax(ax, **local_set_ax_kwargs)
    return ax


def plot_ncc_size_duration_relation(ax, results, show_info=True, data_color=cf.BLUE, fit_color=cf.RED, data_label=None, fit_label=None, marker_kwargs=None, line_kwargs=None, set_ax_kwargs=None):
    """Plot the fitted size-duration scaling from avalanche results."""
    fit = results["size_duration_fit"]
    marker_kwargs = cf.update_dict({}, marker_kwargs)
    line_kwargs = cf.update_dict({}, line_kwargs)
    set_ax_kwargs = cf.update_dict({}, set_ax_kwargs)
    durations = np.asarray(fit["duration_values"], dtype=float)
    mean_size = np.asarray(fit["mean_size"], dtype=float)
    counts = np.asarray(fit["counts"], dtype=float)
    if durations.size > 0:
        if "s" not in marker_kwargs:
            marker_kwargs["s"] = 20 + 60 * np.sqrt(counts / np.max(counts))
        cf.plt_scatter(ax, durations, mean_size, color=data_color, label=data_label, **marker_kwargs)
    fit_duration = np.asarray(fit["fit_duration"], dtype=float)
    fit_mean_size = np.asarray(fit["fit_mean_size"], dtype=float)
    if fit_duration.size > 0:
        cf.plt_line(ax, fit_duration, fit_mean_size, color=fit_color, label=fit_label, **line_kwargs)
    if show_info:
        lines = [
            r"$\gamma$ = " + cf.round_float(float(fit.get("gamma", np.nan)), digits=3, format_type="general"),
            r"$\frac{\alpha - 1}{\tau - 1}$ = " + cf.round_float(float(results.get("predicted_gamma", np.nan)), digits=3, format_type="general"),
        ]
        _add_fit_text(ax, lines, text_x=0.96, text_y=0.04, ha="right", va="bottom")
    local_set_ax_kwargs = {
        "xlabel": _get_duration_xlabel(results),
        "ylabel": "mean size",
        "title": "Size-duration relation",
        "xlog": True,
        "ylog": True,
        "legend": True,
        "legend_loc": "upper left",
    }
    local_set_ax_kwargs = cf.update_dict(local_set_ax_kwargs, set_ax_kwargs)
    cf.set_ax(ax, **local_set_ax_kwargs)
    return ax


def plot_ncc_avalanche_summary(results, fig=None, axes=None, show_info=True, powerlaw_kwargs=None, size_duration_kwargs=None):
    """Plot the three standard NCC avalanche summary panels."""
    if fig is None or axes is None:
        fig, axes = cf.get_fig_ax(nrows=1, ncols=3, ax_width=4.2, ax_height=3.0)
    axes_flat = np.asarray(axes).reshape(-1)
    if axes_flat.size < 3:
        raise ValueError("axes must contain at least 3 axes.")
    powerlaw_kwargs = cf.update_dict({}, powerlaw_kwargs)
    size_duration_kwargs = cf.update_dict({}, size_duration_kwargs)
    plot_ncc_powerlaw_distribution(axes_flat[0], results, property_name="size", show_info=show_info, **powerlaw_kwargs)
    plot_ncc_powerlaw_distribution(axes_flat[1], results, property_name="duration", show_info=show_info, **powerlaw_kwargs)
    plot_ncc_size_duration_relation(axes_flat[2], results, show_info=show_info, **size_duration_kwargs)
    return fig, axes


def save_ncc_avalanche_summary(results, output, save_fig_kwargs=None, summary_kwargs=None):
    """Save the standard summary figure for avalanche results."""
    save_fig_kwargs = cf.update_dict({"bbox_inches": "tight"}, save_fig_kwargs)
    summary_kwargs = cf.update_dict({}, summary_kwargs)
    fig, axes = plot_ncc_avalanche_summary(results, **summary_kwargs)
    cf.save_fig(fig, output, **save_fig_kwargs)
    return fig, axes
