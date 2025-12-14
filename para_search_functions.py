import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import os


class EarlyStop(Exception):
    pass


def find_input_for_target_with_max_iter(
    forward_fn,
    target_value,
    bounds=None,
    x0=None,
    max_iter=100,
    method="auto",
    weights=None,
    tol_objective=1e-8,
    tol_x=1e-6,
    patience=10,
    verbose=False,
    plot_history=False,
    save_figdir=None,
):
    target_array = np.asarray(target_value)
    weights_array = np.asarray(weights) if weights is not None else None

    history = {
        "x": [],
        "residual_sq": [],
        "objective": [],
    }

    def compute_residual_vector(x):
        output = np.asarray(forward_fn(x))
        residual = output - target_array
        if weights_array is not None:
            residual = residual * weights_array
        return residual

    def objective_scalar(x):
        r = compute_residual_vector(x)
        return np.sum(r ** 2)

    best_objective = np.inf
    stall_counter = 0

    def callback(xk):
        nonlocal best_objective, stall_counter

        r = compute_residual_vector(xk)
        obj = np.sum(r ** 2)

        history["x"].append(np.copy(xk))
        history["residual_sq"].append(r ** 2)
        history["objective"].append(obj)

        if verbose:
            print(
                f"iter={len(history['objective'])} "
                f"objective={obj:.3e} "
                f"x={xk}"
            )

        if obj < best_objective - tol_objective:
            best_objective = obj
            stall_counter = 0
        else:
            stall_counter += 1

        if stall_counter >= patience:
            raise EarlyStop("no improvement")

        if len(history["x"]) > 1:
            dx = np.linalg.norm(history["x"][-1] - history["x"][-2])
            if dx < tol_x:
                raise EarlyStop("x converged")
        
        if plot_history:
            fig, ax = plot_optimization_trajectory(history)
            fig.suptitle("Optimization trajectory")
            plt.tight_layout()
            if save_figdir is not None:
                os.makedirs(save_figdir, exist_ok=True)
                fig.savefig(
                    f"{save_figdir}/optimization_trajectory"
                )
            plt.close(fig)

    if method == "auto":
        if x0 is not None:
            method = "local"
        elif bounds is not None:
            method = "global"
        else:
            raise ValueError("need x0 or bounds")

    try:
        if method == "local":
            result = scipy.optimize.minimize(
                objective_scalar,
                np.asarray(x0),
                method="L-BFGS-B" if bounds is not None else "Nelder-Mead",
                bounds=bounds,
                options={"maxiter": max_iter},
                callback=callback,
            )
            return result.x, result.fun, history

        elif method == "global":
            result = scipy.optimize.differential_evolution(
                objective_scalar,
                bounds,
                maxiter=max_iter,
                callback=lambda xk, _: callback(xk),
            )
            return result.x, result.fun, history

    except EarlyStop:
        return history["x"][-1], history["objective"][-1], history


def plot_optimization_trajectory(history, ax=None):
    xs = np.asarray(history["x"])
    residual_sq = np.asarray(history["residual_sq"])

    if xs.ndim != 2:
        raise ValueError("history['x'] must be 2D")

    n_iter, n_param = xs.shape

    if residual_sq.ndim == 1:
        residual_sq = residual_sq[:, None]

    if residual_sq.ndim != 2:
        raise ValueError("history['residual_sq'] must be 1D or 2D")

    _, n_output = residual_sq.shape

    n_axes = n_param + n_output

    if ax is None:
        fig, ax = plt.subplots(n_axes, 1, figsize=(7, 4 * n_axes))
    else:
        fig = ax[0].figure if isinstance(ax, (list, tuple, np.ndarray)) else ax.figure

    if not isinstance(ax, (list, tuple, np.ndarray)):
        raise ValueError("ax must be an iterable of Axes")

    if len(ax) != n_axes:
        raise ValueError("len(ax) must equal n_param + n_output")

    k = 0
    for i in range(n_param):
        ax[k].plot(xs[:, i], marker="o")
        ax[k].set_ylabel(f"x[{i}]")
        ax[k].grid(True)
        k += 1

    for j in range(n_output):
        ax[k].plot(residual_sq[:, j], marker="o")
        ax[k].set_ylabel(f"residual_sq[{j}]")
        ax[k].grid(True)
        k += 1

    ax[-1].set_xlabel("iteration")

    return fig, ax


def test_find_input_and_plot():
    def forward_fn(x):
        return np.array([
            np.sin(x[0]) + x[1] ** 2,
            x[0] ** 2 + np.cos(x[1]),
        ])

    target_value = np.array([1.0, 1.5])

    x0 = np.array([0.5, 0.5])
    bounds = [(-3, 3), (-3, 3)]

    x_opt, objective_opt, history = find_input_for_target_with_max_iter(
        forward_fn=forward_fn,
        target_value=target_value,
        x0=x0,
        bounds=bounds,
        max_iter=200,
        patience=20,
        verbose=True,
        plot_history=True,
        save_figdir='./test_results',
    )

    fig, ax = plot_optimization_trajectory(history)
    fig.suptitle("Optimization trajectory")
    plt.tight_layout()
    plt.show()

    results = {
        "x_opt": x_opt,
        "objective_opt": objective_opt,
        "history": history,
        "fig": fig,
        "ax": ax,
    }
    return results