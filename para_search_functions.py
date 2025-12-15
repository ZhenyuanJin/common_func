import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import os


class EarlyStop(Exception):
    pass


def find_input_for_target_with_max_iter(
    func,
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
    plot_every=None,
    save_figdir=None,
):
    target_array = np.asarray(target_value)
    weights_array = np.asarray(weights) if weights is not None else None

    history = {
        "x": [],
        "func_value": [],
        "objective": [],
    }

    func_cache = {}

    def x_to_key(x):
        return tuple(np.round(np.asarray(x, dtype=float), 12))

    def eval_func(x):
        key = x_to_key(x)
        if key not in func_cache:
            func_cache[key] = np.asarray(func(x))
        return func_cache[key]

    def compute_objective(func_value):
        residual = func_value - target_array
        if weights_array is not None:
            residual = residual * weights_array
        return np.sum(residual ** 2)

    def objective_scalar(x):
        func_value = eval_func(x)
        return compute_objective(func_value)

    best_objective = np.inf
    stall_counter = 0

    def callback(xk):
        nonlocal best_objective, stall_counter

        func_value = eval_func(xk)
        obj = compute_objective(func_value)

        history["x"].append(np.copy(xk))
        history["func_value"].append(func_value)
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

        if (
            plot_history
            and plot_every is not None
            and plot_every > 0
            and len(history["x"]) % plot_every == 0
        ):
            fig, ax = plot_optimization_trajectory(history)
            fig.suptitle("Optimization trajectory")
            plt.tight_layout()
            if save_figdir is not None:
                os.makedirs(save_figdir, exist_ok=True)
                fig.savefig(f"{save_figdir}/optimization_trajectory")
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
            if bounds is not None:
                print('Bounds are not used in local optimization.')
            result = scipy.optimize.minimize(
                objective_scalar,
                np.asarray(x0),
                method="Nelder-Mead",
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

    except EarlyStop as e:
        print("Early stopping:", e)
        return history["x"][-1], history["objective"][-1], history


def plot_optimization_trajectory(history, ax=None):
    xs = np.asarray(history["x"])
    func_value = np.asarray(history["func_value"])

    if xs.ndim != 2:
        raise ValueError("history['x'] must be 2D")

    if func_value.ndim == 1:
        func_value = func_value[:, None]

    if func_value.ndim != 2:
        raise ValueError("history['func_value'] must be 1D or 2D")

    n_iter, n_param = xs.shape
    _, n_output = func_value.shape

    n_axes = n_param + n_output

    if ax is None:
        fig, ax = plt.subplots(n_axes, 1, figsize=(7, 4 * n_axes))
    else:
        fig = ax[0].figure

    if len(ax) != n_axes:
        raise ValueError("len(ax) must equal n_param + n_output")

    k = 0
    for i in range(n_param):
        ax[k].plot(xs[:, i], marker="o")
        ax[k].set_ylabel(f"x[{i}]")
        ax[k].grid(True)
        k += 1

    for j in range(n_output):
        ax[k].plot(func_value[:, j], marker="o")
        ax[k].set_ylabel(f"func_value[{j}]")
        ax[k].grid(True)
        k += 1

    ax[-1].set_xlabel("iteration")

    return fig, ax


def test_find_input_and_plot():
    def func(x):
        print(f"func called with x={x}")
        return np.array([
            np.sin(x[0]) + x[1] ** 2,
            x[0] ** 2 + np.cos(x[1]),
        ])

    target_value = np.array([10.0, 15.5])

    x0 = np.array([0.5, 0.5])
    bounds = [(-10, 10), (-10, 10)]

    x_opt, objective_opt, history = find_input_for_target_with_max_iter(
        func=func,
        target_value=target_value,
        x0=x0,
        bounds=bounds,
        max_iter=200,
        patience=20,
        verbose=True,
        plot_history=True,
        plot_every=5,
        save_figdir="./test_results",
        tol_objective=0.,
        tol_x=0.,
    )

    fig, ax = plot_optimization_trajectory(history)
    fig.suptitle("Optimization trajectory")
    plt.tight_layout()
    plt.show()

    return x_opt, objective_opt, history