"""Analyze how modules in this common_func directory are used by a repository."""

from __future__ import annotations

import ast
import os
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "node_modules",
    "venv",
}

_ANALYZER_MODULE = Path(__file__).stem


def _discover_function_modules(common_func_path: Path) -> set[str]:
    """Return module names represented by ``*_functions.py`` files."""
    return {
        path.stem
        for path in common_func_path.glob("*_functions.py")
        if path.is_file() and path.stem != _ANALYZER_MODULE
    }


def _top_level_aliases(tree: ast.Module, module_names: set[str]) -> dict[str, str]:
    """Map locally bound import aliases to common_func module names."""
    aliases = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Import):
            continue
        for imported in statement.names:
            module_name = imported.name.rsplit(".", 1)[-1]
            if module_name not in module_names:
                continue

            # ``import package.module`` binds ``package``, not ``module``.
            if imported.asname is not None:
                aliases[imported.asname] = module_name
            elif "." not in imported.name:
                aliases[imported.name] = module_name
    return aliases


def _count_alias_attributes(
    tree: ast.Module, aliases: dict[str, str]
) -> dict[str, Counter]:
    """Count direct ``alias.member`` AST attribute accesses."""
    counts = {module_name: Counter() for module_name in set(aliases.values())}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
        ):
            counts[aliases[node.value.id]][node.attr] += 1
    return counts


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _iter_python_files(
    repository_path: Path,
    output_dir: Path,
    exclude_dirs: Iterable[str],
):
    excluded_names = set(exclude_dirs)
    for root, dirs, files in os.walk(repository_path):
        root_path = Path(root)
        dirs[:] = [
            name
            for name in dirs
            if name not in excluded_names
            and not name.startswith(".")
            and not _is_relative_to((root_path / name).resolve(), output_dir)
        ]
        for filename in files:
            if filename.endswith(".py"):
                yield root_path / filename


def _sort_counts(counts):
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _write_report(
    report_path: Path,
    repository_path: Path,
    common_func_path: Path,
    scanned_file_count: int,
    module_counts: dict[str, Counter],
    warnings: list[str],
):
    totals = {module: sum(counts.values()) for module, counts in module_counts.items()}
    total_usage = sum(totals.values())

    lines = [
        "common_func usage report",
        "=" * 80,
        f"Repository: {repository_path}",
        f"common_func directory: {common_func_path}",
        f"Python files scanned: {scanned_file_count}",
        f"Total member uses: {total_usage}",
        "",
        "Module frequency",
        "-" * 80,
    ]

    nonzero_totals = {module: count for module, count in totals.items() if count}
    if nonzero_totals:
        width = max(len(module) for module in nonzero_totals)
        for module, count in _sort_counts(nonzero_totals):
            percentage = count / total_usage * 100 if total_usage else 0
            lines.append(f"{module:<{width}}  {count:>8}  {percentage:>7.2f}%")
    else:
        lines.append("No matching common_func member usage found.")

    for module, total in _sort_counts(nonzero_totals):
        lines.extend(["", f"[{module}] total={total}", "-" * 80])
        member_width = max(len(member) for member in module_counts[module])
        for member, count in _sort_counts(module_counts[module]):
            percentage = count / total * 100
            lines.append(
                f"{member:<{member_width}}  {count:>8}  {percentage:>7.2f}%"
            )

    lines.extend(["", "Warnings", "-" * 80])
    lines.extend(warnings or ["None"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_bar_chart(
    items,
    output_path: Path,
    title: str,
    xlabel: str = "Usage count",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(items)
    figure_height = max(3.2, min(12.0, 0.38 * max(len(items), 1) + 1.8))
    fig, ax = plt.subplots(figsize=(10, figure_height))

    if items:
        labels = [label for label, _ in reversed(items)]
        values = [value for _, value in reversed(items)]
        bars = ax.barh(labels, values, color="#4472C4")
        ax.bar_label(bars, padding=3)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
    else:
        ax.text(
            0.5,
            0.5,
            "No matching usage found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_cf_usage(
    repository_path,
    output_dir,
    common_func_path=None,
    top_n=20,
    exclude_dirs=None,
):
    """Analyze common_func member usage in all Python files under a repository.

    Parameters
    ----------
    repository_path : str or os.PathLike
        Root directory of the repository to scan recursively.
    output_dir : str or os.PathLike
        Directory in which the text report and PNG charts are written.
    common_func_path : str or os.PathLike, optional
        Directory containing the ``*_functions.py`` modules. Defaults to this
        file's directory.
    top_n : int, default=20
        Maximum number of members displayed in each member chart.
    exclude_dirs : iterable of str, optional
        Additional directory basenames to skip while scanning.

    Returns
    -------
    dict
        Module totals, member counts, scan metadata, warnings, and output paths.
    """
    repository_path = Path(repository_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    common_func_path = Path(
        common_func_path if common_func_path is not None else Path(__file__).parent
    ).expanduser().resolve()

    if not repository_path.is_dir():
        raise NotADirectoryError(f"Repository directory does not exist: {repository_path}")
    if not common_func_path.is_dir():
        raise NotADirectoryError(
            f"common_func directory does not exist: {common_func_path}"
        )
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")

    module_names = _discover_function_modules(common_func_path)
    if not module_names:
        raise ValueError(
            f"No *_functions.py modules found in common_func directory: "
            f"{common_func_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_names = DEFAULT_EXCLUDE_DIRS | set(exclude_dirs or ())
    module_counts = {module: Counter() for module in sorted(module_names)}
    warnings = []
    scanned_file_count = 0

    for python_file in _iter_python_files(
        repository_path, output_dir, excluded_names
    ):
        scanned_file_count += 1
        try:
            source = python_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(python_file))
        except (OSError, UnicodeError, SyntaxError) as error:
            relative_path = python_file.relative_to(repository_path)
            warnings.append(
                f"{relative_path}: {error.__class__.__name__}: {error}"
            )
            continue

        aliases = _top_level_aliases(tree, module_names)
        for module, counts in _count_alias_attributes(tree, aliases).items():
            module_counts[module].update(counts)

    report_path = output_dir / "cf_usage_report.txt"
    modules_plot_path = output_dir / "cf_usage_modules.png"
    top_members_plot_path = output_dir / "cf_usage_top_members.png"

    _write_report(
        report_path,
        repository_path,
        common_func_path,
        scanned_file_count,
        module_counts,
        warnings,
    )

    module_totals = {
        module: sum(counts.values()) for module, counts in module_counts.items()
    }
    nonzero_module_totals = {
        module: count for module, count in module_totals.items() if count
    }
    _save_bar_chart(
        _sort_counts(nonzero_module_totals),
        modules_plot_path,
        "common_func module usage",
    )

    global_members = {
        f"{module}.{member}": count
        for module, counts in module_counts.items()
        for member, count in counts.items()
    }
    _save_bar_chart(
        _sort_counts(global_members)[:top_n],
        top_members_plot_path,
        f"Top {top_n} common_func members",
    )

    module_plot_paths = {}
    for module, counts in module_counts.items():
        if not counts:
            continue
        plot_path = output_dir / f"cf_usage_{module}.png"
        _save_bar_chart(
            _sort_counts(counts)[:top_n],
            plot_path,
            f"Top {top_n} members in {module}",
        )
        module_plot_paths[module] = str(plot_path)

    return {
        "repository_path": str(repository_path),
        "common_func_path": str(common_func_path),
        "scanned_file_count": scanned_file_count,
        "total_usage_count": sum(module_totals.values()),
        "module_totals": module_totals,
        "member_counts": {
            module: dict(_sort_counts(counts))
            for module, counts in module_counts.items()
        },
        "warnings": warnings,
        "output_paths": {
            "report": str(report_path),
            "modules_plot": str(modules_plot_path),
            "top_members_plot": str(top_members_plot_path),
            "module_plots": module_plot_paths,
        },
    }
