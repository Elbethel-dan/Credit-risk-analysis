from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# ==============================
# Named Constants 
# ==============================

DEFAULT_FIGSIZE_LARGE = (12, 6)
DEFAULT_FIGSIZE_MEDIUM = (8, 5)
DEFAULT_FIGSIZE_SMALL = (5, 3)

DEFAULT_TITLE_SIZE = 14
DEFAULT_LABEL_SIZE = 12
SMALL_TITLE_SIZE = 10
SMALL_LABEL_SIZE = 9

DEFAULT_DPI = 150
DEFAULT_SAVEFIG_DPI = 300

DEFAULT_BOX_COLOR = "#2f7fc5"
DEFAULT_OUTLIER_COLOR = "red"
DEFAULT_HIST_COLOR = "blue"


# ==============================
# Configuration Dataclass
# ==============================

@dataclass
class PlotConfig:
    style: str = "white"
    figure_dpi: int = DEFAULT_DPI
    savefig_dpi: int = DEFAULT_SAVEFIG_DPI


# ==============================
# Plotter Class
# ==============================

class Plotter:
    """
    Reusable Plotter module for common visualizations.

    Supported plots:
    - Line plot
    - Box plot
    - Histogram
    - Bar plot
    - Correlation heatmap
    """

    def __init__(self, config: Optional[PlotConfig] = None) -> None:
        self.config = config or PlotConfig()
        self._initialize_style()

    # ==============================
    # Internal Utility Methods
    # ==============================

    def _initialize_style(self) -> None:
        sns.set_style(self.config.style)
        plt.rcParams["figure.dpi"] = self.config.figure_dpi
        plt.rcParams["savefig.dpi"] = self.config.savefig_dpi

    def _setup_figure(self, figsize: tuple) -> None:
        plt.figure(figsize=figsize)

    def _apply_labels(
        self,
        title: str,
        xlabel: Optional[str],
        ylabel: Optional[str],
        title_size: int = DEFAULT_TITLE_SIZE,
        label_size: int = DEFAULT_LABEL_SIZE,
    ) -> None:
        plt.title(title, fontsize=title_size)
        if xlabel:
            plt.xlabel(xlabel, fontsize=label_size)
        if ylabel:
            plt.ylabel(ylabel, fontsize=label_size)

    def _finalize_plot(self) -> None:
        sns.despine()
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    # ==============================
    # Public Plot Methods
    # ==============================

    def line_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        vlines: Optional[List[Dict[str, Any]]] = None,
    ) -> None:

        self._setup_figure(DEFAULT_FIGSIZE_LARGE)

        sns.lineplot(x=x, y=y, data=data)

        self._apply_labels(
            title or f"Line Plot of {y} over {x}",
            xlabel or x,
            ylabel or y,
        )

        if vlines:
            for line in vlines:
                plt.axvline(
                    pd.Timestamp(line["date"]),
                    color=line.get("color", "red"),
                    linestyle=line.get("linestyle", "--"),
                    label=line.get("label"),
                )

        if vlines:
            plt.legend()

        self._finalize_plot()

    def box_plot(
        self,
        data: pd.DataFrame,
        y: str,
        title: Optional[str] = None,
        ylabel: Optional[str] = None,
    ) -> None:

        self._setup_figure(DEFAULT_FIGSIZE_SMALL)

        sns.boxplot(
            y=y,
            data=data,
            color=DEFAULT_BOX_COLOR,
            flierprops={
                "marker": "o",
                "markerfacecolor": DEFAULT_OUTLIER_COLOR,
                "markersize": 4,
            },
        )

        self._apply_labels(
            title or f"Box Plot of {y}",
            None,
            ylabel or y,
            title_size=SMALL_TITLE_SIZE,
            label_size=SMALL_LABEL_SIZE,
        )

        self._finalize_plot()

    def histogram_plot(
        self,
        data: Union[pd.Series, np.ndarray],
        bins: int = 30,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: str = DEFAULT_HIST_COLOR,
    ) -> None:

        self._setup_figure(DEFAULT_FIGSIZE_MEDIUM)

        plt.hist(data, bins=bins, color=color, edgecolor="black")

        self._apply_labels(
            title or "Histogram",
            xlabel or "Value",
            ylabel or "Frequency",
        )

        self._finalize_plot()

    def bar_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ) -> None:

        self._setup_figure(DEFAULT_FIGSIZE_MEDIUM)

        sns.barplot(x=x, y=y, data=data)

        self._apply_labels(
            title or f"Bar Plot of {y} by {x}",
            xlabel or x,
            ylabel or y,
        )

        self._finalize_plot()

    def correlation_heatmap(
        self,
        data: pd.DataFrame,
        method: str = "pearson",
        title: Optional[str] = None,
    ) -> None:

        self._setup_figure(DEFAULT_FIGSIZE_LARGE)

        corr_matrix: pd.DataFrame = data.corr(method=method)

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
        )

        self._apply_labels(
            title or f"Correlation Heatmap ({method.capitalize()})",
            None,
            None,
        )

        self._finalize_plot()


    def count_plot(
        self,
        data: pd.DataFrame,
        column: str,
        title: Optional[str] = None,
        horizontal: bool = True,
        sort: bool = True,
    ) -> None:
        """
        Draws a count plot for categorical variables.

        Parameters:
        - data: DataFrame
        - column: categorical column name
        - title: plot title
        - horizontal: whether bars should be horizontal
        - sort: whether to sort categories by frequency
        """

        self._setup_figure(DEFAULT_FIGSIZE_LARGE)

        order = None
        if sort:
            order = data[column].value_counts().index

        if horizontal:
            sns.countplot(
                y=data[column],
                order=order,
            )
        else:
            sns.countplot(
                x=data[column],
                order=order,
            )

        self._apply_labels(
            title or f"Count Plot of {column}",
            None if horizontal else column,
            column if horizontal else None,
        )

        self._finalize_plot()

