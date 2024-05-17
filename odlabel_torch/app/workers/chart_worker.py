import threading
from queue import Queue
from typing import Union
from .utils.chart_utils import (
    input_chart,
    output_charts,
    close_matplotlib_figures,
    display_chart,
)


class ChartWorker(threading.Thread):
    def __init__(
        self,
        images_folder,
        chart_colors,
        train_detections=None,
        val_detections=None,
        actual_train_count=0,
        actual_val_count=0,
    ):
        super().__init__()
        self.images_folder = images_folder
        self.chart_colors = chart_colors
        self.train_detections = train_detections if train_detections is not None else []
        self.val_detections = val_detections if val_detections is not None else []
        self.actual_train_count = actual_train_count
        self.actual_val_count = actual_val_count
        self.input_figures = None
        self.train_figures = None
        self.val_figures = None
        self.image_files = []
        self.queue: Union[Queue, None] = None

    def run(self):
        self.process_input_charts()
        self.process_output_charts()

    def process_input_charts(self):
        try:
            (
                fig_format,
                fig_resolution,
                fig_quality,
                fig_colorspace,
                self.image_files,
            ) = input_chart(self.images_folder, self.chart_colors)
            self.input_figures = [
                fig_format,
                fig_resolution,
                fig_quality,
                fig_colorspace,
            ]
            if self.queue is not None:
                self.queue.put(("input_figures", self.input_figures))
        except Exception as e:
            if self.queue is not None:
                self.queue.put(("error", str(e)))

    def process_output_charts(self):
        if self.train_detections:
            train_fig_count, train_fig_conf, train_fig_labels, train_fig_heatmap = (
                output_charts(self.train_detections, self.actual_train_count)
            )
            self.train_figures = [
                train_fig_count,
                train_fig_conf,
                train_fig_labels,
                train_fig_heatmap,
            ]
            if self.queue is not None:
                self.queue.put(("train_figures", self.train_figures))
        if self.val_detections:
            val_fig_count, val_fig_conf, val_fig_labels, val_fig_heatmap = (
                output_charts(self.val_detections, self.actual_val_count)
            )
            self.val_figures = [
                val_fig_count,
                val_fig_conf,
                val_fig_labels,
                val_fig_heatmap,
            ]
            if self.queue is not None:
                self.queue.put(("val_figures", self.val_figures))

    def get_input_figures(self):
        return self.input_figures

    def get_train_figures(self):
        return self.train_figures

    def get_val_figures(self):
        return self.val_figures

    def close_figures(self):
        close_matplotlib_figures()

    def display_chart_in_frame(self, fig, frame, row, col):
        display_chart(fig, frame, row, col)
