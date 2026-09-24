from .project_page import ProjectPage
from .calibration_page import CalibrationPage
from .videos_page import VideosPage
from .run_page import RunPage
from .viz3d_page import Viz3DPage
from .results_page import ResultsPage
from .benchmark_page import BenchmarkPage

ALL_PAGES = [ProjectPage, CalibrationPage, VideosPage, RunPage, Viz3DPage,
             ResultsPage, BenchmarkPage]

__all__ = ["ProjectPage", "CalibrationPage", "VideosPage", "RunPage",
           "Viz3DPage", "ResultsPage", "BenchmarkPage", "ALL_PAGES"]
