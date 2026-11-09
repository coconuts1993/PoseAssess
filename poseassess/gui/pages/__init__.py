from .project_page import ProjectPage
from .calibration_page import CalibrationPage
from .wii_board_page import WiiBoardPage
from .videos_page import VideosPage
from .capture_page import CapturePage
from .run_page import RunPage
from .viz3d_page import Viz3DPage
from .results_page import ResultsPage
from .benchmark_page import BenchmarkPage

# Navigation order. The Wii Balance Board pages are inserted with sub-numbers ("2b.", "3b.")
# so the existing pages keep their titles/numbers.
ALL_PAGES = [ProjectPage, CalibrationPage, WiiBoardPage, VideosPage, CapturePage, RunPage,
             Viz3DPage, ResultsPage, BenchmarkPage]

__all__ = ["ProjectPage", "CalibrationPage", "WiiBoardPage", "VideosPage", "CapturePage",
           "RunPage", "Viz3DPage", "ResultsPage", "BenchmarkPage", "ALL_PAGES"]
