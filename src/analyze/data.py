from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.io.wavfile import write

from constants import FIBER_BUNDLE_A, FIBER_BUNDLE_B, ABDOMEN_FIBER_NAMES


@dataclass
class Audio:
    time: npt.NDArray[np.float64]
    hz: int
    data: npt.NDArray[np.float64]

    def window(self, start, end):
        mask = (self.time >= start) & (self.time <= end)
        return Audio(
            self.time[mask],
            self.hz,
            self.data[mask]
        )

    def write(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        write(path, self.hz, self.data)


@dataclass
class FiberData:
    chest: Audio
    abdomen: dict[str, Audio]

    def window(self, start, end):
        return FiberData(
            self.chest.window(start, end) if self.chest is not None else None,
            {name: e.window(start, end) for (name, e) in self.abdomen.items()},
        )

    @staticmethod
    def apply(filter):
        def run_all(data: FiberData):
            return FiberData(
                filter(data.chest),
                {s: filter(e) for s, e, in data.abdomen.items()},
            )

        run_all.__name__ = filter.__name__ + "_ALL"
        return run_all

    @staticmethod
    def apply_abdomen(filter):
        def run_all(data: FiberData):
            return FiberData(
                data.chest,
                {s: filter(e) for s, e, in data.abdomen.items()},
            )

        run_all.__name__ = filter.__name__ + "_ALL"
        return run_all

    @staticmethod
    def apply_chest(filter):
        def run_all(data: FiberData):
            return FiberData(
                filter(data.chest),
                data.abdomen,
            )

        run_all.__name__ = filter.__name__ + "_ALL"
        return run_all


@dataclass
class FiberPair:
    chest: Audio
    abdomen: Audio

    @staticmethod
    def apply(filter):
        def run_dual(data: FiberPair):
            return FiberPair(
                filter(data.chest),
                filter(data.abdomen),
            )

        run_dual.__name__ = filter.__name__ + "_DUAL"
        return run_dual


def load_fibers(path):
    raw_chest = np.load(path / FIBER_BUNDLE_A)
    chest = Audio(
        raw_chest[:, 0],
        round(1 / (raw_chest[1, 0] - raw_chest[0, 0])),
        raw_chest[:, 1],
    )

    raw_abdomen = np.load(path / FIBER_BUNDLE_B)
    abdomen_time = raw_abdomen[:, 0]
    abdomen_hz = round(1 / (abdomen_time[1] - abdomen_time[0]))
    abdomen = {ABDOMEN_FIBER_NAMES[0]: Audio(
        raw_chest[:, 0],
        round(1 / (raw_chest[1, 0] - raw_chest[0, 0])),
        raw_chest[:, 2],
    )}

    for i in range(1, raw_abdomen.shape[1]):
        data = raw_abdomen[:, i]
        abdomen[ABDOMEN_FIBER_NAMES[i]] = Audio(
            abdomen_time,
            abdomen_hz,
            data
        )

    return FiberData(
        chest,
        abdomen,
    )


def load_data(path):
    fibers = load_fibers(Path(path))
    print("Fiber data loaded: ")
    print(f"  Abdomen: {" | ".join([f"{name} ({audio.hz}hz)" for (name, audio) in fibers.abdomen.items()])}")
    print(f"  Chest ({fibers.chest.hz} hz)")
    return fibers


def load_no_chest_data(path: str):
    COLS = "2A", "2B", "2C"
    path = Path(path)

    raw_abdomen = np.load(path / FIBER_BUNDLE_B)
    abdomen_time = raw_abdomen[:, 0]
    abdomen_hz = round(1 / (abdomen_time[1] - abdomen_time[0]))
    abdomen = {}

    for i in range(1, raw_abdomen.shape[1]):
        data = raw_abdomen[:, i]
        abdomen[COLS[i - 1]] = Audio(
            abdomen_time,
            abdomen_hz,
            data
        )


def load_no_chest_data_FULL(path: str):
    COLS = "1A", "1B", "2A", "2B", "2C", "2D"
    path = Path(path)

    abdomen = {}

    raw_chest = np.load(path / FIBER_BUNDLE_A)
    chest_time = raw_chest[:, 0]
    chest_hz = round(1 / (chest_time[1] - chest_time[0]))

    for i in range(1, raw_chest.shape[1]):
        data = raw_chest[:, i]
        abdomen[COLS[i - 1]] = Audio(
            chest_time,
            chest_hz,
            data
        )

    raw_abdomen = np.load(path / FIBER_BUNDLE_B)
    abdomen_time = raw_abdomen[:, 0]
    abdomen_hz = round(1 / (abdomen_time[1] - abdomen_time[0]))

    for i in range(1, raw_abdomen.shape[1]):
        data = raw_abdomen[:, i]
        abdomen[COLS[i - 1 + 2]] = Audio(
            abdomen_time,
            abdomen_hz,
            data
        )

    # fft_x = abdomen["2A"].data
    # N = fft_x.shape[0]
    #
    # fft_abdomen = np.fft.rfft(fft_x)
    # fft_freq = np.fft.rfftfreq(N, d=1/abdomen_hz)
    # magnitude = np.abs(fft_abdomen)
    #
    # fig, ax = plt.subplots(figsize=(10, 4))
    #
    # ax.plot(fft_freq, magnitude, linewidth=0.8, color="#2c7bb6")
    # ax.set_xlim(0, 250)
    # ax.set_xlabel("Frequency (Hz)", fontsize=12)
    # ax.set_ylabel("Magnitude", fontsize=12)
    # ax.set_title("FFT — Abdomen channel 2A", fontsize=13)
    #
    # # Light grid on y only; keeps it readable without clutter
    # ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    # ax.set_axisbelow(True)
    # ax.spines[["top", "right"]].set_visible(False)
    #
    # fig.tight_layout()
    # fig.savefig(out_path / "fft.png", dpi=150)
    # plt.close(fig)

    return FiberData(
        None,
        abdomen,
    )


def windowed(start, end):
    """Generic windowing stage: works on any value with a ``.window(start, end)``
    method (FiberData, SOTResult, ...). The same stage is reused by both the
    fiber pipeline and the SOT pipeline."""

    def select_window(data):
        print(f"Selecting window {start}s-{end}s")
        return data.window(start, end)

    return select_window


def use_fiber(fiber):
    def select_fiber(data: FiberData):
        return FiberPair(
            data.chest,
            data.abdomen[fiber]
        )

    return select_fiber
