import os
from pathlib import Path


PROJECT_DIR = str(Path(__file__).resolve().parents[2]) + "/"

BANNER_TEST_DIR = "Banner_data/Banner_test_20251220"
DEFAULT_DATA_DIR = PROJECT_DIR + BANNER_TEST_DIR

# Fine-tuned SSNet checkpoints (lib/tune-ssnet), used by the main pipeline.
FETAL_MODEL_PATH = os.path.join(PROJECT_DIR, "lib/tune-ssnet/models/tuned-model-v3/model_best.pt")
FETAL_MODEL_CFG = os.path.join(PROJECT_DIR, "lib/tune-ssnet/models/tuned-model-v3/model.yaml")
MATERNAL_MODEL_PATH = os.path.join(PROJECT_DIR, "lib/tune-ssnet/models/maternal-tuned-model-v2/model_best.pt")
MATERNAL_MODEL_CFG = os.path.join(PROJECT_DIR, "lib/tune-ssnet/models/maternal-tuned-model-v2/model.yaml")

# Base pretrained NeoSSNet checkpoint (lib/neossnet submodule), used to
# synthesize the "lung" training target in generate_training_snippets.py.
NEOSSNET_MODEL_PATH = os.path.join(PROJECT_DIR, "lib/neossnet/models/model_best.pt")
NEOSSNET_MODEL_CFG = os.path.join(PROJECT_DIR, "lib/neossnet/models/model.yaml")

# ---------------------------------------------------------------------------
# Raw data filenames (Banner_data-style patient directories)
# ---------------------------------------------------------------------------

FIBER_BUNDLE_A = "ps4000.npy"   # chest device bundle
FIBER_BUNDLE_B = "ps3000a.npy"  # abdomen device bundle
MIC_FILE = "microphone.wav"
PVS_FILE = "pvs.npy"

ABDOMEN_FIBER_NAMES = ["1B", "2A", "2B", "2C", "2D"]

# ---------------------------------------------------------------------------
# Sample rates
# ---------------------------------------------------------------------------

NEOSSNET_MODEL_HZ = 4000           # NeoSSNet / tune-ssnet model sample rate
XCORR_TARGET_FS = 200.0   # grid rate for impulse-train cross-correlation lag estimators

# ---------------------------------------------------------------------------
# Acoustic bandpass bands (Hz)
# ---------------------------------------------------------------------------

MATERNAL_ACOUSTIC_BAND_HZ = (40.0, 80.0)        # chest bandpass before maternal beat detection
FETAL_ACOUSTIC_BAND_HZ = (190.0, 220.0)         # abdomen bandpass for fetal cardiac detection
FETAL_ACOUSTIC_BAND_NARROW_HZ = (190.0, 210.0)  # narrower fetal band used after NeoSSNet separation
SOURCE_PREP_BAND_HZ = (40.0, 200.0)             # wideband prep filter before ICA/source separation
BROADBAND_FILTER_HZ = (20.0, 250.0)             # initial wideband filter before source separation
POWERLINE_NOTCH_HZ = 50                         # mains hum notch frequency

# ---------------------------------------------------------------------------
# Physiological BPM ranges
# ---------------------------------------------------------------------------

MATERNAL_BPM_RANGE = (45.0, 140.0)
FETAL_BPM_RANGE = (90.0, 220.0)