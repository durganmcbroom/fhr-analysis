import numpy as np
from scipy.signal import cheby1, butter, filtfilt, sosfiltfilt, iirnotch

from analyze.data import Audio, FiberData, FiberPair


def bp_filter(audio, low, high, order=4, filter_type='cheby1', rp=1):
    if filter_type == 'cheby1':
        sos = cheby1(order, rp=rp, Wn=[low, high], fs=audio.hz, btype='bandpass', output='sos')
    else:
        sos = butter(order, [low, high], fs=audio.hz, btype='bandpass', output='sos')
    return Audio(
        audio.time,
        audio.hz,
        sosfiltfilt(sos, audio.data, axis=0)
    )

def notch_filter(audio, freq, quality=30):
    b,a = iirnotch(freq, quality, audio.hz)

    return Audio(
        audio.time,
        audio.hz,
        filtfilt(b,a, audio.data)
    )


def on_all(filter):
    def apply(data):
        return [
            filter(e)
            for e in data
        ]
    apply.__name__ = f"{filter.__name__}_ALL"
    return apply

def abdomen_bp(low, high, filter_type="cheby1"):
    def run_abdomen_bp(data):

        if isinstance(data, FiberData):
            return FiberData(
                data.chest,
                {name: bp_filter(audio, low, high, filter_type=filter_type)
                for name, audio in data.abdomen.items()}
            )
        else:
            return FiberPair(
                data.chest,
                bp_filter(data.abdomen, low, high, filter_type=filter_type),
            )

    run_abdomen_bp.__name__ = "abdomen_bp"
    return run_abdomen_bp


def bp(low, high, filter_type="cheby1"):
    def run_bandpass(data):
        return bp_filter(data, low, high, filter_type=filter_type)

    return run_bandpass


def notch(freq):
    def run_notch(data):
        return notch_filter(data, freq)

    return run_notch