import asyncio
from threading import Thread

from bleak import BleakClient, BleakScanner
import numpy as np
from bitstring import BitArray
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QFileDialog, QDoubleSpinBox
)
from PyQt5.QtCore import pyqtSignal, QObject
from qasync import QEventLoop, asyncClose
import pyqtgraph as pg
import time
import ctypes
import threading
import os
from picosdk.ps3000a import ps3000a
# from picosdk.ps4000 import ps4000
from picosdk.functions import assert_pico_ok
import sounddevice as sd
from scipy.io.wavfile import read as wav_read


DISPLAY_TIME = 10
SAVE_TIME = 300


class PolarVeritySense:
    BATTERY_LEVEL_UUID = "00002a19-0000-1000-8000-00805f9b34fb"

    PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
    PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"

    PPG_START = bytearray([0x02, 0x01, 0x00, 0x01, 0x37, 0x00, 0x01, 0x01, 0x16, 0x00, 0x04, 0x01, 0x04])
    PPG_STOP = bytearray([0x03, 0x01])

    MAX_EMIT_LEN = DISPLAY_TIME * 2 * 55

    def __init__(self, device, signal, global_start_time, session_dir):
        self.device = device
        self.signal = signal
        self.global_start_time = global_start_time
        self.session_dir = session_dir

        self.connected = False
        self.started = False

        self.previous_timestamp = -1
        self.t = []

        self.save_t = 0
        self.save_index = 0

        self.ppg0 = []
        self.ppg1 = []
        self.ppg2 = []
        self.ambient = []

    async def connect(self):
        if not self.connected:
            self.client = BleakClient(self.device)
            await self.client.connect()
            self.connected = True
    
    async def disconnect(self):
        if self.connected:
            await self.client.disconnect()
            self.connected = False

    async def get_battery_level(self):
        data = await self.client.read_gatt_char(PolarVeritySense.BATTERY_LEVEL_UUID)
        return data[0]

    async def start_ppg_stream(self):
        if not self.started:
            await self.client.write_gatt_char(PolarVeritySense.PMD_CONTROL, PolarVeritySense.PPG_START)
            await self.client.start_notify(PolarVeritySense.PMD_DATA, self.decode_data)
            self.started = True
            print("Starting PPG stream")
    
    async def stop_ppg_stream(self):
        if self.started:
            await self.client.stop_notify(PolarVeritySense.PMD_DATA)
            await self.client.write_gatt_char(PolarVeritySense.PMD_CONTROL, PolarVeritySense.PPG_STOP)
            self.started = False
            print("Stopping PPG stream")
            np.save(self.session_dir + "/pvs" , np.stack((self.t, self.ppg0, self.ppg1, self.ppg2, self.ambient), axis=1), allow_pickle=False)

    def decode_data(self, sender, data):
        if data[0] != 0x01:
            print("Unexpected measurement type")
        else:
            timestamp = PolarVeritySense.convert_to_unsigned_long(data, 1, 8) / 1e9 + 1211010636.1  # empirically determined
            frame_type = data[9]

            # print(time.time() - timestamp)  # approx. 0.5

            if frame_type != 0x80:
                print("Unexpected frame type")
            else:
                self.ppg0.append(PolarVeritySense.convert_array_to_signed_int(data, 10, 3))
                self.ppg1.append(PolarVeritySense.convert_array_to_signed_int(data, 13, 3))
                self.ppg2.append(PolarVeritySense.convert_array_to_signed_int(data, 16, 3))
                self.ambient.append(PolarVeritySense.convert_array_to_signed_int(data, 19, 3))
                samples_size = 1

                offset = 22
                while offset < len(data):
                    delta_size = data[offset]
                    sample_count = data[offset + 1]
                    offset += 2

                    samples = ''.join(format(byte, '08b')[::-1] for byte in data[offset: offset + (delta_size * sample_count // 2)])
                    for sample in range(0, len(samples), delta_size * 4):
                        deltas = [BitArray(bin=samples[sample + delta_size * i: sample + delta_size * (i + 1)][::-1]).int for i in range(4)]
                        ppg0 = self.ppg0[-1] + deltas[0]
                        ppg1 = self.ppg1[-1] + deltas[1]
                        ppg2 = self.ppg2[-1] + deltas[2]
                        ambient = self.ambient[-1] + deltas[3]

                        self.ppg0.append(ppg0)
                        self.ppg1.append(ppg1)
                        self.ppg2.append(ppg2)
                        self.ambient.append(ambient)
                        samples_size += 1

                    offset += delta_size * sample_count // 2
                
                if self.previous_timestamp == -1:
                    delta = 1 / 55
                    t = np.linspace(timestamp - delta * (samples_size - 1), timestamp, num=samples_size)
                else:
                    delta = (timestamp - self.previous_timestamp) / samples_size
                    t = np.linspace(self.previous_timestamp + delta, timestamp, num=samples_size)
                self.t = np.concatenate((self.t, t - self.global_start_time))
                self.previous_timestamp = timestamp

                self.signal.data.emit(np.stack((self.t, self.ppg0), axis=1)[-PolarVeritySense.MAX_EMIT_LEN:])

                if self.t[-1] > self.save_t + SAVE_TIME:
                    np.save(
                        self.session_dir + "/tmp/pvs_" + str(self.save_t),
                        np.stack((self.t, self.ppg0, self.ppg1, self.ppg2, self.ambient), axis=1)[self.save_index:],
                        allow_pickle=False
                    )
                    self.save_t += SAVE_TIME
                    self.save_index = len(self.t)

    @staticmethod
    def convert_array_to_signed_int(data, offset, length):
        return int.from_bytes(
            bytearray(data[offset : offset + length]), byteorder="little", signed=True,
        )

    @staticmethod
    def convert_to_unsigned_long(data, offset, length):
        return int.from_bytes(
            bytearray(data[offset : offset + length]), byteorder="little", signed=False,
        )


class PicoScope:
    channel_range = 8  # 10MV, 20MV, 50MV, 100MV, 200MV, 500MV, 1V, 2V, 5V, 10V, 20V, 50V
    v_range = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200][channel_range]

    sizeOfOneBuffer = 5000
    numBuffersToCapture = 1800
    totalSamples = sizeOfOneBuffer * numBuffersToCapture

    sampleInterval = ctypes.c_int32(200)
    sampleUnits = 3  # FS, PS, NS, US, MS, S
    actualSampleInterval = sampleInterval.value / 1e6

    MAX_EMIT_LEN = int(1e6 / sampleInterval.value * DISPLAY_TIME * 2)
    
    def __init__(self, ps, signal, global_start_time, session_dir):
        self.ps = ps
        self.signal = signal
        self.global_start_time = global_start_time
        self.session_dir = session_dir

        self.chandle = ctypes.c_int16()
        self.status = {}

        self.kill = False
        self.opened = threading.Event()  # unit is open and configured
        self.go = threading.Event()      # released to start streaming
        self.ready = threading.Event()   # streaming has begun

    def open(self):
        if self.ps == ps3000a:
            self.status["openunit"] = self.ps.ps3000aOpenUnit(ctypes.byref(self.chandle), None)
            try:
                assert_pico_ok(self.status["openunit"])
            except:
                powerStatus = self.status["openunit"]
                if powerStatus == 286:
                    self.status["changePowerSource"] = self.ps.ps3000aChangePowerSource(self.chandle, powerStatus)
                elif powerStatus == 282:
                    self.status["changePowerSource"] = self.ps.ps3000aChangePowerSource(self.chandle, powerStatus)
                else:
                    raise
                assert_pico_ok(self.status["changePowerSource"])

            self.status["setChA"] = self.ps.ps3000aSetChannel(self.chandle, self.ps.PS3000A_CHANNEL['PS3000A_CHANNEL_A'], 1, 1, PicoScope.channel_range, 0.0)
            assert_pico_ok(self.status["setChA"])
            self.status["setChB"] = self.ps.ps3000aSetChannel(self.chandle, self.ps.PS3000A_CHANNEL['PS3000A_CHANNEL_B'], 1, 1, PicoScope.channel_range, 0.0)
            assert_pico_ok(self.status["setChB"])
            self.status["setChC"] = self.ps.ps3000aSetChannel(self.chandle, self.ps.PS3000A_CHANNEL['PS3000A_CHANNEL_C'], 1, 1, PicoScope.channel_range, 0.0)
            assert_pico_ok(self.status["setChC"])
            self.status["setChD"] = self.ps.ps3000aSetChannel(self.chandle, self.ps.PS3000A_CHANNEL['PS3000A_CHANNEL_D'], 1, 1, PicoScope.channel_range, 0.0)
            assert_pico_ok(self.status["setChD"])

            self.bufferAMax = np.zeros(shape=PicoScope.sizeOfOneBuffer, dtype=np.int16)
            self.bufferBMax = np.zeros(shape=PicoScope.sizeOfOneBuffer, dtype=np.int16)
            self.bufferCMax = np.zeros(shape=PicoScope.sizeOfOneBuffer, dtype=np.int16)
            self.bufferDMax = np.zeros(shape=PicoScope.sizeOfOneBuffer, dtype=np.int16)

            self.status["setDataBuffersA"] = self.ps.ps3000aSetDataBuffers(
                self.chandle,
                self.ps.PS3000A_CHANNEL['PS3000A_CHANNEL_A'],
                self.bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                PicoScope.sizeOfOneBuffer,
                0,
                self.ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE']
            )
            assert_pico_ok(self.status["setDataBuffersA"])
            self.status["setDataBuffersB"] = self.ps.ps3000aSetDataBuffers(
                self.chandle,
                self.ps.PS3000A_CHANNEL['PS3000A_CHANNEL_B'],
                self.bufferBMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                PicoScope.sizeOfOneBuffer,
                0,
                self.ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE']
            )
            assert_pico_ok(self.status["setDataBuffersB"])
            self.status["setDataBuffersC"] = self.ps.ps3000aSetDataBuffers(
                self.chandle,
                self.ps.PS3000A_CHANNEL['PS3000A_CHANNEL_C'],
                self.bufferCMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                PicoScope.sizeOfOneBuffer,
                0,
                self.ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE']
            )
            assert_pico_ok(self.status["setDataBuffersC"])
            self.status["setDataBuffersD"] = self.ps.ps3000aSetDataBuffers(
                self.chandle,
                self.ps.PS3000A_CHANNEL['PS3000A_CHANNEL_D'],
                self.bufferDMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                PicoScope.sizeOfOneBuffer,
                0,
                self.ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE']
            )
            assert_pico_ok(self.status["setDataBuffersD"])
        else:
            self.status["openunit"] = self.ps.ps4000OpenUnit(ctypes.byref(self.chandle))
            assert_pico_ok(self.status["openunit"])
        
            self.status["setChA"] = self.ps.ps4000SetChannel(self.chandle, self.ps.PS4000_CHANNEL['PS4000_CHANNEL_A'], 1, 1, PicoScope.channel_range)
            assert_pico_ok(self.status["setChA"])
            self.status["setChB"] = self.ps.ps4000SetChannel(self.chandle, self.ps.PS4000_CHANNEL['PS4000_CHANNEL_B'], 1, 1, PicoScope.channel_range)
            assert_pico_ok(self.status["setChB"])

            self.bufferAMax = np.zeros(shape=PicoScope.sizeOfOneBuffer, dtype=np.int16)
            self.bufferBMax = np.zeros(shape=PicoScope.sizeOfOneBuffer, dtype=np.int16)

            self.status["setDataBuffersA"] = self.ps.ps4000SetDataBuffers(
                self.chandle,
                self.ps.PS4000_CHANNEL['PS4000_CHANNEL_A'],
                self.bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                PicoScope.sizeOfOneBuffer
            )
            assert_pico_ok(self.status["setDataBuffersA"])
            self.status["setDataBuffersB"] = self.ps.ps4000SetDataBuffers(
                self.chandle,
                self.ps.PS4000_CHANNEL['PS4000_CHANNEL_B'],
                self.bufferBMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                PicoScope.sizeOfOneBuffer
            )
            assert_pico_ok(self.status["setDataBuffersB"])
    
    def close_unit(self):
        if self.ps == ps3000a:
            self.ps.ps3000aCloseUnit(self.chandle)
        else:
            self.ps.ps4000CloseUnit(self.chandle)

    def streaming_callback(self, handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
        self.wasCalledBack = True
        destEnd = self.nextSample + noOfSamples
        sourceEnd = startIndex + noOfSamples

        self.bufferCompleteA[self.nextSample:destEnd] = self.bufferAMax[startIndex:sourceEnd]
        self.bufferCompleteB[self.nextSample:destEnd] = self.bufferBMax[startIndex:sourceEnd]
        self.bufferCompleteA[self.nextSample:destEnd] *= PicoScope.v_range / self.maxADC.value
        self.bufferCompleteB[self.nextSample:destEnd] *= PicoScope.v_range / self.maxADC.value
        if self.ps == ps3000a:
            self.bufferCompleteC[self.nextSample:destEnd] = self.bufferCMax[startIndex:sourceEnd]
            self.bufferCompleteD[self.nextSample:destEnd] = self.bufferDMax[startIndex:sourceEnd]
            self.bufferCompleteC[self.nextSample:destEnd] *= PicoScope.v_range / self.maxADC.value
            self.bufferCompleteD[self.nextSample:destEnd] *= PicoScope.v_range / self.maxADC.value

        self.nextSample += noOfSamples
        if autoStop:
            self.autoStopOuter = True

    def run(self):
        if self.ps == ps3000a:
            self.status["runStreaming"] = self.ps.ps3000aRunStreaming(
                self.chandle,
                ctypes.byref(PicoScope.sampleInterval),
                PicoScope.sampleUnits,
                0,
                PicoScope.totalSamples,
                1,
                1, 
                self.ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE'],
                PicoScope.sizeOfOneBuffer
            )
        else:
            self.status["runStreaming"] = self.ps.ps4000RunStreaming(
                self.chandle,
                ctypes.byref(PicoScope.sampleInterval),
                PicoScope.sampleUnits,
                0,
                PicoScope.totalSamples,
                1,
                1,
                PicoScope.sizeOfOneBuffer
            )
        assert_pico_ok(self.status["runStreaming"])
        print(self.ps.name + ": Capturing at", 1 / PicoScope.actualSampleInterval, "Hz for", PicoScope.totalSamples * PicoScope.actualSampleInterval, "s")
        self.t = time.time() - self.global_start_time + np.linspace(0, (PicoScope.totalSamples - 1) * PicoScope.actualSampleInterval, num=PicoScope.totalSamples)
        self.save_t = 0
        self.save_index = 0
        self.ready.set()

        self.bufferCompleteA = np.zeros(shape=PicoScope.totalSamples)
        self.bufferCompleteB = np.zeros(shape=PicoScope.totalSamples)
        if self.ps == ps3000a:
            self.bufferCompleteC = np.zeros(shape=PicoScope.totalSamples)
            self.bufferCompleteD = np.zeros(shape=PicoScope.totalSamples)

        self.nextSample = 0
        self.autoStopOuter = False
        self.wasCalledBack = False

        if self.ps == ps3000a:
            self.maxADC = ctypes.c_int16()
            self.status["maximumValue"] = self.ps.ps3000aMaximumValue(self.chandle, ctypes.byref(self.maxADC))
            assert_pico_ok(self.status["maximumValue"])
        else:
            self.maxADC = ctypes.c_int16(32767)

        self.cFuncPtr = self.ps.StreamingReadyType(self.streaming_callback)

        while self.nextSample < self.totalSamples and not self.autoStopOuter and not self.kill:
            self.wasCalledBack = False
            if self.ps == ps3000a:
                self.status["getStreamingLatestValues"] = self.ps.ps3000aGetStreamingLatestValues(self.chandle, self.cFuncPtr, None) 
                if self.wasCalledBack:
                    data = np.stack((self.t, self.bufferCompleteA, self.bufferCompleteB, self.bufferCompleteC, self.bufferCompleteD), axis=1)
                    self.signal.data.emit(data[max(0, self.nextSample - PicoScope.MAX_EMIT_LEN):self.nextSample])
                    if self.t[self.nextSample - 1] > self.save_t + SAVE_TIME:
                        np.save(
                            self.session_dir + "/tmp/ps3000a_" + str(self.save_t),
                            data[self.save_index:self.nextSample],
                            allow_pickle=False
                        )
                        self.save_t += SAVE_TIME
                        self.save_index = self.nextSample
                else:
                    time.sleep(0.01)
            else:
                self.status["getStreamingLatestValues"] = self.ps.ps4000GetStreamingLatestValues(self.chandle, self.cFuncPtr, None) 
                if self.wasCalledBack:
                    data = np.stack((self.t, self.bufferCompleteA, self.bufferCompleteB), axis=1)
                    self.signal.data.emit(data[max(0, self.nextSample - PicoScope.MAX_EMIT_LEN):self.nextSample])
                    if self.t[self.nextSample - 1] > self.save_t + SAVE_TIME:
                        np.save(
                            self.session_dir + "/tmp/ps4000_" + str(self.save_t),
                            data[self.save_index:self.nextSample],
                            allow_pickle=False
                        )
                        self.save_t += SAVE_TIME
                        self.save_index = self.nextSample
                else:
                    time.sleep(0.01)

        if self.ps == ps3000a:
            self.status["stop"] = self.ps.ps3000aStop(self.chandle)
            assert_pico_ok(self.status["stop"])
            self.status["close"] = self.ps.ps3000aCloseUnit(self.chandle)
            assert_pico_ok(self.status["close"])
            print(self.status)
            np.save(self.session_dir + "/" + self.ps.name, np.stack((self.t, self.bufferCompleteA, self.bufferCompleteB, self.bufferCompleteC, self.bufferCompleteD), axis=1), allow_pickle=False)
        else:
            self.status["stop"] = self.ps.ps4000Stop(self.chandle)
            assert_pico_ok(self.status["stop"])
            self.status["close"] = self.ps.ps4000CloseUnit(self.chandle)
            assert_pico_ok(self.status["close"])
            print(self.status)
            np.save(self.session_dir + "/" + self.ps.name, np.stack((self.t, self.bufferCompleteA, self.bufferCompleteB), axis=1), allow_pickle=False)


class SoundPlayer:
    BLOCKSIZE = 1024  # ~128 ms latency at 8 kHz; raise if playback glitches

    def __init__(self, path, device, volume_db):
        self.fs, arr = wav_read(path)
        arr = arr.astype(float)
        if arr.ndim > 1:
            arr = arr[:, 0]
        self.arr = arr / np.max(np.abs(arr))

        self.device = device
        self.volume = 10 ** (volume_db / 20)
        self.idx = 0
        self.stream = None

    def set_volume_db(self, volume_db):
        self.volume = 10 ** (volume_db / 20)

    def open(self):
        self.stream = sd.OutputStream(
            device=self.device, channels=1, samplerate=self.fs,
            callback=self.callback, blocksize=SoundPlayer.BLOCKSIZE
        )

    def play(self):
        self.stream.start()
        print("Playing sound on device", self.device, "at", self.fs, "Hz")

    def callback(self, outdata, frames, time_info, status):
        chunk = self.arr[self.idx:self.idx + frames] * self.volume
        outdata[:len(chunk), 0] = chunk
        if len(chunk) < frames:
            outdata[len(chunk):] = 0
            raise sd.CallbackStop
        self.idx += frames

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


class Signal(QObject):
    data = pyqtSignal(object)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.pvs = None
        self.ps4000 = None
        self.ps3000a = None

        self.sound_file = None
        self.sound_player = None
        self.pico_threads = []

        self.running = False
        self.run_id = 0

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)

        controls = QHBoxLayout()
        controls.setContentsMargins(8, 8, 8, 0)

        self.sound_file_button = QPushButton("Choose Sound File...")
        self.sound_file_button.clicked.connect(self.choose_sound_file)
        controls.addWidget(self.sound_file_button)

        self.sound_file_label = QLabel("No sound file selected")
        controls.addWidget(self.sound_file_label)

        controls.addStretch()

        controls.addWidget(QLabel("Playback device:"))
        self.device_combo = QComboBox()
        for i, d in enumerate(sd.query_devices()):
            if d['max_output_channels'] > 0:
                self.device_combo.addItem(f"{i}: {d['name']} ({d['max_output_channels']} out)", i)
        default_output = sd.default.device[1]
        default_index = self.device_combo.findData(default_output)
        if default_index != -1:
            self.device_combo.setCurrentIndex(default_index)
        controls.addWidget(self.device_combo)

        controls.addWidget(QLabel("Volume:"))
        self.volume_spinbox = QDoubleSpinBox()
        self.volume_spinbox.setRange(-80.0, 0.0)
        self.volume_spinbox.setDecimals(1)
        self.volume_spinbox.setSingleStep(0.5)
        self.volume_spinbox.setValue(-44.4)
        self.volume_spinbox.setSuffix(" dB")
        self.volume_spinbox.valueChanged.connect(self.change_volume)
        controls.addWidget(self.volume_spinbox)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(lambda: asyncio.ensure_future(self.start()))
        controls.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda: asyncio.ensure_future(self.stop()))
        controls.addWidget(self.stop_button)

        layout.addLayout(controls)

        self.graphWidget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphWidget)

        self.plots = {
            "PPG0": self.graphWidget.addPlot(row=0, col=0),
            "1A": self.graphWidget.addPlot(row=1, col=0),
            "1B": self.graphWidget.addPlot(row=2, col=0),
            "2A": self.graphWidget.addPlot(row=3, col=0),
            "2B": self.graphWidget.addPlot(row=4, col=0),
            "2C": self.graphWidget.addPlot(row=5, col=0),
            "2D": self.graphWidget.addPlot(row=6, col=0)
        }

        self.curves = {}
        for name, plot in self.plots.items():
            plot.setLabel("left", name)
            plot.setMouseEnabled(x=False, y=False)

            if name != "2A":
                plot.setXLink(self.plots["2A"])

            if name[0] == "1":
                self.curves[name] = plot.plot([], [], pen=(225, 109, 103))
            elif name[0] == "2":
                self.curves[name] = plot.plot([], [], pen=(62, 167, 160))
            else:
                self.curves[name] = plot.plot([], [], pen=(63, 169, 217))
            
            if name != "PPG0":
                self.curves[name].setDownsampling(ds=50, method='peak')
        
    def choose_sound_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose Sound File", "", "WAV files (*.wav);;All files (*)")
        if path:
            self.sound_file = path
            self.sound_file_label.setText(os.path.basename(path))

    async def start(self):
        self.start_button.setEnabled(False)
        self.sound_file_button.setEnabled(False)
        self.device_combo.setEnabled(False)
        self.running = True
        self.run_id += 1
        run_id = self.run_id

        i = 1
        while True:
            session_dir = f"session-{i:02d}"
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)
                os.makedirs(session_dir + "/tmp")
                break
            i += 1

        # open the output stream up front so playback can begin the instant
        # the scopes are streaming
        if self.sound_file is not None:
            self.sound_player = SoundPlayer(self.sound_file, self.device_combo.currentData(), self.volume_spinbox.value())
            self.sound_player.open()

        global_start_time = time.time()

        ps4000_signal = Signal()
        ps4000_signal.data.connect(self.update_ps4000_graph)
        # self.ps4000 = PicoScope(ps4000, ps4000_signal, global_start_time, session_dir)

        ps3000a_signal = Signal()
        ps3000a_signal.data.connect(self.update_ps3000a_graph)
        self.ps3000a = PicoScope(ps3000a, ps3000a_signal, global_start_time, session_dir)

        scopes = []
        for scope in (self.ps4000, self.ps3000a):
            if scope is not None:
                thread = threading.Thread(target=self.run_scope, args=(scope,))
                thread.start()
                scopes.append((scope, thread))
        self.pico_threads = [thread for _, thread in scopes]

        self.stop_button.setEnabled(True)

        # wait for every scope to finish opening (usually a few seconds); a
        # scope that fails to open is skipped so the others can run without it
        for scope, thread in scopes:
            while thread.is_alive() and not scope.opened.is_set():
                if not self.running or self.run_id != run_id:
                    return
                await asyncio.sleep(0.05)

        # release all open scopes at once so they start streaming together
        for scope, _ in scopes:
            if scope.opened.is_set():
                scope.go.set()

        # wait until they are actually streaming before starting playback
        for scope, thread in scopes:
            if not scope.opened.is_set():
                continue
            while thread.is_alive() and not scope.ready.is_set():
                if not self.running or self.run_id != run_id:
                    return
                await asyncio.sleep(0.05)
            if not scope.ready.is_set():
                print(scope.ps.name + " never started streaming")

        if not self.running or self.run_id != run_id:
            return

        if self.sound_player is not None:
            self.sound_player.play()

        device = await BleakScanner.find_device_by_name("Polar Sense DE957E2E", timeout=3)
        if device is None:
            print("Polar Sense DE957E2E not found")
        elif self.running and self.run_id == run_id:
            pvs_signal = Signal()
            pvs_signal.data.connect(self.update_pvs_graph)
            pvs = PolarVeritySense(device, pvs_signal, global_start_time, session_dir)
            await pvs.connect()
            print("Battery:", await pvs.get_battery_level())
            await pvs.start_ppg_stream()
            if self.running and self.run_id == run_id:
                self.pvs = pvs
            else:
                # stopped while the sensor was connecting; shut it back down
                await pvs.stop_ppg_stream()
                await pvs.disconnect()

    async def stop(self):
        self.running = False
        self.stop_button.setEnabled(False)

        if self.sound_player is not None:
            self.sound_player.close()
            self.sound_player = None

        if self.pvs is not None:
            await self.pvs.stop_ppg_stream()
            await self.pvs.disconnect()
            self.pvs = None

        if self.ps4000 is not None:
            self.ps4000.kill = True
        if self.ps3000a is not None:
            self.ps3000a.kill = True
        for t in self.pico_threads:
            await asyncio.get_event_loop().run_in_executor(None, t.join)
        self.pico_threads = []
        self.ps4000 = None
        self.ps3000a = None

        self.start_button.setEnabled(True)
        self.sound_file_button.setEnabled(True)
        self.device_combo.setEnabled(True)

    def change_volume(self, volume_db):
        if self.sound_player is not None:
            self.sound_player.set_volume_db(volume_db)

    def run_scope(self, scope):
        name = scope.ps.name
        try:
            scope.open()
        except Exception as e:
            print(name + " failed to open: ", e)
            return
        scope.opened.set()

        # rendezvous: hold here until every open scope is released together
        while not scope.go.is_set() and not scope.kill:
            time.sleep(0.01)
        if scope.kill:
            scope.close_unit()
            return

        try:
            scope.run()
        except Exception as e:
            print(name + " failed: ", e)
    
    @asyncClose
    async def closeEvent(self, event):
        await self.stop()
    
    def update_pvs_graph(self, data):
        self.curves["PPG0"].setData(data[:, 0], data[:, 1])

    def update_ps4000_graph(self, data):
        self.curves["1A"].setData(data[:, 0], data[:, 1])
        self.curves["1B"].setData(data[:, 0], data[:, 2])

    def update_ps3000a_graph(self, data):
        self.curves["2A"].setData(data[:, 0], data[:, 1])
        self.curves["2B"].setData(data[:, 0], data[:, 2])
        self.curves["2C"].setData(data[:, 0], data[:, 3])
        self.curves["2D"].setData(data[:, 0], data[:, 4])

if __name__ == "__main__":
    app = QApplication([])

    event_loop = QEventLoop(app)
    asyncio.set_event_loop(event_loop)

    app_close_event = asyncio.Event()
    app.aboutToQuit.connect(app_close_event.set)

    main_window = MainWindow()
    main_window.show()

    event_loop.run_until_complete(app_close_event.wait())

    event_loop.close()
