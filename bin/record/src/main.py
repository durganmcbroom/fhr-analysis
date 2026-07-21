import asyncio
from threading import Thread

import numpy as np
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
    
    def __init__(self, ps, signal, session_dir):
        self.ps = ps
        self.signal = signal
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
        # Every scope shares this identical 0-based time grid. Because they are
        # all released from the rendezvous together, giving them the same grid
        # makes the exported recordings start at 0 and line up sample-for-sample.
        self.t = np.linspace(0, (PicoScope.totalSamples - 1) * PicoScope.actualSampleInterval, num=PicoScope.totalSamples)
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
            "1A": self.graphWidget.addPlot(row=0, col=0),
            "1B": self.graphWidget.addPlot(row=1, col=0),
            "2A": self.graphWidget.addPlot(row=2, col=0),
            "2B": self.graphWidget.addPlot(row=3, col=0),
            "2C": self.graphWidget.addPlot(row=4, col=0),
            "2D": self.graphWidget.addPlot(row=5, col=0)
        }

        self.curves = {}
        for name, plot in self.plots.items():
            plot.setLabel("left", name)
            plot.setMouseEnabled(x=False, y=False)

            if name != "2A":
                plot.setXLink(self.plots["2A"])

            if name[0] == "1":
                self.curves[name] = plot.plot([], [], pen=(225, 109, 103))
            else:
                self.curves[name] = plot.plot([], [], pen=(62, 167, 160))

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

        ps4000_signal = Signal()
        ps4000_signal.data.connect(self.update_ps4000_graph)
        # self.ps4000 = PicoScope(ps4000, ps4000_signal, session_dir)

        ps3000a_signal = Signal()
        ps3000a_signal.data.connect(self.update_ps3000a_graph)
        self.ps3000a = PicoScope(ps3000a, ps3000a_signal, session_dir)

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

    async def stop(self):
        self.running = False
        self.stop_button.setEnabled(False)

        if self.sound_player is not None:
            self.sound_player.close()
            self.sound_player = None

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
