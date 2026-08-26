import matplotlib.pyplot as plt
from collections import deque


class LivePlotter:
    def __init__(self, keys=None, maxlen: int = 50):
        if keys is None:
            keys = [""]
        self._buffers = [deque(maxlen=maxlen) for _ in keys]
        self._x_buffer = deque(maxlen=maxlen)
        self._x = 0

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.lines = []
        for key in keys:
            (line,) = self.ax.plot([], [], marker=".", label=key)
            self.lines.append(line)
        self.ax.legend()

    def plot(self, data, x=None):
        assert len(data) == len(self._buffers)

        self._x = x if x is not None else (self._x + 1)
        self._x_buffer.append(self._x)

        for buffer, data, line in zip(self._buffers, data, self.lines):
            buffer.append(data)
            line.set_xdata(self._x_buffer)
            line.set_ydata(buffer)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
