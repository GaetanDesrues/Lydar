from pyvistaqt import BackgroundPlotter


class LydarPlotter(BackgroundPlotter):
    def start(self, show=True):
        if show:
            self.app_window.show()
        self.app.exec_()
