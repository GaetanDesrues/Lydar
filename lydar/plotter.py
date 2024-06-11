from pyvistaqt import BackgroundPlotter


class LydarPlotter(BackgroundPlotter):
    def start(self, show=True):
        
        if show:
            self.add_key_event("c", lambda: print(self.camera_position))
            # self.render()
            # self.app.processEvents()
            self.app_window.show()
        else:
            self.app_window.show()
            
        self.app.exec()
