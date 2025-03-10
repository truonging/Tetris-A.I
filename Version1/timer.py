import pygame.time

class Timer:
    def __init__(self, duration, repeated=False,func=None):
        self.duration = duration
        self.repeated = repeated
        self.func = func

        self.start_time = 0
        self.active = False

    def activate(self):
        self.active = True
        self.start_time = pygame.time.get_ticks()

    def deactivate(self):
        self.active = False
        self.start_time = 0

    def update(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.start_time >= self.duration and self.active:
            if self.func and self.start_time!=0:
                self.func()
            self.deactivate()
            if self.repeated:
                self.activate()