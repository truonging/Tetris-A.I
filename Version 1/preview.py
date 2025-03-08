from settings import *
from pygame.image import load
from os import path

class Preview:
    def __init__(self):
        self.surface = pygame.Surface((SIDEBAR_WIDTH,GAME_HEIGHT*PREVIEW_HEIGHT_FRACTION))
        self.rect = self.surface.get_rect(topright=(WINDOW_WIDTH-PADDING,PADDING))
        self.display_surface = pygame.display.get_surface()

        self.shape_surfaces = {
            shape: load(path.join('assets', 'graphics', f'{shape}.png')).convert_alpha() for shape in TETROMINOS.keys()
        }
        self.preview_height = self.surface.get_height() / 2

    def _shrink_image(self, image):
        width, height = image.get_size()
        new_size = (int(width * SCALED_SIZE), int(height * SCALED_SIZE))
        return pygame.transform.smoothscale(image, new_size)

    def display_previews(self, shapes):
        preview_height = self.preview_height
        for i in range(len(shapes)):
            preview = self._shrink_image(self.shape_surfaces[shapes[i]])
            rect = preview.get_rect(center=(self.surface.get_width()/2,i*preview_height+preview_height/2))
            self.surface.blit(preview,rect)

    def run(self,game):
        self.surface.fill(GRAY)
        self.display_previews(game.previews)
        self.display_surface.blit(self.surface, self.rect)
        pygame.draw.rect(self.display_surface, LINE_COLOR,self.rect,2,2)