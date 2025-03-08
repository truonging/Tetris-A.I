from settings import *
from os.path import join

class AI_board:
    def __init__(self):
        self.surface = pygame.Surface((SIDEBAR_WIDTH,GAME_HEIGHT*AI_HEIGHT_FRACTION-PADDING))
        self.rect = self.surface.get_rect(bottomright=(WINDOW_WIDTH-PADDING,WINDOW_HEIGHT-PADDING))
        self.display_surface = pygame.display.get_surface()

        self.increment_height = self.surface.get_height() / 7

        self.font = pygame.font.Font(join('assets', 'graphics', 'Russo_One.ttf'), int(18 * SCALED_SIZE))

        self.total_heights = 0
        self.bumpiness = 0
        self.holes = 0
        self.y_pos = 0
        self.pillar = 0
        self.reward = 0
        self.ai_confidence = 0
        self.random = 'FALSE'

    def display_text(self, pos, text):
        text_surface = self.font.render(text, True, 'white')
        text_rext = text_surface.get_rect(center=pos)
        self.surface.blit(text_surface, text_rext)

    def update_from(self,reward,confidence,state, random):
        self.total_heights = int(state[0])
        self.bumpiness = state[1]
        self.holes = int(state[3])
        self.y_pos = 20 - int(state[4])
        self.random = 'TRUE' if random else 'FALSE'
        self.reward = int(reward)
        self.ai_confidence = str(confidence)[:10]

    def reset(self):
        self.total_heights = 0
        self.bumpiness = 0
        self.holes = 0
        self.y_pos = 0
        self.pillar = 0
        self.reward = 0
        self.ai_confidence = 0
        self.random = 'FALSE'

    def run(self, game, reward, confidence, state, random):
        self.surface.fill(GRAY)
        self.update_from(reward, confidence, state, random)
        A = [('total heights',self.total_heights),('bumpiness',self.bumpiness),
             ('holes',self.holes),('y pos',self.y_pos),('random',self.random),
             ('reward',self.reward),('confidence',self.ai_confidence)]
        for i in range(len(A)):
            text = A[i]
            x = self.surface.get_width() / 2
            y = self.increment_height / 2 + i * self.increment_height
            self.display_text((x,y),f'{text[0]}: {text[1]}')
        self.display_surface.blit(self.surface, self.rect)
        pygame.draw.rect(self.display_surface, LINE_COLOR, self.rect, 2, 2)