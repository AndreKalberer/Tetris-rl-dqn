"""Watch or play Offline Sprint; optional headless MP4 recording."""
import argparse
import os
from pathlib import Path

from dqn.train_candidates import load_policy
from offline.engine import Game,Rules,SPAWN,cells,collides
from offline.bot import Bot
from offline.evaluate import checkpoint_default,episode,replay

COLORS=[(14,20,29),(47,205,220),(238,211,73),(170,99,218),(80,193,128),
        (224,95,112),(79,132,222),(235,161,72)]


class View:
    def __init__(self,pygame):
        self.pg=pygame
        self.screen=pygame.display.set_mode((760,680))
        pygame.display.set_caption('Offline Sprint - local Tetris bot')
        self.font=pygame.font.SysFont('Segoe UI',22)
        self.small=pygame.font.SysFont('Segoe UI',16)
        self.large=pygame.font.SysFont('Segoe UI',32,bold=True)

    def text(self,value,x,y,large=False,muted=False):
        font=self.large if large else self.small if muted else self.font
        self.screen.blit(font.render(str(value),True,(143,157,176) if muted else (232,237,244)),(x,y))

    def miniature(self,piece,x,y):
        if piece<0:
            return
        for dx,dy in cells(piece,0):
            self.pg.draw.rect(self.screen,COLORS[piece+1],(x+dx*17,y+dy*17,15,15))

    def draw(self,game,mode='Bot',paused=False):
        from dataclasses import replace
        pg=self.pg
        self.screen.fill((18,25,35))
        self.text('OFFLINE SPRINT' if game.rules.target_lines is not None else 'OFFLINE ENDURANCE',36,15,large=True)
        self.text('Timed controls / gravity enabled',365,29,muted=True)
        left,top,size=40,70,27
        pg.draw.rect(self.screen,(9,14,22),(left,top,10*size,20*size))
        for y,row in enumerate(game.visible):
            for x,value in enumerate(row):
                rect=(left+x*size+1,top+y*size+1,size-2,size-2)
                pg.draw.rect(self.screen,COLORS[int(value)] if value else (23,32,44),rect,
                             0 if value else 1)
        if not game.over and not game.won:
            ghost=game.pose
            while not collides(game.board,game.piece,replace(ghost,y=ghost.y+1)):
                ghost=replace(ghost,y=ghost.y+1)
            for dx,dy in cells(game.piece,ghost.rotation):
                y=ghost.y+dy-game.rules.hidden_rows
                if y>=0:
                    pg.draw.rect(self.screen,(89,108,130),
                        (left+(ghost.x+dx)*size+2,top+y*size+2,size-4,size-4),1)
            for dx,dy in cells(game.piece,game.pose.rotation):
                y=game.pose.y+dy-game.rules.hidden_rows
                if y>=0:
                    pg.draw.rect(self.screen,COLORS[game.piece+1],
                        (left+(game.pose.x+dx)*size+1,top+y*size+1,size-2,size-2))
        self.text(f'{game.lines} / {game.rules.target_lines} lines' if game.rules.target_lines is not None
                  else f'{game.lines:,} lines',365,79,large=True)
        self.text(f'{game.frames/game.rules.fps:.2f} seconds',365,127)
        self.text(f'{game.placements} pieces placed',365,161)
        self.text('40 lines cleared' if game.won else 'Topped out' if game.over else 'Paused' if paused else mode,365,202)
        self.text('HOLD',365,253,muted=True)
        self.miniature(game.held,365,280)
        self.text('NEXT',500,253,muted=True)
        for i,piece in enumerate(game.queue):
            self.miniature(piece,500,280+i*48)
        self.text(f'Seed {game.seed} | Offline Sprint v1',40,625,muted=True)
        self.text('B Bot/Human   P Pause   R Restart   N New seed',40,651,muted=True)
        self.text('Arrows Move / Down Soft drop',365,568,muted=True)
        self.text('X/Up Rotate CW   Z Rotate CCW',365,593,muted=True)
        self.text('Space Hard drop   C Hold   Esc Quit',365,618,muted=True)
        pg.display.flip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=None)
    parser.add_argument('--seed',type=int,default=1001)
    parser.add_argument('--beam',type=int,default=4)
    parser.add_argument('--human',action='store_true')
    parser.add_argument('--record',type=Path,help='Headless fixed-clock MP4 recording')
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args()
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
    os.environ.setdefault('SDL_VIDEO_CENTERED','1')
    if args.record:
        os.environ['SDL_VIDEODRIVER']='dummy'
        os.environ['SDL_AUDIODRIVER']='dummy'
    import pygame
    pygame.init()
    view=View(pygame)
    config,model=load_policy(args.checkpoint or checkpoint_default())
    rules=Rules()
    if args.record:
        import cv2
        args.record.parent.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(args.record),cv2.VideoWriter_fourcc(*'mp4v'),20,(760,680))
        if not writer.isOpened():
            raise RuntimeError('Could not open MP4 writer')
        def frame(game):
            if game.frames%3==0 or game.won or game.over:
                view.draw(game)
                rgb=pygame.surfarray.array3d(view.screen).transpose(1,0,2)
                writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                if args.screenshot and game.won:
                    pygame.image.save(view.screen,str(args.screenshot))
        row=episode(config,model,args.seed,rules,args.beam,frame_callback=frame)
        assert replay(row,rules)
        for _ in range(40):
            writer.write(cv2.cvtColor(pygame.surfarray.array3d(view.screen).transpose(1,0,2),cv2.COLOR_RGB2BGR))
        writer.release()
        print({k:v for k,v in row.items() if k not in ['inputs','locks']})
        pygame.quit()
        return
    clock=pygame.time.Clock()
    pygame.key.set_repeat(160,50)
    game=Game(args.seed,rules)
    bot=Bot(config,model,args.beam)
    bot_mode=not args.human
    paused=False
    accumulator=0.0
    from collections import deque
    commands=deque()
    mapping={pygame.K_LEFT:'left',pygame.K_RIGHT:'right',pygame.K_DOWN:'soft',pygame.K_UP:'cw',
             pygame.K_x:'cw',pygame.K_z:'ccw',pygame.K_SPACE:'hard',pygame.K_c:'hold'}
    running=True
    while running:
        dt=clock.tick(60)/1000
        if dt>1:
            paused=True  # Suspend/resume must not fast-forward an unseen game.
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
            elif event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE: running=False
                elif event.key==pygame.K_p: paused=not paused
                elif event.key in [pygame.K_r,pygame.K_n]:
                    if event.key==pygame.K_n: args.seed+=1
                    game=Game(args.seed,rules)
                    bot=Bot(config,model,args.beam)
                    commands.clear()
                elif event.key==pygame.K_b:
                    bot_mode=not bot_mode
                    bot=Bot(config,model,args.beam)
                    commands.clear()
                elif not bot_mode and not paused and event.key in mapping:
                    commands.append(mapping[event.key])
        if paused or game.won or game.over:
            accumulator=0
        else:
            accumulator+=dt
            while accumulator>=1/rules.fps and not game.over and not game.won:
                game.tick(bot.command(game) if bot_mode else commands.popleft() if commands else 'none')
                accumulator-=1/rules.fps
        view.draw(game,'Bot playing' if bot_mode else 'Human controls',paused)
    pygame.quit()


if __name__=='__main__':
    main()
