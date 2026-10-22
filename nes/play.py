"""Play the provisional NES engine or watch its untrained stacking probe."""
import argparse
import json
import os
from pathlib import Path

from nes.controller import StackingProbe
from nes.engine import Buttons,CELLS,FPS,Game,NAMES,Phase,Rules,SPAWN,TYPE,gravity_frames


COLORS=((13,20,32),(156,103,225),(83,142,229),(228,100,114),(237,210,84),
        (86,191,137),(235,162,80),(69,207,220))


class View:
    def __init__(self,pygame):
        self.pg=pygame
        self.screen=pygame.display.set_mode((800,700))
        pygame.display.set_caption('NES Tetris research engine - provisional')
        self.font=pygame.font.SysFont('Segoe UI',22)
        self.small=pygame.font.SysFont('Segoe UI',16)
        self.large=pygame.font.SysFont('Segoe UI',30,bold=True)

    def text(self,value,x,y,large=False,muted=False):
        font=self.large if large else self.small if muted else self.font
        self.screen.blit(font.render(str(value),True,(145,160,180) if muted else (235,240,248)),(x,y))

    def draw(self,game,bot=False,paused=False):
        pg=self.pg
        self.screen.fill((18,25,36))
        self.text('NES / NTSC A-TYPE',34,19,large=True)
        self.text('PROVISIONAL ENGINE',486,30,muted=True)
        left,top,size=38,83,27
        pg.draw.rect(self.screen,(9,14,23),(left,top,270,540))
        for y,row in enumerate(game.board):
            for x,piece in enumerate(row):
                pg.draw.rect(self.screen,COLORS[int(piece)] if piece else (31,42,58),
                    (left+x*size+1,top+y*size+1,size-2,size-2),0 if piece else 1)
        if game.phase==Phase.ACTIVE:
            c=game.control
            for dx,dy in CELLS[c.orientation]:
                if c.y+dy>=0:
                    pg.draw.rect(self.screen,COLORS[TYPE[c.orientation]+1],
                        (left+(c.x+dx)*size+1,top+(c.y+dy)*size+1,size-2,size-2))
        self.text(f'{game.score:06d} POINTS',350,83,large=True)
        self.text(f'{game.lines:,} lines',350,133)
        self.text(f'Level {game.level} / start {game.rules.start_level}',350,173)
        self.text(f'{FPS/gravity_frames(game.level):.2f} rows/sec',350,213)
        self.text(f'{game.placements:,} pieces placed',350,253)
        self.text(f'{game.frames/FPS:.2f} simulated seconds',350,293)
        self.text('NEXT',350,345,muted=True)
        for dx,dy in CELLS[game.next_orientation]:
            pg.draw.rect(self.screen,COLORS[TYPE[game.next_orientation]+1],
                (390+dx*22,382+dy*22,20,20))
        label=('GAME OVER' if game.over else 'PAUSED' if paused else
               'Untrained stacking probe' if bot else 'Human controls')
        self.text(label,350,454)
        self.text(f'Phase: {game.phase.name.lower()}',350,491,muted=True)
        self.text('No hold / no hard drop / one-piece preview',350,529,muted=True)
        self.text('ROM parity and NES RL training are pending',350,556,muted=True)
        self.text('Arrows: move/down    X: A (CW)    Z: B (CCW)',34,644,muted=True)
        self.text('B: probe/human    P: pause    R: restart    N: next seed    Esc: quit',34,670,muted=True)
        pg.display.flip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--start-level',type=int,default=0)
    parser.add_argument('--bot',action='store_true',help='Untrained stacking probe, not the modern policy')
    parser.add_argument('--record',type=Path,help='Headless development-probe recording')
    parser.add_argument('--screenshot',type=Path)
    parser.add_argument('--record-seconds',type=float,default=30)
    args=parser.parse_args()
    rules=Rules(start_level=args.start_level)
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
    if args.record:
        if args.record_seconds<=0:
            parser.error('Recording duration must be positive')
        os.environ['SDL_VIDEODRIVER']='dummy'
        os.environ['SDL_AUDIODRIVER']='dummy'
    import pygame
    pygame.init()
    view=View(pygame)
    game=Game(args.seed,rules)
    if args.record:
        import cv2
        from nes.evaluate import episode
        args.record.parent.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(args.record),cv2.VideoWriter_fourcc(*'mp4v'),FPS/3,(800,700))
        if not writer.isOpened():
            raise RuntimeError('Could not create recording')
        def record_frame(current):
            if current.frames%3==0 or current.over:
                view.draw(current,True)
                rgb=pygame.surfarray.array3d(view.screen).transpose(1,0,2)
                writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        try:
            row,game=episode(args.seed,args.start_level,int(args.record_seconds*FPS),60,record_frame)
            view.draw(game,True)
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True,exist_ok=True)
                pygame.image.save(view.screen,str(args.screenshot))
            args.record.with_suffix('.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in row.items() if k not in ('inputs','locks')}))
        finally:
            writer.release()
            pygame.quit()
        return
    clock=pygame.time.Clock()
    accumulator=0.0
    running,paused,bot_mode=True,False,args.bot
    bot=StackingProbe()
    while running:
        dt=clock.tick(120)/1000
        if dt>1:
            paused=True
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
            elif event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE: running=False
                elif event.key==pygame.K_p: paused=not paused
                elif event.key==pygame.K_b:
                    bot_mode=not bot_mode
                    bot=StackingProbe()
                elif event.key in (pygame.K_r,pygame.K_n):
                    if event.key==pygame.K_n: args.seed+=1
                    game=Game(args.seed,rules)
                    bot=StackingProbe()
                    accumulator=0.0
        keys=pygame.key.get_pressed()
        buttons=(int(Buttons.LEFT) if keys[pygame.K_LEFT] else 0)|(
            int(Buttons.RIGHT) if keys[pygame.K_RIGHT] else 0)|(
            int(Buttons.DOWN) if keys[pygame.K_DOWN] else 0)|(
            int(Buttons.A) if keys[pygame.K_x] else 0)|(
            int(Buttons.B) if keys[pygame.K_z] else 0)
        if buttons&3==3:
            buttons&=~3
        if paused or game.over:
            accumulator=0.0
        else:
            accumulator+=dt
            while accumulator>=1/FPS and not game.over:
                game.tick(bot.command(game.observe()) if bot_mode else buttons)
                accumulator-=1/FPS
        view.draw(game,bot_mode,paused)
    pygame.quit()


if __name__=='__main__':
    main()
