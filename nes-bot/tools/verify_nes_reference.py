"""Optional source-level differential check; no ROM or ROM binary is required.

Install py65==1.2.0 and provide the pinned public main.asm/constants.asm files.
Selected routines are assembled in memory only, and executed by a 6502 CPU.
This is not a whole-ROM/PPU/audio or emulator-parity test.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from py65.assembler import Assembler
from py65.devices.mpu6502 import MPU

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nes.engine import (CELLS,GRAVITY,ROTATE_CCW,ROTATE_CW,SPAWN,Control,
                        Randomizer,collides,control_frame)

REVISION='94581c0306fa0009fe11fde48222875549d15b8b'
HASHES={'main.asm':'7dde9fb41caf2d93f6bd8d951eae5cd8d2c2ab916c926751ef4fba21ca26719f',
        'constants.asm':'c6376d030ddf57206588f530bf7c939d8289e8e100501b431d374ab1e7fdb143'}


def expression(text,symbols):
    text=re.sub(r'\$([0-9A-Fa-f]+)',r'0x\1',text)
    # Input files are pinned by SHA-256 before any expressions are evaluated.
    return int(eval(text,{'__builtins__':{}},symbols))


def preprocess(text,symbols):
    result=[]
    active=True
    stack=[]
    for raw in text.splitlines():
        line=raw.split(';')[0].strip()
        if line.startswith('.ifndef '):
            condition=line.split()[1] not in symbols
            stack.append((active,condition))
            active=active and condition
        elif line.startswith('.if '):
            condition=line[4:].replace('<>','!=')
            condition=re.sub(r'(?<![<>!=])=(?!=)','==',condition)
            condition=bool(expression(condition,symbols)) if active else False
            stack.append((active,condition))
            active=active and condition
        elif line=='.else':
            parent,condition=stack[-1]
            active=parent and not condition
        elif line=='.endif':
            active=stack.pop()[0]
        elif active and line:
            assignment=re.match(r'^(\w+)\s*:?=\s*(.*)$',line)
            if assignment:
                try:
                    symbols[assignment[1]]=expression(assignment[2],symbols)
                except (NameError,SyntaxError):
                    pass
            else:
                result.append(line)
    return result


def block(lines,start,end):
    a=next(i for i,line in enumerate(lines) if line==start+':')
    b=next(i for i in range(a+1,len(lines)) if lines[i]==end+':')
    return lines[a:b]


class Oracle:
    def __init__(self,reference):
        self.symbols={'PAL':0,'NWC':0}
        texts={}
        for name,digest in HASHES.items():
            data=(reference/name).read_bytes()
            assert hashlib.sha256(data).hexdigest()==digest,f'Wrong pinned reference: {name}'
            texts[name]=data.decode('utf-8')
        preprocess(texts['constants.asm'],self.symbols)
        lines=preprocess(texts['main.asm'],self.symbols)
        selected=[]
        for start,end in (
            ('rotate_tetrimino','drop_tetrimino'),
            ('drop_tetrimino','unreferenced_framesPerDropTable'),
            ('shift_tetrimino','stageSpriteForCurrentPiece'),
            ('isPositionValid','render_mode_play_and_demo'),
            ('pickRandomTetrimino','tetriminoTypeFromOrientation'),
            ('spawnTable','spawnOrientationFromOrientation'),
            ('generateNextPseudorandomNumber','copyOamStagingToOam'),
            ('updatePlayfield','gameModeState_handleGameOver')):
            selected.extend(block(lines,start,end))
        a=lines.index('orientationTable:')
        selected.append(lines[a])
        for line in lines[a+1:]:
            if not line.startswith('.byte'): break
            selected.append(line)
        self.cpu=MPU(memory=[0]*65536)
        assembler=Assembler(self.cpu)
        statements=[]
        scope=''
        for line in selected:
            label=re.match(r'^([\w@]+):\s*(.*)$',line)
            if label:
                name=label[1]
                if name.startswith('@'):
                    name=scope+'__'+name[1:]
                else:
                    scope=name
                statements.append(('label',name))
                line=label[2]
            if line:
                line=re.sub(r'@(\w+)',lambda m:scope+'__'+m[1],line)
                statements.append(('code',line))
        def encoded(line,pc,first=False):
            if line.startswith('.byte'):
                return [expression(x.strip(),self.symbols)&255 for x in line[5:].split(',')]
            fields=line.split(None,1)
            opcode=fields[0]
            operand=fields[1] if len(fields)>1 else ''
            def substitute(match):
                name=match[0]
                if name.lower() in ('a','x','y'):
                    return name
                value=self.symbols.get(name,0x8000 if first else None)
                if value is None: raise ValueError('Unknown reference symbol: '+name)
                return f'${value:04X}'
            operand=re.sub(r'(?<![$\w])[A-Za-z_]\w*',substitute,operand)
            operand=re.sub(r'\$[0-9A-F]+(?:[+|\-](?:\$[0-9A-F]+|[0-9]+))+',
                           lambda m:f'${expression(m[0],{}):04X}',operand)
            return assembler.assemble(opcode+' '+operand,pc)
        for first in (True,False):
            pc=0x6000
            for kind,line in statements:
                if kind=='label':
                    if not first: assert self.symbols[line]==pc
                    self.symbols[line]=pc
                else:
                    data=encoded(line,pc,first)
                    if not first: self.cpu.memory[pc:pc+len(data)]=data
                    pc+=len(data)

    def reset(self):
        self.cpu.memory[:0x800]=[0]*0x800
        self.cpu.memory[0x400:0x500]=[0xEF]*256
        self.cpu.p=0x20  # Decimal mode disabled, as on the NES 2A03 CPU.

    def run(self,label):
        self.cpu.sp=0xFD
        self.cpu.memory[0x1FE:0x200]=[0xFE,0x7F]
        self.cpu.pc=self.symbols[label]
        for _ in range(10000):
            if self.cpu.pc==0x7FFF: return
            self.cpu.step()
        raise AssertionError('Reference subroutine did not return: '+label)

    def controls(self,board,state,buttons,level):
        self.reset()
        m=self.cpu.memory
        m[0x400:0x4C8]=np.where(board.reshape(-1)!=0,0x7B,0xEF).tolist()
        m[0xB8:0xBA]=[0,4]
        for address,value in ((0x40,state.x),(0x41,state.y),(0x42,state.orientation),
            (0x44,level),(0x45,(state.fall+1)&255),(0x46,state.das),(0x48,1),(0x49,32),
            (0x4E,state.soft),(0x4F,state.soft_points),(0xB5,buttons & ~state.previous),(0xB6,buttons)):
            m[address]=value
        for routine in ('shift_tetrimino','rotate_tetrimino','drop_tetrimino'):
            self.run(routine)
        return Control(m[0x40],m[0x41],m[0x42],m[0x45],m[0x46],m[0x4E],m[0x4F],buttons),m[0x48]==2

    def randomizer(self,state,count,last):
        self.reset()
        m=self.cpu.memory
        m[0x17:0x19]=[state>>8,state&255]
        m[0x19],m[0x1A],m[0xC0]=last,count,4
        self.run('pickRandomTetrimino')
        return self.cpu.a,(m[0x17]<<8)|m[0x18],m[0x1A],m[0x19]


def verify(reference,cases,output):
    oracle=Oracle(reference)
    m,symbols=oracle.cpu.memory,oracle.symbols
    def signed(value): return value if value<128 else value-256
    for orientation,cells in enumerate(CELLS):
        start=symbols['orientationTable']+orientation*12
        actual={(signed(m[start+i+2]),signed(m[start+i])) for i in (0,3,6,9)}
        assert actual==set(cells),(orientation,actual,cells)
    start=symbols['rotationTable']
    assert tuple(m[start:start+38:2])==ROTATE_CCW
    assert tuple(m[start+1:start+38:2])==ROTATE_CW
    assert tuple(m[symbols['spawnTable']:symbols['spawnTable']+7])==SPAWN
    assert tuple(m[symbols['framesPerDropTable']:symbols['framesPerDropTable']+30])==GRAVITY
    rng=np.random.default_rng(73021)
    for index in range(cases):
        state,count,last=int(rng.integers(65536)),int(rng.integers(256)),int(rng.choice((0,)+SPAWN))
        randomizer=Randomizer()
        randomizer.state,randomizer.count,randomizer.last=state,count,last
        actual=randomizer.draw()
        assert (actual,randomizer.state,randomizer.count,randomizer.last)==oracle.randomizer(state,count,last)
        while True:
            board=np.zeros((20,10),dtype=np.uint8)
            top=int(rng.integers(4,20))
            board[top:]=rng.random((20-top,10))<0.35
            state=Control(x=int(rng.integers(10)),y=int(rng.integers(19)),orientation=int(rng.integers(19)),
                fall=int(rng.integers(48)),das=int(rng.choice((0,10,15,16))),
                soft=int(rng.choice((0,1,2,0xA0,255))),soft_points=int(rng.integers(30)),
                previous=int(rng.choice((0,1,2,4,64,128,129,130,192))))
            if not collides(board,state.x,state.y,state.orientation): break
        buttons=int(rng.choice((0,1,2,3,4,5,6,64,65,66,68,128,129,130,132,192)))
        level=int(rng.choice((0,9,18,19,29,40)))
        actual=control_frame(board,state,buttons,level)
        expected=oracle.controls(board,state,buttons,level)
        assert actual==expected,dict(case=index,state=asdict(state),buttons=buttons,level=level,
                                     actual=(asdict(actual[0]),actual[1]),expected=(asdict(expected[0]),expected[1]))
    result=dict(reference_revision=REVISION,reference_sha256=HASHES,cpu='py65 1.2.0 NMOS 6502 (decimal flag clear)',
        orientation_tables_verified=19,rotation_spawn_gravity_tables_verified=True,
        randomizer_cases=cases,active_control_cases=cases,case_seed=73021,
        engine_sha256=hashlib.sha256(Path('nes/engine.py').read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Source-level routine comparisons only. Whole-ROM, frame scheduler, PPU/audio timing, top-boundary memory and late-game glitches remain unvalidated.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--cases',type=int,default=3000)
    parser.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args()
    if arguments.cases<1: parser.error('Cases must be positive')
    verify(arguments.reference_dir,arguments.cases,arguments.output)
