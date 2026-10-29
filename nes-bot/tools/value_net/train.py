"""Fit the value net V(board) on gen.py trajectories and export it for sim.load_value_net.

Target (Monte Carlo): discounted future clear points after this board, a tetris = 1.0 (single 1/30, double 1/12,
triple 1/4), discount GAMMA per piece; a game that tops out ends with -DEATH.  The last TAIL boards of games that
stopped at the line limit are dropped (their future is cut off).  Input: the 200 cells plus 10 column heights / 20.
Net: 210-HID-HID-1 with ReLU, Adam 1e-3, batch 512; the epoch with the best validation loss (every 10th game) is kept.

Usage: python nes-bot/tools/value_net/train.py OUT.npz traj*.npz   (env: GAMMA=.98 DEATH=10 EPOCHS=8 HID=128)
value-net-v3.npz: HID=128, EPOCHS=8 on 7 files (900 games, 2.8M boards: 400 plain, 200 at EPS .02, 100 at EPS .05,
100 at EPS .05 with MAXH 5, 100 at EPS .15).
"""
import copy, os, sys

import numpy as np
import torch
import torch.nn as nn

GAMMA = float(os.environ.get('GAMMA', .98))
DEATH = float(os.environ.get('DEATH', 10))
EPOCHS = int(os.environ.get('EPOCHS', 8))
HID = int(os.environ.get('HID', 128))
TAIL = 300
REWARD = np.array([0, 40, 100, 300, 1200])/1200.


def inputs(bits):
    b = bits.reshape(-1, 20, 10).astype(np.float32)
    top = np.where(b.any(1), b.argmax(1), 20)
    return np.concatenate([b.reshape(-1, 200), (20-top).astype(np.float32)/20.], 1)


def load(files):
    X, Y, game = [], [], []
    for f in files:
        z = np.load(f)
        off = 0
        for length, dead in zip(z['lengths'], z['dead']):
            c = z['cleared'][off:off+length]
            bits = np.unpackbits(z['boards'][off:off+length], axis=1)[:, :200]
            off += length
            r = REWARD[c]
            g = np.zeros(length)
            acc = -DEATH if dead else 0.
            for t in range(length-1, -1, -1):
                g[t] = acc  # value of the board after placement t: the rewards from t+1 on
                acc = r[t]+GAMMA*acc
            keep = length if dead else max(0, length-TAIL)
            X.append(inputs(bits[:keep]))
            Y.append(g[:keep])
            game.append(np.full(keep, len(game)))
    return (torch.tensor(np.concatenate(X)), torch.tensor(np.concatenate(Y), dtype=torch.float32),
            np.concatenate(game))


if __name__ == '__main__':
    X, Y, game = load(sys.argv[2:])
    print('samples', len(Y), 'mean', round(Y.mean().item(), 3), 'std', round(Y.std().item(), 3), flush=True)
    val = torch.tensor(game % 10 == 0)
    train = torch.nonzero(~val).squeeze(1)
    net = nn.Sequential(nn.Linear(210, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    best, keep = float('inf'), None
    for epoch in range(EPOCHS):
        order = train[torch.randperm(len(train))]
        for i in range(0, len(order), 512):
            idx = order[i:i+512]
            loss = ((net(X[idx]).squeeze(1)-Y[idx])**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            mse = ((net(X[val]).squeeze(1)-Y[val])**2).mean().item()
        if mse < best:
            best, keep = mse, copy.deepcopy(net.state_dict())
        print(epoch, 'val mse', round(mse, 4), 'R2', round(1-mse/Y[val].var().item(), 3), flush=True)
    np.savez(sys.argv[1], **{name: keep[key].double().numpy() for name, key in
                             (('W1', '0.weight'), ('b1', '0.bias'), ('W2', '2.weight'), ('b2', '2.bias'),
                              ('W3', '4.weight'), ('b3', '4.bias'))})
