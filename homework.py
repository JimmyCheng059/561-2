import numpy as np
import string
import time
import sys
import math
from Queue import Queue
import heapq as hq


class State:
    def __init__(self, bd, seglist, klist, mins, maxs, nextb):
        self.board = bd
        self.segList = seglist
        self.okList = klist
        self.minscore = mins
        self.maxscore = maxs
        self.nextbig = nextb


class Segment:
    def __init__(self, siz, list):
        self.size = siz
        self.list = list


def parser(filename):
    files = open(filename, 'r').readlines()
    index = 0
    klist = list()
    for line in files:
        if index == 0:
            size = int(line)
            cell = size * size
            board = np.arange(cell).reshape(size, size)
        elif index == 1:
            typnum = int(line)
        elif index == 2:
            time = float(line)
        else:
            num = 0
            for slot in line.strip():
                if slot == "*":
                    slot = 10
                else:
                    klist.append((index - 3) * size + num)
                board[index - 3][num] = slot
                num += 1
        index += 1
    return dict(size=size, typnum=typnum, time=time, board=board, list=klist)


def segmentation(board, okList, n):
    frontier = Queue()
    seglist = list()
    allocated = list()
    nextbig = 0
    for spot in okList:
        if spot not in allocated:
            allocated.append(spot)
            frontier.put_nowait(spot)
            seg = list()
            seg.append(spot)
            x = spot / n
            y = spot % n
            target = board[x][y]
            while not frontier.empty():
                cur = frontier.get_nowait()
                cx = cur / n
                cy = cur % n
                if cx + 1 <= n - 1:
                    if board[cx + 1][cy] == target and cur + n not in seg:
                        seg.append(cur + n)
                        allocated.append(cur + n)
                        frontier.put_nowait(cur + n)
                if cx - 1 >= 0:
                    if board[cx - 1][cy] == target and cur - n not in seg:
                        seg.append(cur - n)
                        allocated.append(cur - n)
                        frontier.put_nowait(cur - n)
                if cy + 1 <= n - 1:
                    if board[cx][cy + 1] == target and cur + 1 not in seg:
                        seg.append(cur + 1)
                        allocated.append(cur + 1)
                        frontier.put_nowait(cur + 1)
                if cy - 1 >= 0:
                    if board[cx][cy - 1] == target and cur - 1 not in seg:
                        seg.append(cur - 1)
                        allocated.append(cur - 1)
                        frontier.put_nowait(cur - 1)
            newseg = Segment(len(seg), seg)
            nextbig = max(newseg.size, nextbig)
            seglist.append(newseg)
    return seglist, nextbig * nextbig


def select(state, pick, minimax, bdsize):
    tmpboard = np.copy(state.board)
    newboard = np.copy(state.board)
    newlist = state.okList[:]
    affected = list()
    for fruit in state.segList[pick].list:
        x = fruit / bdsize
        y = fruit % bdsize
        tmpboard[x][y] = 10
        if y not in affected:
            affected.append(y)
    for column in affected:
        fruit_pivot = bdsize - 1
        spece_pivot = 0
        for row in range(bdsize - 1, -1, -1):
            if tmpboard[row][column] != 10:
                newboard[fruit_pivot][column] = tmpboard[row][column]
                fruit_pivot -= 1
            elif tmpboard[row][column] == 10:
                newboard[spece_pivot][column] = tmpboard[row][column]
                space = spece_pivot * bdsize + column
                if space in newlist:
                    newlist.remove(spece_pivot * bdsize + column)
                spece_pivot += 1
    newseglist = segmentation(newboard, newlist, bdsize)
    minv = 0
    maxv = 0
    if minimax:
        maxv = state.segList[pick].size * state.segList[pick].size
    else:
        minv = state.segList[pick].size * state.segList[pick].size
    newState = State(newboard, newseglist[0], newlist, state.minscore + minv, state.maxscore + maxv, newseglist[1])
    if len(newState.segList) > 0:
        nextB = newState.nextbig - (float(len(newState.okList))/len(newState.segList)) * (float(len(newState.okList))/len(newState.segList))
    else:
        nextB = newState.nextbig
    if minimax:
        nextB = -1 * nextB
    segcount = len(newState.segList)
    if newState.maxscore - newState.minscore < 0:
        segcount = -1 * segcount
    return ((newState.maxscore - newState.minscore + nextB) * 700) - segcount, newState, pick


def successors(state, player, size):
    movelist = []
    if not player:
        for i in range(0, len(state.segList)):
            nextStateDict = select(state, i, player, size)
            hq.heappush(movelist, nextStateDict)
    else:
        for i in range(0, len(state.segList)):
            nextStateDict = select(state, i, player, size)
            newtup = (-nextStateDict[0], nextStateDict[1], nextStateDict[2])
            hq.heappush(movelist, newtup)
    return movelist


def minimax(state, depth, player, size, alpha, beta, limit, clock):
    if len(state.segList) == 0:
        if state.maxscore > state.minscore:
            return sys.maxint
        elif state.maxscore < state.minscore:
            return -sys.maxint
        else:
            return 0
    if time.time() - start >= clock or depth == limit:
        nextB = state.nextbig - (float(len(state.okList))/len(state.segList)) * (float(len(state.okList))/len(state.segList))
        if not player:
            nextB = -1 * nextB
        segcount = len(state.segList)
        if state.maxscore - state.minscore < 0:
            segcount = -1 * segcount
        return ((state.maxscore - state.minscore + nextB) * 700) - segcount
    if player:
        v = -sys.maxint
        bestchild = None
        for child in successors(state, player, size):
            tmp = v
            v = max(v, minimax(child[1], depth + 1, False, size, alpha, beta, limit, clock))
            if depth == 0:
                if tmp < v:
                    bestchild = child
            if beta <= v:
                break
            alpha = max(alpha, v)
        if depth == 0:
            return bestchild
        return v
    else:
        v = sys.maxint
        for child in successors(state, player, size):
            v = min(v, minimax(child[1], depth + 1, True, size, alpha, beta, limit, clock))
            if v <= alpha:
                break
            beta = min(beta, v)
        return v


def output(result, board, n):
    file_output = open("output.txt", 'w')
    file_output.write(result)
    for x in range(0, n):
        file_output.write("\n")
        for y in range(0, n):
            if board[x][y] == 10:
                file_output.write('*')
            else:
                file_output.write(str(board[x][y]))


start = time.time()

init = parser("input.txt")
initseglist = segmentation(init["board"], init["list"], init["size"])
initState = State(init["board"], initseglist[0], init["list"], 0, 0, initseglist[1])
clock = min(init["time"]/(float(len(initseglist[0]))/2), init["time"])

cutoff = round(math.sqrt(float(676/len(initseglist[0]))))

if time.time() - start >= clock * 0.99:
    a = successors(initState, True, init["size"])[0]
else:
    a = minimax(initState, 0, True, init["size"], -sys.maxint, sys.maxint, cutoff, clock)

choice = initseglist[0][a[2]].list[0]
choicex = choice / init["size"]
choicey = choice % init["size"]
num2alpha = dict(zip(range(0, 26), string.ascii_uppercase))
move = num2alpha[choicey] + str(choicex + 1)

output(move, a[1].board, init["size"])
