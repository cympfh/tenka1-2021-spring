import functools
import json
import logging
import math
import random
import time
from collections import defaultdict
from typing import List, Optional, Tuple
from urllib.request import urlopen

import rich
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")

# ゲームサーバのアドレス
GAME_SERVER = "https://contest.2021-spring.gbc.tenka1.klab.jp"
# あなたのトークン
TOKEN = "f918f14864ee745d8c54f133176a376e"
# GameAPIを呼ぶ際のインターバル
GAME_INFO_SLEEP_TIME = 1100


def call_api(x: str) -> bytes:
    with urlopen(f"{GAME_SERVER}{x}") as res:
        return res.read()


def call_move(index: int, x: int, y: int) -> dict:
    res_str = call_api(f"/api/move/{TOKEN}/{index}-{x}-{y}")
    res_json = json.loads(res_str)
    return res_json


def call_move_next(index: int, x: int, y: int) -> dict:
    res_str = call_api(f"/api/move_next/{TOKEN}/{index}-{x}-{y}")
    res_json = json.loads(res_str)
    return res_json


def call_game() -> dict:
    res_str = call_api(f"/api/game/{TOKEN}")
    res_json = json.loads(res_str)
    return res_json


def call_master_data() -> dict:
    res_str = call_api("/api/master_data")
    res_json = json.loads(res_str)
    return res_json


@functools.lru_cache()
def compose(xs: Tuple[str]) -> Optional[str]:
    s = xs[0]
    for x in xs[1:]:
        max_common_len = 0
        m = min(10, len(s), len(x))
        for c in reversed(range(1, m)):
            if s[-c:] == x[:c]:
                max_common_len = c
                break
        if max_common_len == 0:
            return None
        s = s + x[max_common_len:]
    return s


class Graph:
    def __init__(self, tasks):
        n = len(tasks)
        g = [[] for _ in range(n)]
        clen = defaultdict(int)
        for i in range(n):
            for j in range(n):
                max_common_len = 0
                s: str = tasks[i]["s"]
                t: str = tasks[j]["s"]
                if s != t and s.endswith(t):
                    max_common_len = len(t)
                elif s != t and t.startswith(s):
                    max_common_len = len(s)
                else:
                    m = min(10, len(s), len(t))
                    for c in reversed(range(1, m)):
                        if s[-c:] == t[:c]:
                            max_common_len = max(max_common_len, c)
                            break
                if max_common_len > 0:
                    g[i].append(j)
                    clen[(i, j)] = max_common_len
        self.g = g
        self.clen = clen

    def is_connected(self, i, j) -> bool:
        return self.clen[(i, j)] > 0

    def compose(self, tasks, vs: List[int]) -> Optional[str]:
        s = tasks[vs[0]]["s"]
        for i in range(1, len(vs)):
            t = tasks[vs[i]]["s"]
            clen = self.clen[(vs[i - 1], vs[i])]
            if clen == 0:
                return None
            s = s + t[clen:]
        return s


class Bot:
    def __init__(self):
        self.game_info = call_game()
        self.time = 0
        self.graph = Graph(self.game_info["task"])
        self.master_data = call_master_data()
        self.start_game_time_ms = self.game_info["now"]
        log.info("Start: %s", self.start_game_time_ms)
        self.start_time_ms = time.perf_counter_ns() // 1000000
        self.next_call_game_info_time_ms = (
            self.get_now_game_time_ms() + GAME_INFO_SLEEP_TIME
        )
        self.agent_move_finish_ms = [0] * self.master_data["num_agent"]
        self.agent_move_point_queue = [[] for _ in range(self.master_data["num_agent"])]
        self.agent_last_point = [[] for _ in range(self.master_data["num_agent"])]
        for i in range(self.master_data["num_agent"]):
            agent_move = self.game_info["agent"][i]["move"]
            self.agent_last_point[i] = [agent_move[-1]["x"], agent_move[-1]["y"]]
            self.set_move_point(i)

    def length_of_run(self, s: str) -> float:
        """`s` の最初から最後までの距離"""
        n = len(s)
        z = 0.0
        for i in range(n - 1):
            u = ord(s[i]) - 65
            v = ord(s[i + 1]) - 65
            p = self.master_data["checkpoints"][u]
            q = self.master_data["checkpoints"][v]
            z += math.sqrt((p["x"] - q["x"]) ** 2 + (p["y"] - q["y"]) ** 2)
        return z

    @functools.lru_cache()
    def length_of_loop(self, s: str) -> float:
        n = len(s)
        z = 0.0
        for i in range(n):
            u = ord(s[i]) - 65
            v = ord(s[(i + 1) % n]) - 65
            p = self.master_data["checkpoints"][u]
            q = self.master_data["checkpoints"][v]
            z += math.sqrt((p["x"] - q["x"]) ** 2 + (p["y"] - q["y"]) ** 2)
        return z

    def choice_task(self, index: int = 0) -> str:
        """効率の良いタスク列を返"""
        tasks = self.game_info["task"]
        n = len(tasks)

        ws = []
        ts = []
        for i in range(n):
            ws.append(float(tasks[i]["weight"]))
            ts.append(float(tasks[i]["total"] + 1.0))

        max_score = -1.0
        ret = tasks[0]["s"]  # DUMMY
        max_i = -1
        max_j = -1
        max_k = -1

        # 一つをぐるぐる
        for i in range(n):
            z = self.length_of_loop(tasks[i]["s"])
            if z < 1.0:
                continue
            score = ws[i] / ts[i] / z
            # log.debug("Loop1: %s, score=%s, z=%s", i, score, z)
            if score > max_score:
                max_score = score
                max_i = i
                ret = tasks[i]["s"]

        # 2つをぐるぐる
        for i in range(n):
            for j in self.graph.g[i]:
                composed = self.graph.compose(tasks, [i, j])
                if composed is None:
                    continue
                if len(composed) < 2:
                    continue
                z = self.length_of_loop(composed)
                if z < 1.0:
                    continue
                score = (ws[i] / ts[i] + ws[j] / ts[j]) / z
                # log.debug("Loop2: %s + %s, score=%s, z=%s", i, j, score, z)
                if score > max_score:
                    max_score = score
                    ret = composed
                    max_i = i
                    max_j = j

        # 3つ
        for i in range(n):
            for j in self.graph.g[i]:
                for k in self.graph.g[j]:
                    composed = self.graph.compose(tasks, [i, j, k])
                    if composed is None:
                        continue
                    if len(composed) < 2:
                        continue
                    z = self.length_of_loop(composed)
                    if z < 1.0:
                        continue
                    score = (ws[i] / ts[i] + ws[j] / ts[j] + ws[k] / ts[k]) / z
                    if score > max_score:
                        max_score = score
                        ret = composed
                        max_i = i
                        max_j = j
                        max_k = k

        if max_j == -1:
            log.info("Agent#%s, Task Loop1: %s score=%s from %s", index, ret, max_score, tasks[max_i])
        elif max_k == -1:
            log.info(
                "Agent#%s, Task Loop2: %s score=%s from %s + %s",
                index,
                ret,
                max_score,
                tasks[max_i],
                tasks[max_j],
            )
        else:
            log.info(
                "Agent#%s, Task Loop3: %s score=%s from %s + %s + %s",
                index,
                ret,
                max_score,
                tasks[max_i],
                tasks[max_j],
                tasks[max_k],
            )
        if max_i >= 0:
            self.game_info["task"][max_i]["total"] += 1
        if max_j >= 0:
            self.game_info["task"][max_j]["total"] += 1
        if max_k >= 0:
            self.game_info["task"][max_k]["total"] += 1
        return ret

    def get_now_game_time_ms(self) -> int:
        now_ms = time.perf_counter_ns() // 1000000
        return self.start_game_time_ms + (now_ms - self.start_time_ms)

    def get_checkpoint(self, name: str) -> Tuple[int, int]:
        index = ord(name) - ord("A")
        checkpoint = self.master_data["checkpoints"][index]
        return checkpoint["x"], checkpoint["y"]

    def set_move_point(self, index: int):
        """移動予定を設定"""
        s = self.choice_task(index)
        # next_task = self.choice_task(index)
        log.info("Agent#%s, next run => %s", index, s)
        for i in range(len(s)):
            before_x = self.agent_last_point[index][0]
            before_y = self.agent_last_point[index][1]
            move_x, move_y = self.get_checkpoint(s[i])

            # 移動先が同じ場所の場合判定が入らないため別の箇所に移動してからにする
            if move_x == before_x and move_y == before_y:
                tmp_x = before_x + 1
                tmp_y = before_y
                if tmp_x > 30:
                    tmp_x = 28
                self.agent_move_point_queue[index].append([tmp_x, tmp_y])

            self.agent_move_point_queue[index].append([move_x, move_y])
            self.agent_last_point[index] = [move_x, move_y]

    def move_next(self, index: int) -> dict:
        move_next_point = self.agent_move_point_queue[index].pop(0)
        move_next_res = call_move_next(
            index + 1, move_next_point[0], move_next_point[1]
        )
        if move_next_res["status"] != "ok":
            log.warning(move_next_res)
            return
        assert len(move_next_res["move"]) > 1
        self.agent_move_finish_ms[index] = move_next_res["move"][1]["t"] + 100
        # タスクを全てやりきったら次のタスクを取得
        if not self.agent_move_point_queue[index]:
            self.set_move_point(index)
        return move_next_res

    def get_now_score(self) -> float:
        """game_infoの状態でのスコアを取得"""
        score = 0.0
        tasks = self.game_info["task"]
        for i in range(len(tasks)):
            if tasks[i]["total"] == 0:
                continue
            score += tasks[i]["weight"] * tasks[i]["count"] / tasks[i]["total"]
        return score

    def solve(self):
        while True:
            now_game_time_ms = self.get_now_game_time_ms()

            # エージェントを移動させる
            for i in range(self.master_data["num_agent"]):
                if self.agent_move_finish_ms[i] < now_game_time_ms:
                    move_next_res = self.move_next(i)
                    # 次の移動予定がない場合もう一度実行する
                    if len(move_next_res["move"]) == 2:
                        self.move_next(i)

            # ゲーム情報更新
            if self.next_call_game_info_time_ms < now_game_time_ms:
                log.info("Update GameInfo")
                self.game_info = call_game()
                if self.time % 20 == 0:
                    log.info("%d Tasks", len(self.game_info["task"]))
                    for task in self.game_info["task"]:
                        log.info(task)
                # 前処理
                tasks = list(self.game_info["task"])
                self.game_info["task"] = [
                    task for task in tasks if task["weight"] > 100
                ]
                self.graph = Graph(self.game_info["task"])

                self.next_call_game_info_time_ms = (
                    self.get_now_game_time_ms() + GAME_INFO_SLEEP_TIME
                )
                log.warning("Score: %s", self.get_now_score())

            self.time += 1
            time.sleep(0.1)


if __name__ == "__main__":
    bot = Bot()
    bot.solve()
