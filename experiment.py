import os
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRAPHS_DIR = "graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

BLUE = "#2196F3"
RED = "#F44336"

@dataclass
class HopConfig:
    name: str
    latency_ms: float
    jitter_ms: float
    loss_prob: float

class NetworkSimulator:
    SCALE = 0.0001

    def __init__(self, hops):
        self.hops = hops
        cumulative = 0.0
        self.true_rtt = []
        for h in hops:
            cumulative += h.latency_ms
            self.true_rtt.append(round(cumulative * 2, 2))

    def probe(self, ttl):
        if not (1 <= ttl <= len(self.hops)):
            return None

        for i in range(ttl):
            if random.random() < self.hops[i].loss_prob:
                return None

        rtt = sum(
            h.latency_ms + random.gauss(0, h.jitter_ms)
            for h in self.hops[:ttl]
        ) * 2
        rtt = max(0.1, rtt)
        time.sleep(rtt * self.SCALE)
        return rtt

def sequential_traceroute(net, probes=3):
    results = []
    t0 = time.perf_counter()

    for ttl in range(1, len(net.hops) + 1):
        rtts = []
        for _ in range(probes):
            rtt = net.probe(ttl)
            if rtt is not None:
                rtts.append(rtt)
        results.append((ttl, rtts))

    return results, (time.perf_counter() - t0) * 1000

def parallel_traceroute(net, probes=3):
    n = len(net.hops)
    raw = {ttl: [] for ttl in range(1, n + 1)}
    lock = threading.Lock()

    def send(ttl):
        return ttl, net.probe(ttl)

    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n * probes) as pool:
        futures = [
            pool.submit(send, ttl)
            for ttl in range(1, n + 1)
            for _ in range(probes)
        ]
        for fut in futures:
            ttl, rtt = fut.result()
            if rtt is not None:
                with lock:
                    raw[ttl].append(rtt)

    elapsed = (time.perf_counter() - t0) * 1000
    return [(ttl, raw[ttl]) for ttl in range(1, n + 1)], elapsed

def mean(values):
    return sum(values) / len(values) if values else None

def rtt_error_pct(results, true_rtt):
    errors = []
    for (_, rtts), true in zip(results, true_rtt):
        m = mean(rtts)
        errors.append(abs(m - true) / true * 100 if m is not None else None)
    return errors

def observed_loss_pct(results, probes):
    return [(1 - len(rtts) / probes) * 100 for _, rtts in results]

TOPOLOGY_CLEAN = [
    HopConfig("Router-1 (ISP)", 5, 1.0, 0.00),
    HopConfig("Router-2", 8, 1.5, 0.00),
    HopConfig("Router-3", 6, 1.2, 0.00),
    HopConfig("Router-4 (IXP)", 35, 5.0, 0.00),
    HopConfig("Router-5", 12, 2.0, 0.00),
    HopConfig("Router-6", 9, 1.8, 0.00),
    HopConfig("Router-7", 7, 1.5, 0.00),
    HopConfig("Router-8 (DST)", 4, 0.8, 0.00),
]

TOPOLOGY_LOSSY = [
    HopConfig("Router-1 (ISP)", 5, 1.0, 0.00),
    HopConfig("Router-2", 8, 1.5, 0.05),
    HopConfig("Router-3", 6, 1.2, 0.05),
    HopConfig("Router-4 (IXP)", 35, 5.0, 0.15),
    HopConfig("Router-5", 12, 2.0, 0.10),
    HopConfig("Router-6", 9, 1.8, 0.05),
    HopConfig("Router-7", 7, 1.5, 0.00),
    HopConfig("Router-8 (DST)", 4, 0.8, 0.00),
]

N_RUNS = 5
PROBES = 3

def run_scenario(net, label):
    seq_times, par_times = [], []
    seq_last, par_last = None, None

    for run in range(N_RUNS):
        random.seed(run * 7 + 13)
        seq_r, seq_t = sequential_traceroute(net, PROBES)
        par_r, par_t = parallel_traceroute(net, PROBES)
        seq_times.append(seq_t)
        par_times.append(par_t)
        seq_last, par_last = seq_r, par_r

    return {
        "label": label,
        "net": net,
        "seq_times": seq_times,
        "par_times": par_times,
        "seq_avg": sum(seq_times) / N_RUNS,
        "par_avg": sum(par_times) / N_RUNS,
        "seq_results": seq_last,
        "par_results": par_last,
        "seq_err": rtt_error_pct(seq_last, net.true_rtt),
        "par_err": rtt_error_pct(par_last, net.true_rtt),
        "seq_loss": observed_loss_pct(seq_last, PROBES),
        "par_loss": observed_loss_pct(par_last, PROBES),
    }

def print_scenario(exp):
    speedup = exp["seq_avg"] / exp["par_avg"]
    print(f"\n{'═' * 70}")
    print(f"  Сценарий: {exp['label']}")
    print(f"{'═' * 70}")
    print(f"  Последовательный (среднее по {N_RUNS} запускам): {exp['seq_avg']:.1f} мс")
    print(f"  Параллельный (среднее по {N_RUNS} запускам): {exp['par_avg']:.1f} мс")
    print(f"  Ускорение: {speedup:.1f}×\n")

    hdr = f"  {'Хоп':<22} {'True RTT':>10} {'SEQ RTT':>10} {'PAR RTT':>10} {'SEQ err%':>9} {'PAR err%':>9}"
    print(hdr)
    print(f"  {'─' * (len(hdr) - 2)}")

    for i, hop in enumerate(exp["net"].hops):
        true = exp["net"].true_rtt[i]
        seq_m = mean(exp["seq_results"][i][1])
        par_m = mean(exp["par_results"][i][1])
        se = exp["seq_err"][i]
        pe = exp["par_err"][i]
        print(
            f" {hop.name:<22}"
            f" {true:>10.1f}"
            f" {seq_m:>10.1f}" if seq_m else f" {'—':>10}",
            end="",
        )
        print(
            f" {par_m:>10.1f}" if par_m else f" {'—':>10}",
            f" {se:>9.1f}" if se is not None else f" {'—':>9}",
            f" {pe:>9.1f}" if pe is not None else f" {'—':>9}",
        )

def _save(name):
    path = os.path.join(GRAPHS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Сохранён: {path}")

def _hop_labels(exp):
    return [f"H{i+1}" for i in range(len(exp["net"].hops))]

def plot_rtt_error(exp, title, filename):
    x = list(range(len(exp["net"].hops)))
    labels = _hop_labels(exp)
    seq_e = [e if e is not None else 0 for e in exp["seq_err"]]
    par_e = [e if e is not None else 0 for e in exp["par_err"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, seq_e, "b-o", linewidth=2, label="Последовательный", markersize=6)
    ax.plot(x, par_e, "r-s", linewidth=2, label="Параллельный", markersize=6)
    ax.axhline(15, color="orange", linestyle="--", linewidth=1.5, label="Порог 15%")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Хоп")
    ax.set_ylabel("Погрешность (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    _save(filename)

def plot_time_runs(exp_clean, exp_lossy):
    runs = list(range(1, N_RUNS + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(runs, exp_clean["seq_times"], "b-o", label="SEQ (без потерь)", markersize=6)
    ax.plot(runs, exp_clean["par_times"], "b--s", label="PAR (без потерь)", markersize=6, alpha=0.75)
    ax.plot(runs, exp_lossy["seq_times"], "r-o", label="SEQ (с потерями)", markersize=6)
    ax.plot(runs, exp_lossy["par_times"], "r--s", label="PAR (с потерями)", markersize=6, alpha=0.75)
    ax.set_title("Время выполнения по запускам", fontweight="bold")
    ax.set_xlabel("Номер запуска")
    ax.set_ylabel("Время (мс)")
    ax.set_xticks(runs)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    _save("sim_3_time_runs.png")

if __name__ == "__main__":
    net_clean = NetworkSimulator(TOPOLOGY_CLEAN)
    theoretical = sum(net_clean.true_rtt) * PROBES / net_clean.true_rtt[-1]

    print("Пилотный эксперимент: Параллельный vs Последовательный Traceroute")
    print(f"Хопов: 8 | Зондов/хоп: {PROBES} | Запусков: {N_RUNS} | Масштаб задержки: ×{NetworkSimulator.SCALE}")
    print(f"Теоретическое ускорение: {theoretical:.1f}×  "
          f"(3 × {sum(net_clean.true_rtt):.0f} мс / {net_clean.true_rtt[-1]:.0f} мс)")

    net_lossy = NetworkSimulator(TOPOLOGY_LOSSY)

    print("\nСценарий: без потерь...")
    exp_clean = run_scenario(net_clean, "Без потерь пакетов")

    print("Сценарий: с потерями пакетов...")
    exp_lossy = run_scenario(net_lossy, "С потерями пакетов")

    print_scenario(exp_clean)
    print_scenario(exp_lossy)

    print("\nСохранение графиков...")
    plot_time_runs(exp_clean, exp_lossy)
    plot_rtt_error(exp_clean, "Погрешность RTT — сценарий без потерь", "sim_rtt_error_clean.png")
    plot_rtt_error(exp_lossy, "Погрешность RTT — сценарий с потерями", "sim_rtt_error_lossy.png")