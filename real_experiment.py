import os
import re
import socket
import subprocess
import threading
import time
import matplotlib
matplotlib.use("Agg")
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

GRAPHS_DIR = "graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

TR_TARGET = "8.8.8.8"
TR_MAX_HOPS = 15
TR_PROBES = 3
TR_TIMEOUT = 2
TR_N_RUNS = 3

TCP_TARGETS = {
	"Cloudflare (1.1.1.1)": ("1.1.1.1", 443),
	"Google (8.8.8.8)": ("8.8.8.8", 443),
	"GitHub": ("github.com", 443),
	"NPM registry": ("registry.npmjs.org", 443),
	"jsDelivr CDN": ("cdn.jsdelivr.net", 443),
	"Yandex": ("ya.ru", 443),
}
TCP_PROBES = 5
TCP_N_RUNS = 3

def _parse_hop(line):
	m = re.match(r'^\s*(\d+)', line)
	if not m:
		return None
	ttl = int(m.group(1))
	host_m = re.search(r'(\S+)\s+\(([^)]+)\)', line)
	ip = host_m.group(2) if host_m else '*'
	hostname = host_m.group(1) if host_m else '*'
	rtts = [float(x) for x in re.findall(r'(\d+\.?\d*)\s+ms', line)]
	return {'ttl': ttl, 'ip': ip, 'hostname': hostname, 'rtts': rtts}

def _trim_stars(hops):
	while hops and not hops[-1]['rtts']:
		hops.pop()
	return hops

def tr_sequential(target):
	cmd = [
		'traceroute',
		'-q', str(TR_PROBES),
		'-w', str(TR_TIMEOUT),
		'-m', str(TR_MAX_HOPS),
		target,
	]
	t0 = time.perf_counter()
	proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
	elapsed_ms = (time.perf_counter() - t0) * 1000
	hops = [h for h in (_parse_hop(l) for l in proc.stdout.splitlines()[1:]) if h]
	return _trim_stars(hops), elapsed_ms

def _probe_single_ttl(target, ttl):
	cmd = [
		'traceroute',
		'-f', str(ttl),
		'-m', str(ttl),
		'-q', str(TR_PROBES),
		'-w', str(TR_TIMEOUT),
		target,
	]
	t0 = time.perf_counter()
	proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
	elapsed_ms = (time.perf_counter() - t0) * 1000
	for line in proc.stdout.splitlines()[1:]:
		hop = _parse_hop(line)
		if hop:
			return hop, elapsed_ms
	return {'ttl': ttl, 'ip': '*', 'hostname': '*', 'rtts': []}, elapsed_ms

def tr_parallel(target, n_hops):
	results = {}
	t0 = time.perf_counter()
	with ThreadPoolExecutor(max_workers=n_hops) as pool:
		futures = {
			pool.submit(_probe_single_ttl, target, ttl): ttl
			for ttl in range(1, n_hops + 1)
		}
		for fut in as_completed(futures):
			hop, _ = fut.result()
			results[futures[fut]] = hop
	elapsed_ms = (time.perf_counter() - t0) * 1000
	return [results[t] for t in sorted(results)], elapsed_ms

def check_tr_available():
	try:
		proc = subprocess.run(
			['traceroute', '-q', '1', '-w', '1', '-m', '3', '8.8.8.8'],
			capture_output=True, text=True, timeout=10
		)
		hops = [h for h in (_parse_hop(l) for l in proc.stdout.splitlines()[1:]) if h]
		return any(h['rtts'] for h in hops)
	except Exception:
		return False

def run_tr_experiment():
	print(f"\n  Цель: {TR_TARGET}")
	print("  → Обнаружение маршрута...", end=' ', flush=True)
	route, _ = tr_sequential(TR_TARGET)
	n_hops = len(route)
	if n_hops == 0:
		print("маршрут не обнаружен (ICMP заблокирован?)")
		return None
	print(f"найдено хопов: {n_hops}")
	seq_times, par_times = [], []
	seq_last, par_last = None, None
	for run in range(TR_N_RUNS):
		print(f"  → Запуск {run + 1}/{TR_N_RUNS}...", end=' ', flush=True)
		seq_r, seq_t = tr_sequential(TR_TARGET)
		par_r, par_t = tr_parallel(TR_TARGET, n_hops)
		seq_times.append(seq_t)
		par_times.append(par_t)
		seq_last, par_last = seq_r, par_r
		print(f"SEQ={seq_t:.0f} мс  PAR={par_t:.0f} мс")
	return {
		'target': TR_TARGET,
		'n_hops': n_hops,
		'seq_times': seq_times,
		'par_times': par_times,
		'seq_avg': sum(seq_times) / TR_N_RUNS,
		'par_avg': sum(par_times) / TR_N_RUNS,
		'seq_results': seq_last,
		'par_results': par_last,
	}

def print_tr_results(exp):
	speedup = exp['seq_avg'] / exp['par_avg']
	print(f"\n  SEQ среднее : {exp['seq_avg']:.0f} мс")
	print(f"  PAR среднее : {exp['par_avg']:.0f} мс")
	print(f"  Ускорение   : {speedup:.1f}×")
	seq_by_ttl = {h['ttl']: h for h in exp['seq_results']}
	par_by_ttl = {h['ttl']: h for h in exp['par_results']}
	print(f"\n  {'№':>3}  {'IP':>16}  {'SEQ RTT':>10}  {'PAR RTT':>10}  {'Δ':>8}")
	print(f"  {'─' * 55}")
	for ttl in range(1, exp['n_hops'] + 1):
		sh = seq_by_ttl.get(ttl, {'ip': '*', 'rtts': []})
		ph = par_by_ttl.get(ttl, {'ip': '*', 'rtts': []})
		ip = sh['ip'] if sh['ip'] != '*' else ph.get('ip', '*')
		sm = _mean(sh['rtts'])
		pm = _mean(ph['rtts'])
		ss = f"{sm:.1f} мс" if sm else "    *"
		ps = f"{pm:.1f} мс" if pm else "    *"
		ds = f"{abs(sm - pm):.1f} мс" if (sm and pm) else "—"
		print(f"  {ttl:>3}  {ip:>16}  {ss:>10}  {ps:>10}  {ds:>8}")

def tcp_rtt(host, port, timeout=5.0):
	try:
		t0 = time.perf_counter()
		with socket.create_connection((host, port), timeout=timeout):
			pass
		return (time.perf_counter() - t0) * 1000
	except Exception:
		return None

def tcp_sequential(targets, probes):
	results = {name: [] for name in targets}
	t0 = time.perf_counter()
	for name, (host, port) in targets.items():
		for _ in range(probes):
			rtt = tcp_rtt(host, port)
			if rtt is not None:
				results[name].append(rtt)
	return results, (time.perf_counter() - t0) * 1000

def tcp_parallel(targets, probes):
	results = {name: [] for name in targets}
	lock = threading.Lock()
	def probe_one(name, host, port):
		return name, tcp_rtt(host, port)
	n_workers = len(targets) * probes
	t0 = time.perf_counter()
	with ThreadPoolExecutor(max_workers=n_workers) as pool:
		futures = [
			pool.submit(probe_one, name, host, port)
			for name, (host, port) in targets.items()
			for _ in range(probes)
		]
		for fut in as_completed(futures):
			name, rtt = fut.result()
			if rtt is not None:
				with lock:
					results[name].append(rtt)
	return results, (time.perf_counter() - t0) * 1000

def run_tcp_experiment():
	seq_times, par_times = [], []
	seq_last, par_last = None, None
	theoretical = len(TCP_TARGETS)
	for run in range(TCP_N_RUNS):
		print(f"  → Запуск {run + 1}/{TCP_N_RUNS}...", end=' ', flush=True)
		seq_r, seq_t = tcp_sequential(TCP_TARGETS, TCP_PROBES)
		par_r, par_t = tcp_parallel(TCP_TARGETS, TCP_PROBES)
		seq_times.append(seq_t)
		par_times.append(par_t)
		seq_last, par_last = seq_r, par_r
		print(f"SEQ={seq_t:.0f} мс  PAR={par_t:.0f} мс")
	return {
		'seq_times': seq_times,
		'par_times': par_times,
		'seq_avg': sum(seq_times) / TCP_N_RUNS,
		'par_avg': sum(par_times) / TCP_N_RUNS,
		'seq_results': seq_last,
		'par_results': par_last,
		'theoretical': theoretical,
	}

def print_tcp_results(exp):
	speedup = exp['seq_avg'] / exp['par_avg']
	print(f"\n  Теоретическое ускорение: ≤{exp['theoretical']}× (при равных RTT)")
	print(f"  SEQ среднее : {exp['seq_avg']:.0f} мс")
	print(f"  PAR среднее : {exp['par_avg']:.0f} мс")
	print(f"  Ускорение   : {speedup:.1f}×")
	print(f"\n  {'Хост':<25} {'SEQ RTT':>10} {'PAR RTT':>10} {'Δ':>8}")
	print(f"  {'─' * 57}")
	for name in TCP_TARGETS:
		sm = _mean(exp['seq_results'].get(name, []))
		pm = _mean(exp['par_results'].get(name, []))
		ss = f"{sm:.1f} мс" if sm else "—"
		ps = f"{pm:.1f} мс" if pm else "—"
		ds = f"{abs(sm - pm):.1f} мс" if (sm and pm) else "—"
		print(f"  {name:<25} {ss:>10} {ps:>10} {ds:>8}")

def _mean(values):
	return sum(values) / len(values) if values else None

def _save(filename):
	path = os.path.join(GRAPHS_DIR, filename)
	plt.savefig(path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f"  Сохранён: {path}")

def plot_tcp(exp):
	names = list(TCP_TARGETS.keys())
	seq_means = [_mean(exp['seq_results'].get(n, [])) or 0 for n in names]
	par_means = [_mean(exp['par_results'].get(n, [])) or 0 for n in names]
	x = list(range(len(names)))
	runs = list(range(1, TCP_N_RUNS + 1))
	fig, ax = plt.subplots(figsize=(8, 5))
	ax.plot(x, seq_means, 'b-o', linewidth=2, label='Последовательный', markersize=7)
	ax.plot(x, par_means, 'r-s', linewidth=2, label='Параллельный', markersize=7)
	ax.set_title('TCP: средний RTT по хостам', fontweight='bold')
	ax.set_xlabel('Хост')
	ax.set_ylabel('RTT (мс)')
	ax.set_xticks(x)
	ax.set_xticklabels(names, rotation=15, ha='right')
	ax.legend()
	ax.grid(True, alpha=0.3, linestyle='--')
	fig.tight_layout()
	_save('tcp_1_rtt_per_host.png')
	fig, ax = plt.subplots(figsize=(7, 5))
	ax.plot(runs, exp['seq_times'], 'b-o', linewidth=2, label='Последовательный', markersize=8)
	ax.plot(runs, exp['par_times'], 'r-s', linewidth=2, label='Параллельный', markersize=8)
	ax.axhline(exp['seq_avg'], color='blue', linestyle='--', alpha=0.4,
			label=f"SEQ avg = {exp['seq_avg']:.0f} мс")
	ax.axhline(exp['par_avg'], color='red', linestyle='--', alpha=0.4,
			label=f"PAR avg = {exp['par_avg']:.0f} мс")
	ax.set_title('TCP: полное время серии зондов по запускам', fontweight='bold')
	ax.set_xlabel('Запуск')
	ax.set_ylabel('Время (мс)')
	ax.set_xticks(runs)
	ax.legend()
	ax.grid(True, alpha=0.3, linestyle='--')
	_save('tcp_2_time_runs.png')

def plot_traceroute(tr_exp):
	seq_by_ttl = {h['ttl']: h for h in tr_exp['seq_results']}
	par_by_ttl = {h['ttl']: h for h in tr_exp['par_results']}
	ttls = list(range(1, tr_exp['n_hops'] + 1))
	seq_rtts = [_mean(seq_by_ttl.get(t, {}).get('rtts', [])) for t in ttls]
	par_rtts = [_mean(par_by_ttl.get(t, {}).get('rtts', [])) for t in ttls]
	hop_labels = [f"H{t}" for t in ttls]
	runs = list(range(1, TR_N_RUNS + 1))
	fig, ax = plt.subplots(figsize=(7, 5))
	valid = [(i, s, p) for i, (s, p) in enumerate(zip(seq_rtts, par_rtts)) if s and p]
	if valid:
		xi, si, pi = zip(*valid)
		ax.plot(xi, si, 'b-o', linewidth=2, label='Последовательный', markersize=6)
		ax.plot(xi, pi, 'r-s', linewidth=2, label='Параллельный', markersize=6)
	ax.set_title(f'Traceroute {tr_exp["target"]}: RTT по хопам', fontweight='bold')
	ax.set_xlabel('Хоп')
	ax.set_ylabel('RTT (мс)')
	ax.set_xticks(range(len(ttls)))
	ax.set_xticklabels(hop_labels)
	ax.legend()
	ax.grid(True, alpha=0.3, linestyle='--')
	_save('tr_1_rtt_per_hop.png')
	fig, ax = plt.subplots(figsize=(7, 5))
	ax.plot(runs, tr_exp['seq_times'], 'b-o', linewidth=2, label='Последовательный', markersize=8)
	ax.plot(runs, tr_exp['par_times'], 'r-s', linewidth=2, label='Параллельный', markersize=8)
	ax.axhline(tr_exp['seq_avg'], color='blue', linestyle='--', alpha=0.4,
			label=f"SEQ avg = {tr_exp['seq_avg']:.0f} мс")
	ax.axhline(tr_exp['par_avg'], color='red', linestyle='--', alpha=0.4,
			label=f"PAR avg = {tr_exp['par_avg']:.0f} мс")
	ax.set_title(f'Traceroute {tr_exp["target"]}: время по запускам', fontweight='bold')
	ax.set_xlabel('Запуск')
	ax.set_ylabel('Время (мс)')
	ax.set_xticks(runs)
	ax.legend()
	ax.grid(True, alpha=0.3, linestyle='--')
	_save('tr_2_time_runs.png')
	fig, ax = plt.subplots(figsize=(6, 5))
	BLUE, RED = '#2196F3', '#F44336'
	bars = ax.bar([0, 1], [tr_exp['seq_avg'], tr_exp['par_avg']], 0.5, color=[BLUE, RED], alpha=0.85)
	ax.errorbar([0], [tr_exp['seq_avg']],
				yerr=[[tr_exp['seq_avg'] - min(tr_exp['seq_times'])],
					[max(tr_exp['seq_times']) - tr_exp['seq_avg']]],
				fmt='none', color='darkblue', capsize=10, linewidth=2)
	ax.errorbar([1], [tr_exp['par_avg']],
				yerr=[[tr_exp['par_avg'] - min(tr_exp['par_times'])],
					[max(tr_exp['par_times']) - tr_exp['par_avg']]],
				fmt='none', color='darkred', capsize=10, linewidth=2)
	for bar, val in zip(bars, [tr_exp['seq_avg'], tr_exp['par_avg']]):
		ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
				f"{val:.0f} мс", ha='center', fontsize=11, fontweight='bold')
	speedup = tr_exp['seq_avg'] / tr_exp['par_avg']
	mid = 0.5
	max_h = max(tr_exp['seq_avg'], tr_exp['par_avg'])
	ax.annotate(f"Ускорение {speedup:.1f}×", xy=(mid, max_h + 12),
				ha='center', fontsize=12, fontweight='bold', color='#2E7D32')
	ax.set_title(f'Traceroute {tr_exp["target"]}: среднее ± разброс', fontweight='bold')
	ax.set_ylabel('Время (мс)')
	ax.set_xticks([0, 1])
	ax.set_xticklabels(['Последовательный', 'Параллельный'])
	ax.grid(True, alpha=0.3, axis='y', linestyle='--')
	_save('tr_3_speedup.png')

if __name__ == '__main__':
	print("=" * 60)
	print("  Реальный сетевой эксперимент")
	print("=" * 60)
	print("\n[Часть 1] ICMP traceroute")
	print(f"  Целевой хост    : {TR_TARGET}")
	print(f"  Зондов на хоп   : {TR_PROBES}")
	print(f"  Таймаут зонда   : {TR_TIMEOUT} с")
	print(f"  Запусков        : {TR_N_RUNS}")
	print()
	if check_tr_available():
		print("  ✓ ICMP доступен — запускаю traceroute-эксперимент")
		tr_exp = run_tr_experiment()
		if tr_exp:
			print()
			print_tr_results(tr_exp)
			plot_traceroute(tr_exp)
	else:
		print("  ✗ ICMP/UDP заблокирован в данной среде.")
		print("    Traceroute-эксперимент требует запуска на физической машине,")
		print("    где не блокируются raw sockets (ПК, ноутбук, VPS).")
		print()
		print("    Для запуска: python3 real_experiment.py")
		print("    (traceroute должен возвращать не только *)")
	print(f"\n[Часть 2] TCP-зонды (без ICMP)")
	print(f"  Хостов          : {len(TCP_TARGETS)}")
	print(f"  Зондов на хост  : {TCP_PROBES}")
	print(f"  Запусков        : {TCP_N_RUNS}")
	print(f"  Всего зондов SEQ: {len(TCP_TARGETS) * TCP_PROBES * TCP_N_RUNS}")
	print(f"  Всего зондов PAR: {len(TCP_TARGETS) * TCP_PROBES * TCP_N_RUNS}")
	print()
	tcp_exp = run_tcp_experiment()
	print()
	print_tcp_results(tcp_exp)
	plot_tcp(tcp_exp)
	print("\n" + "=" * 60)
	print(" ИТОГ")
	print("=" * 60)
	tcp_spd = tcp_exp['seq_avg'] / tcp_exp['par_avg']
	n = len(TCP_TARGETS)
	print(f"\n TCP-эксперимент:")
	print(f" SEQ = {tcp_exp['seq_avg']:.0f} мс  PAR = {tcp_exp['par_avg']:.0f} мс")
	print(f" Ускорение: {tcp_spd:.1f}× (теоретически ≤{n}×, т.к. {n} хостов)")
	print()
	print(f" Вывод: параллельный алгоритм быстрее в {tcp_spd:.1f}× на реальной сети.")