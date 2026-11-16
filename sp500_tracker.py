import urllib.request
import json
import csv
import os
import time
import math
import threading
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --- CONFIGURATION ---
ANCHOR_DATE_STR = "2026-11-16"   # Set your desired start date (e.g., "1970-01-02", "2016-11-04")
RISK_SCALING_FACTOR = 1000.0     # Adjust risk scaling factor
HTTP_HOST = "0.0.0.0"          # Use "0.0.0.0" to expose on your network
HTTP_PORT = 8000
LOOP_SECONDS = 60
MAX_HISTORY = 5000               # Max points kept in memory for the chart
DATA_DIR = os.environ.get("TRACKER_DATA_DIR", "/data")   # Persistent storage
HISTORY_FILE = os.path.join(DATA_DIR, "history.csv")
META_FILE = os.path.join(DATA_DIR, "meta.json")
HISTORY_FIELDS = ["t", "close", "trend", "pos", "pnl"]

# --- SHARED STATE (written by tracking loop, read by HTTP server) ---
STATE_LOCK = threading.Lock()
HISTORY = {
    "t": deque(maxlen=MAX_HISTORY),      # epoch seconds
    "close": deque(maxlen=MAX_HISTORY),
    "trend": deque(maxlen=MAX_HISTORY),
    "pos": deque(maxlen=MAX_HISTORY),
    "pnl": deque(maxlen=MAX_HISTORY),
}
META = {"anchor": "", "loop": LOOP_SECONDS}


# ---------------------------------------------------------------------------
# PERSISTENCE (append-only CSV in DATA_DIR + a small meta file)
# ---------------------------------------------------------------------------
def init_storage(anchor_date):
    """Prepare DATA_DIR and restore state. Returns (position, pnl, last_price, persist_ok)."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        probe = os.path.join(DATA_DIR, ".write_test")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
    except OSError as e:
        print(f"WARNING: cannot write to {DATA_DIR} ({e}). Running WITHOUT persistence.")
        return 0.0, 0.0, None, False

    config = {"anchor": str(anchor_date), "scale": RISK_SCALING_FACTOR}
    old = None
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE) as f:
                old = json.load(f)
        except (OSError, ValueError):
            old = None

    # Positions/PnL depend on anchor and scale: don't mix runs with different settings.
    if os.path.exists(HISTORY_FILE) and old != config:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archived = os.path.join(DATA_DIR, f"history.{stamp}.csv")
        os.rename(HISTORY_FILE, archived)
        print(f"Config changed (anchor/scale). Archived old history to {archived}; starting fresh.")

    with open(META_FILE, "w") as f:
        json.dump(config, f)

    position, pnl, last_price = 0.0, 0.0, None
    if os.path.exists(HISTORY_FILE):
        rows = deque(maxlen=MAX_HISTORY)
        with open(HISTORY_FILE, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append({k: float(r[k]) for k in HISTORY_FIELDS})
                except (KeyError, ValueError, TypeError):
                    continue  # skip a partial/corrupt line (e.g. crash mid-write)
        for r in rows:
            for k in HISTORY_FIELDS:
                HISTORY[k].append(r[k])
        if rows:
            last = rows[-1]
            position, pnl, last_price = last["pos"], last["pnl"], last["close"]
            print(f"Restored {len(rows)} points from {HISTORY_FILE} "
                  f"(position={position:.4f}, pnl={pnl:.2f}, last close={last_price:.2f}).")
    else:
        with open(HISTORY_FILE, "w", newline="") as f:
            csv.writer(f).writerow(HISTORY_FIELDS)
    return position, pnl, last_price, True


def persist_row(row):
    try:
        with open(HISTORY_FILE, "a", newline="") as f:
            csv.writer(f).writerow(row)
            f.flush()
            os.fsync(f.fileno())
    except OSError as e:
        print(f"WARNING: could not persist history row: {e}")


def fetch_sp500_daily():
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
        "?period1=0&period2=9999999999&interval=1d&events=div%2Csplit"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose", [None] * len(timestamps))

    rows = []
    for i, ts in enumerate(timestamps):
        o = quote["open"][i]
        h = quote["high"][i]
        l = quote["low"][i]
        c = quote["close"][i]
        v = quote["volume"][i]
        if c is None:
            continue
        date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        rows.append({
            "Date": date_str, "Open": o, "High": h, "Low": l,
            "Close": c, "Adj Close": adjclose[i] if adjclose else None, "Volume": v,
        })

    out_dir = os.path.expanduser("~/csv")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sp500_daily.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {out_path}")


def load_sp500_csv():
    path = os.path.expanduser("~/csv/sp500_daily.csv")
    if not os.path.exists(path):
        fetch_sp500_daily()

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "date": datetime.strptime(r["Date"], "%Y-%m-%d").date(),
                "close": float(r["Close"]),
            })
    rows.sort(key=lambda x: x["date"])
    return rows


def fit_ols_log_prices(closes):
    n = len(closes)
    times = list(range(n))
    log_c = [math.log(c) for c in closes]
    mean_t = sum(times) / n
    mean_log_c = sum(log_c) / n
    numerator = sum((times[i] - mean_t) * (log_c[i] - mean_log_c) for i in range(n))
    denominator = sum((times[i] - mean_t) ** 2 for i in range(n))
    beta = numerator / denominator
    alpha = mean_log_c - beta * mean_t
    return alpha, beta


def get_live_price():
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?interval=1m&range=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            result = data["chart"]["result"][0]
            quotes = result["indicators"]["quote"][0]["close"]
            valid_quotes = [q for q in quotes if q is not None]
            if valid_quotes:
                return valid_quotes[-1]
    except Exception as e:
        print(f"Error fetching live price: {e}")
    return None


def find_row_index_by_date(dates, target_date):
    """Finds exact match or snaps to the closest available trading day."""
    if target_date in dates:
        return dates.index(target_date)

    # If exact date not found (e.g. weekend/holiday), find the closest future trading day
    for i, d in enumerate(dates):
        if d >= target_date:
            print(f"Notice: Target date {target_date} not found in trading calendar. Snapped to nearest trading day: {d}")
            return i

    raise ValueError(f"Target date {target_date} is past the end of the historical dataset.")


# ---------------------------------------------------------------------------
# SVG RENDERING
# ---------------------------------------------------------------------------
W, H = 1120, 800
ML, MR = 78, 112          # left / right margins
PW = W - ML - MR          # plot width

# name, panel top, panel height
PANELS = [("S&amp;P 500 close vs. trend", 70, 270),
          ("Position size", 400, 150),
          ("Cumulative PnL", 610, 130)]

C_PRICE = "#1d4e89"
C_TREND = "#c27c0e"
C_POS = "#1f7a5c"
C_PNL = "#9c2f57"
C_INK = "#1c2430"
C_MUTED = "#6b7683"
C_GRID = "#d9dee4"
C_PAPER = "#f6f4ef"


def _fmt(v):
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.2f}M"
    if a >= 1e4:
        return f"{v / 1e3:.1f}k"
    if a >= 100:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:,.2f}"
    return f"{v:.3f}"


def _nice_range(lo, hi, include_zero=False):
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo < 1e-9:
        pad = max(abs(hi) * 0.01, 1.0)
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.1
    return lo - pad, hi + pad


def _panel(title, top, height, ts, series, t0, t1, include_zero, fill_zero=False, step=False, unit=""):
    """series: list of (values, color, dashed, label). Returns SVG string."""
    out = []
    allv = [v for vals, _, _, _ in series for v in vals]
    lo, hi = _nice_range(min(allv), max(allv), include_zero)

    def X(t):
        return ML + (t - t0) / (t1 - t0) * PW

    def Y(v):
        return top + height - (v - lo) / (hi - lo) * height

    out.append(f'<text x="{ML}" y="{top - 14}" class="ptitle">{title}</text>')
    # horizontal grid + y labels
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        y = Y(v)
        out.append(f'<line x1="{ML}" x2="{ML + PW}" y1="{y:.1f}" y2="{y:.1f}" stroke="{C_GRID}" stroke-width="1"/>')
        out.append(f'<text x="{ML - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{_fmt(v)}</text>')
    # zero line
    if lo < 0 < hi:
        y0 = Y(0)
        out.append(f'<line x1="{ML}" x2="{ML + PW}" y1="{y0:.1f}" y2="{y0:.1f}" stroke="{C_INK}" stroke-width="1" opacity=".55"/>')

    for vals, color, dashed, label in series:
        pts = []
        for i, (t, v) in enumerate(zip(ts, vals)):
            x, y = X(t), Y(v)
            if step and pts:
                pts.append((x, pts[-1][1]))
            pts.append((x, y))
        if len(pts) == 1:
            x, y = pts[0]
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
        else:
            d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            if fill_zero:
                yz = Y(0) if lo < 0 < hi else (top + height if lo >= 0 else top)
                area = d + f" L{pts[-1][0]:.1f},{yz:.1f} L{pts[0][0]:.1f},{yz:.1f} Z"
                out.append(f'<path d="{area}" fill="{color}" opacity=".13"/>')
            dash = ' stroke-dasharray="6 5"' if dashed else ""
            out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"{dash} stroke-linejoin="round"/>')
        # end-of-line marker + value label
        x, y = pts[-1]
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="{C_PAPER}" stroke-width="1.5"/>')
        out.append(f'<text x="{x + 9:.1f}" y="{y + 4:.1f}" class="endlab" fill="{color}">{_fmt(vals[-1])}{unit}</text>')

    # legend (right aligned in title row)
    lx = ML + PW
    for vals, color, dashed, label in reversed(series):
        w = 8 * len(label) + 34
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        out.append(f'<line x1="{lx - w}" x2="{lx - w + 22}" y1="{top - 18}" y2="{top - 18}" stroke="{color}" stroke-width="2.5"{dash}/>')
        out.append(f'<text x="{lx - w + 28}" y="{top - 14}" class="legend">{label}</text>')
        lx -= w + 6
    return "\n".join(out)


def render_svg():
    with STATE_LOCK:
        ts = list(HISTORY["t"])
        close = list(HISTORY["close"])
        trend = list(HISTORY["trend"])
        pos = list(HISTORY["pos"])
        pnl = list(HISTORY["pnl"])
        anchor = META["anchor"]

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" '
             f'font-family="\'IBM Plex Sans\', \'Segoe UI\', system-ui, sans-serif">',
             '<style>.ptitle{font-size:15px;font-weight:600;fill:%s}'
             '.tick{font-size:11px;fill:%s;font-variant-numeric:tabular-nums}'
             '.endlab{font-size:12px;font-weight:600;font-variant-numeric:tabular-nums}'
             '.legend{font-size:12px;fill:%s}'
             '.head{font-size:22px;font-weight:700;fill:%s}'
             '.sub{font-size:12px;fill:%s}</style>' % (C_INK, C_MUTED, C_MUTED, C_INK, C_MUTED),
             f'<rect width="{W}" height="{H}" rx="10" fill="{C_PAPER}"/>']

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts.append(f'<text x="{ML}" y="34" class="head">S&amp;P 500 live tracker</text>')
    parts.append(f'<text x="{W - MR}" y="34" text-anchor="end" class="sub">'
                 f'anchor {anchor} · scale {RISK_SCALING_FACTOR:g} · updated {now}</text>')

    if not ts:
        parts.append(f'<text x="{W / 2}" y="{H / 2}" text-anchor="middle" class="sub">'
                     f'Waiting for the first price…</text></svg>')
        return "\n".join(parts)

    t0, t1 = ts[0], ts[-1]
    if t1 - t0 < 1:
        t1 = t0 + 1

    (n1, top1, h1), (n2, top2, h2), (n3, top3, h3) = PANELS
    parts.append(_panel(n1, top1, h1, ts,
                        [(close, C_PRICE, False, "Close"), (trend, C_TREND, True, "Trend (c_hat)")],
                        t0, t1, False))
    parts.append(_panel(n2, top2, h2, ts, [(pos, C_POS, False, "Position")],
                        t0, t1, True, fill_zero=True, step=True))
    parts.append(_panel(n3, top3, h3, ts, [(pnl, C_PNL, False, "PnL")],
                        t0, t1, True, fill_zero=True))

    # shared time axis
    ybase = top3 + h3
    for i in range(6):
        t = t0 + (t1 - t0) * i / 5
        x = ML + PW * i / 5
        lab = datetime.fromtimestamp(t).strftime("%H:%M:%S" if t1 - t0 < 86400 else "%m-%d %H:%M")
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{ybase}" y2="{ybase + 5}" stroke="{C_MUTED}"/>')
        parts.append(f'<text x="{x:.1f}" y="{ybase + 20}" text-anchor="middle" class="tick">{lab}</text>')
        for top, h in ((top1, h1), (top2, h2)):
            parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + h}" stroke="{C_GRID}" stroke-width="1" opacity=".5"/>')
    parts.append("</svg>")
    return "\n".join(parts)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>S&amp;P 500 live tracker</title>
<style>
  body{margin:0;padding:24px;background:#e9e6df;font-family:system-ui,sans-serif;color:#1c2430}
  main{max-width:1160px;margin:0 auto}
  #chart{display:block}
  p{font-size:13px;color:#6b7683;margin:10px 4px}
  a{color:#1d4e89}
</style></head>
<body><main>
<div id="chart">Loading…</div>
<p>Refreshes every <span id="s">5</span>s · <a href="/chart.svg">raw SVG</a> · <a href="/data.json">data</a></p>
</main>
<script>
  const el = document.getElementById('chart');
  async function tick(){
    try{
      const r = await fetch('/chart.svg?_=' + Date.now(), {cache:'no-store'});
      el.innerHTML = await r.text();
    }catch(e){}
  }
  tick(); setInterval(tick, 5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(PAGE, "text/html; charset=utf-8")
        elif path == "/chart.svg":
            self._send(render_svg(), "image/svg+xml; charset=utf-8")
        elif path == "/data.json":
            with STATE_LOCK:
                payload = {k: list(v) for k, v in HISTORY.items()}
            self._send(json.dumps(payload), "application/json")
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass  # keep the console for tracking output


def start_http_server():
    server = ThreadingHTTPServer((HTTP_HOST, HTTP_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Dashboard live at http://{HTTP_HOST}:{HTTP_PORT}/  (SVG: /chart.svg)")


def live_tracking_loop():
    data = load_sp500_csv()
    closes = [r["close"] for r in data]
    dates = [r["date"] for r in data]

    alpha, beta = fit_ols_log_prices(closes)

    # Parse target anchor date and resolve to row index automatically
    target_anchor_date = datetime.strptime(ANCHOR_DATE_STR, "%Y-%m-%d").date()
    r_index = find_row_index_by_date(dates, target_anchor_date)

    c_r = closes[r_index]
    c_hat_r = math.exp(alpha + beta * r_index)
    anchor_date = dates[r_index]
    META["anchor"] = str(anchor_date)

    print(f"Loaded {len(closes)} historical rows.")
    print(f"Anchor Date -> Resolved to Row index r={r_index} (Date: {anchor_date}), c_r={c_r:.2f}, c_hat_r={c_hat_r:.2f}")
    print(f"Risk Scaling Factor Set To: {RISK_SCALING_FACTOR}")

    current_position_size, cumulative_pnl, last_price, persist_ok = init_storage(anchor_date)

    start_http_server()
    print("Starting live tracking loop with fixed daily trend index...\n")

    t_live = len(closes) - 1  # Fixed daily trend index for intraday loop

    while True:
        c_t = get_live_price()
        if c_t is None:
            print("Retrying price fetch in 10 seconds...")
            time.sleep(10)
            continue

        c_hat_t = math.exp(alpha + beta * t_live)

        # Mark-to-market PnL: position held over the last interval x price change
        if last_price is not None:
            cumulative_pnl += current_position_size * (c_t - last_price)
        last_price = c_t

        # --- LOG-SPACE FORMULA ---
        value_t = math.log(c_hat_t / c_hat_r) - math.log(c_t / c_hat_t)
        target_position_size = value_t * RISK_SCALING_FACTOR

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Price (c): {c_t:.2f} | Trend (c_hat): {c_hat_t:.2f} | "
              f"Formula Value: {value_t:.4f} | Scaled Position: {target_position_size:.4f} | "
              f"PnL: {cumulative_pnl:.2f}")

        if target_position_size != current_position_size:
            print(f" >>> POSITION ADJUSTMENT: Changing position size from {current_position_size:.4f} to {target_position_size:.4f} <<<")
            current_position_size = target_position_size

        row = [time.time(), c_t, c_hat_t, current_position_size, cumulative_pnl]
        with STATE_LOCK:
            for k, v in zip(HISTORY_FIELDS, row):
                HISTORY[k].append(v)
        if persist_ok:
            persist_row(row)

        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    live_tracking_loop()
