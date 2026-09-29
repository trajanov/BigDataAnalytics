"""CS-777 Chapter 9 — a live web dashboard over the GDELT stream.

Spark does not serve web pages. The streaming query writes its latest result to
a sink, and a separate program reads that result and presents it. Here the sink
is a JSON snapshot that `09.04 GDELT Stream.py` rewrites after every
micro-batch, and this program is a small Flask web app that reads the snapshot
and draws it. Run all three, each in its own terminal, from the same folder:

    .venv/bin/python3 "code/scripts/09.03 GDELT Feed.py"
    .venv/bin/python3 "code/scripts/09.04 GDELT Stream.py"
    .venv/bin/python3 "code/scripts/09.05 GDELT Dashboard.py"

then open http://127.0.0.1:8050 in a browser. The page asks the app for fresh
data every five seconds, so each new export shows up without a reload.

The two programs share nothing but a file. Either can be stopped and restarted
without the other noticing, which is the point of putting a sink between them.

Countries with fewer than MIN_EVENTS events are left out of the Goldstein
rankings, so a single event cannot top the chart. Stop the app with Ctrl-C.
"""

import json
import os
import sys

from flask import Flask, jsonify

# Relative to wherever the program is started, the same folder 09.04 writes to.
DASHBOARD_PATH = os.path.join(os.getcwd(), "gdelt-dashboard")
PORT = 8050       # not Flask's usual 5000, which macOS's AirPlay Receiver takes
TOP = 10          # countries per ranking
MIN_EVENTS = 5    # fewest events a country needs to appear in a Goldstein ranking

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GDELT Stream Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #f5f6f8; color: #1f2933; }
  header { padding: 16px 24px; background: #1f2933; color: #fff; }
  header h1 { margin: 0; font-size: 20px; }
  #status { margin-top: 4px; font-size: 14px; color: #cbd2d9; }
  main { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
         gap: 16px; padding: 16px 24px; }
  section { background: #fff; border-radius: 8px; padding: 12px 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  section h2 { margin: 0 0 8px; font-size: 15px; }
  .chart { position: relative; height: 320px; }
</style>
</head>
<body>
<header>
  <h1>GDELT world events: a live view of the Spark stream</h1>
  <div id="status">waiting for the first micro-batch...</div>
</header>
<main>
  <section><h2>Most positive countries (average Goldstein scale)</h2>
    <div class="chart"><canvas id="positive"></canvas></div></section>
  <section><h2>Most negative countries (average Goldstein scale)</h2>
    <div class="chart"><canvas id="negative"></canvas></div></section>
  <section><h2>Countries with the most events</h2>
    <div class="chart"><canvas id="events"></canvas></div></section>
  <section><h2>Total events after each micro-batch</h2>
    <div class="chart"><canvas id="history"></canvas></div></section>
</main>
<script>
function bars(id, color) {
  return new Chart(document.getElementById(id), {
    type: "bar",
    data: { labels: [], datasets: [{ data: [], backgroundColor: color }] },
    options: { indexAxis: "y", maintainAspectRatio: false, animation: false,
               plugins: { legend: { display: false } } }
  });
}
const charts = {
  positive: bars("positive", "#2f9e44"),
  negative: bars("negative", "#e03131"),
  events: bars("events", "#1971c2"),
  history: new Chart(document.getElementById("history"), {
    type: "line",
    data: { labels: [], datasets: [{ data: [], borderColor: "#1971c2", tension: 0.2 }] },
    options: { maintainAspectRatio: false, animation: false,
               plugins: { legend: { display: false } },
               scales: { x: { title: { display: true, text: "batch" } } } }
  })
};

function show(chart, labels, values) {
  chart.data.labels = labels;
  chart.data.datasets[0].data = values;
  chart.update();
}

async function refresh() {
  try {
    const data = await (await fetch("/data")).json();
    if (data.waiting) {
      document.getElementById("status").textContent = data.waiting;
      return;
    }
    document.getElementById("status").textContent =
      `batch ${data.batch} | updated ${data.updated} | ${data.total} events | ` +
      `${data.unmapped} without a country | rankings need at least ${data.min_events} events`;
    for (const name of ["positive", "negative"]) {
      show(charts[name], data[name].map(c => c.CountryName),
           data[name].map(c => c.AverageGoldsteinScale.toFixed(2)));
    }
    show(charts.events, data.events.map(c => c.CountryName),
         data.events.map(c => c.NumberOfEvents));
    show(charts.history, data.history.map(h => h.batch), data.history.map(h => h.events));
  } catch (err) {
    document.getElementById("status").textContent = "dashboard app not reachable";
  }
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


app = Flask(__name__)
LATEST = os.path.join(DASHBOARD_PATH, "latest.json")


@app.route("/")
def index():
    return PAGE


@app.route("/data")
def data():
    # 09.04 replaces the file in one step, so a read sees either the old
    # snapshot or the new one, never a mixture.
    if not os.path.exists(LATEST):
        return jsonify(waiting=f"waiting for {LATEST} (is 09.04 running, from the same folder?)")
    with open(LATEST) as handle:
        snapshot = json.load(handle)

    # The stream sends every country; choosing what to show is the app's job.
    countries = snapshot["countries"]
    named = [c for c in countries if c["CountryName"]]
    ranked = [c for c in named if c["NumberOfEvents"] >= MIN_EVENTS]
    by_score = sorted(ranked, key=lambda c: c["AverageGoldsteinScale"])
    return jsonify(
        batch=snapshot["batch"],
        updated=snapshot["updated"],
        total=sum(c["NumberOfEvents"] for c in countries),
        unmapped=sum(c["NumberOfEvents"] for c in countries if not c["CountryName"]),
        min_events=MIN_EVENTS,
        positive=by_score[::-1][:TOP],
        negative=by_score[:TOP],
        events=sorted(named, key=lambda c: c["NumberOfEvents"], reverse=True)[:TOP],
        history=snapshot["history"],
    )


def main() -> int:
    print("reading snapshots from", DASHBOARD_PATH, flush=True)
    print(f"open http://127.0.0.1:{PORT} in a browser", flush=True)
    app.run(port=PORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
