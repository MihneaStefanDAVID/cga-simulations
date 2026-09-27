"""Take the UI screenshots used in README.md (macOS, Google Chrome).

    python docs/make_figures.py                 # creates docs/_build/readme_* results
    cp -r docs/_build/readme_* results/          # so that "Browse results" can show them
    streamlit run app.py --server.headless true --server.port 8599 --theme.base light &
    python docs/take_screenshots.py docs/images

Drives headless Chrome over the DevTools protocol (tornado ships with Streamlit), so it waits until
Streamlit has actually rendered the page.
"""

import asyncio, base64, json, subprocess, sys, tempfile, time, urllib.request
from tornado.websocket import websocket_connect
CH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = sys.argv[1]
SHOTS = [  # (file, url, width, height, wait seconds, action)
  ("ui_about.png", "?mode=About+the+algorithm", 1440, 960, 6, None),
  ("ui_trajectory.png", "?mode=Trajectory", 1440, 1110, 6, None),
  ("ui_sweep.png", "?mode=Sweep", 1440, 1180, 6, None),
  ("ui_load.png", "?mode=Load+from+file", 1440, 1000, 5, "project"),
  ("ui_results_trajectory.png", "?mode=Browse+results&exp=readme_trajectory", 1440, 1000, 8, None),
  ("ui_results_sweep.png", "?mode=Browse+results&exp=readme_sweep", 1440, 1180, 8, None),
]
proc = subprocess.Popen([CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--remote-debugging-port=9333",
                         f"--user-data-dir={tempfile.mkdtemp()}", "about:blank"], stderr=subprocess.DEVNULL)
for _ in range(50):
    try:
        tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9333/json")); break
    except Exception: time.sleep(0.2)
ws_url = [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"]
async def main():
    ws = await websocket_connect(ws_url, max_message_size=200*1024*1024)
    mid = 0
    async def call(method, **params):
        nonlocal mid; mid += 1; my = mid
        ws.write_message(json.dumps({"id": my, "method": method, "params": params}))
        while True:
            msg = json.loads(await ws.read_message())
            if msg.get("id") == my: return msg.get("result", {})
    await call("Page.enable")
    for name, q, w, h, wait, action in SHOTS:
        await call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=False)
        await call("Page.navigate", url="http://localhost:8599/" + q)
        await asyncio.sleep(wait)
        if action == "project":
            await call("Runtime.evaluate", expression="[...document.querySelectorAll('label')].find(l => l.innerText.includes(\"project's experiments.yaml\")).click()")
            await asyncio.sleep(3)
        r = await call("Page.captureScreenshot", format="png")
        open(f"{OUT}/{name}", "wb").write(base64.b64decode(r["data"])); print("shot", name)
asyncio.run(main()); proc.terminate()
