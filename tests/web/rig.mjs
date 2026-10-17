// Run the skintokens web rig page in a windowed browser until it reports done or failed.
// Usage: node rig.mjs <chrome|firefox> <url> <out.glb> <log.txt> [max_secs]
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const [browserName, URL, outGlb, logFile, maxArg] = process.argv.slice(2);
const maxSecs = Number(maxArg || 1800);
const root = path.join(os.homedir(), '.cache/puppeteer/chrome');
const cv = fs.readdirSync(root).filter(d => d.startsWith('mac_arm-')).sort().at(-1);
const opts = browserName === 'firefox'
  ? { browser: 'firefox', executablePath: '/Applications/Firefox.app/Contents/MacOS/firefox', headless: false, defaultViewport: { width: 640, height: 360 }, protocolTimeout: 0 }
  : { browser: 'chrome', executablePath: path.join(root, cv, 'chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'), headless: false, defaultViewport: { width: 640, height: 360 }, protocolTimeout: 0 };
const HOOK = `(() => {
  window.__gpuerr = [];
  if (!self.GPUAdapter) return;
  const rd = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (...a) {
    const d = await rd.apply(this, a);
    d.addEventListener('uncapturederror', e => window.__gpuerr.push(e.error.constructor.name + ': ' + e.error.message.slice(0, 600)));
    return d;
  };
})();`;
const log = fs.createWriteStream(logFile);
const t0 = Date.now();
const stamp = () => ((Date.now() - t0) / 1000).toFixed(1).padStart(7);
const b = await puppeteer.launch(opts);
const pg = await b.newPage();
await pg.evaluateOnNewDocument(HOOK);
pg.on('console', m => log.write(`${stamp()} ${m.text()}\n`));
pg.on('pageerror', e => log.write(`${stamp()} PAGEERROR ${e.message}\n`));
await pg.goto(URL);
await pg.bringToFront();
let state = null;
while ((Date.now() - t0) / 1000 < maxSecs) {
  await new Promise(r => setTimeout(r, 5000));
  try { state = await pg.evaluate('window.__rig || null'); } catch (e) { log.write(`${stamp()} EVAL ${e.message}\n`); break; }
  if (state && (state.state === 'done' || state.state === 'failed')) break;
}
const errs = await pg.evaluate('window.__gpuerr').catch(() => ['<page gone>']);
const ua = await pg.evaluate('navigator.userAgent').catch(() => '?');
if (state && state.glb) { fs.writeFileSync(outGlb, Buffer.from(state.glb, 'base64')); delete state.glb; }
const summary = { browser: ua, secs: (Date.now() - t0) / 1000, gpu_errors: errs.length, first_gpu_errors: [...new Set(errs)].slice(0, 5), state };
console.log(JSON.stringify(summary));
log.end();
await b.close();
