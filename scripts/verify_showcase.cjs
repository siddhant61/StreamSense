#!/usr/bin/env node
/*
 * Headless-browser verifier for docs/index.html, the research showcase that
 * GitHub Pages serves from this repository's docs/ folder.
 *
 * It serves docs/ from a loopback-only static server under the base path
 * GitHub Pages uses for a project site (/StreamSense/), opens the page in
 * headless Chromium at several viewport sizes with every request to another
 * host blocked and counted, and checks what a visitor would meet: it loads,
 * nothing errors, nothing asks another host for anything, nothing is stored,
 * the in-page links land, the keyboard works, nothing overflows sideways, and
 * the numbers drawn on the page agree with each other.
 *
 * Usage:
 *   node scripts/verify_showcase.cjs [--out <dir>] [--docs <dir>]
 *   node scripts/verify_showcase.cjs --url https://<user>.github.io/StreamSense/ [--out <dir>]
 *
 * With --url the same checks run against a deployed copy instead of the local
 * file, and the check that compares the served bytes with docs/index.html in
 * this checkout then measures whether what is deployed is what is committed.
 *
 * Requires the `playwright` package and its Chromium:
 *   npm install --no-save playwright axe-core && npx playwright install chromium
 * axe-core is optional; without it the accessibility-rule check is SKIPPED,
 * not passed.
 *
 * Exit code: 0 when every check passes, 1 when any check fails or could not
 * run, 2 when the harness itself could not start. Without --url this checks
 * the page file only and says nothing about the deployed site. In either mode
 * it says nothing about the linked repositories or the cited papers, and it is
 * not a WCAG audit.
 */
'use strict';

const fs = require('fs');
const http = require('http');
const path = require('path');
const crypto = require('crypto');

const args = process.argv.slice(2);
function arg(name, fallback) {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
}
const DOCS = path.resolve(arg('--docs', path.join(__dirname, '..', 'docs')));
const OUT = arg('--out', '');
const REMOTE = arg('--url', '');
const BASE = '/StreamSense/';
const VIEWPORTS = [
  { name: 'desktop-1440', width: 1440, height: 900 },
  { name: 'tablet-820', width: 820, height: 1180 },
  { name: 'mobile-390', width: 390, height: 844 },
  { name: 'small-320', width: 320, height: 568 },
];
const MIME = {
  '.html': 'text/html; charset=utf-8', '.svg': 'image/svg+xml', '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8', '.json': 'application/json', '.png': 'image/png',
  '.md': 'text/markdown; charset=utf-8', '.ico': 'image/x-icon',
};
const sha256 = buf => crypto.createHash('sha256').update(buf).digest('hex');
const squash = s => String(s).replace(/\s+/g, '');
const oneLine = s => String(s).replace(/\s+/g, ' ').trim();

function startServer() {
  const server = http.createServer((req, res) => {
    let p;
    try { p = decodeURIComponent(new URL(req.url, 'http://x').pathname); } catch (e) { res.writeHead(400); return res.end(); }
    if (!p.startsWith(BASE)) { res.writeHead(404, { 'content-type': 'text/plain' }); return res.end('not found'); }
    let rel = p.slice(BASE.length);
    if (rel === '' || rel.endsWith('/')) rel += 'index.html';
    const file = path.resolve(DOCS, rel);
    if (file !== DOCS && !file.startsWith(DOCS + path.sep)) { res.writeHead(403); return res.end(); }
    fs.readFile(file, (err, buf) => {
      if (err) { res.writeHead(404, { 'content-type': 'text/plain' }); return res.end('not found'); }
      res.writeHead(200, { 'content-type': MIME[path.extname(file)] || 'application/octet-stream', 'content-length': buf.length });
      res.end(buf);
    });
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

async function main() {
  let chromium;
  try { ({ chromium } = require('playwright')); } catch (e) {
    console.error('playwright is not installed: npm install --no-save playwright axe-core && npx playwright install chromium');
    process.exit(2);
  }
  let axeSource = null;
  try { axeSource = fs.readFileSync(require.resolve('axe-core/axe.min.js'), 'utf8'); } catch (e) { /* optional */ }

  const indexBytes = fs.readFileSync(path.join(DOCS, 'index.html'));
  const server = REMOTE ? null : await startServer();
  const origin = REMOTE ? new URL(REMOTE).origin : `http://127.0.0.1:${server.address().port}`;
  const url = REMOTE || origin + BASE;
  const local = u => u.startsWith(origin + '/') || u.startsWith('data:') || u.startsWith('about:');

  const checks = [];
  const observations = [];
  const add = (id, description, pass, detail) => checks.push({ id, description, result: pass === null ? 'SKIPPED' : pass ? 'PASS' : 'FAIL', detail: detail === undefined ? '' : String(detail) });
  // A check that throws is a failed check, never a crashed run and never a silent pass.
  const group = async (name, fn) => {
    try { await fn(); } catch (e) { add(`${name}.could-not-run`, `the ${name} checks ran to the end`, false, oneLine(e && e.message ? e.message : e).slice(0, 400)); }
  };
  // Smooth scrolling takes a variable time: wait until the page stops moving.
  const settle = async page => {
    let last = -1, still = 0;
    for (let i = 0; i < 80 && still < 3; i++) {
      await page.waitForTimeout(50);
      const y = await page.evaluate(() => Math.round(scrollY));
      still = y === last ? still + 1 : 0;
      last = y;
    }
  };

  const browser = await chromium.launch();
  const version = browser.version();
  try {
    // ---------------------------------------------------------- each viewport
    for (const vp of VIEWPORTS) {
      await group(vp.name, async () => {
        const context = await browser.newContext({ viewport: { width: vp.width, height: vp.height } });
        try {
          const page = await context.newPage();
          const requests = [], blocked = [], consoleErrors = [], pageErrors = [], failed = [], badStatus = [];
          await page.route('**/*', route => {
            const u = route.request().url();
            if (local(u)) return route.continue();
            blocked.push(u);
            return route.abort();
          });
          page.on('request', r => requests.push(r.url()));
          page.on('requestfailed', r => { if (local(r.url())) failed.push(`${r.url()} ${r.failure() ? r.failure().errorText : ''}`); });
          page.on('response', r => { if (r.status() >= 400) badStatus.push(`${r.status()} ${r.url()}`); });
          page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()); });
          page.on('pageerror', e => pageErrors.push(String(e)));

          const resp = await page.goto(url, { waitUntil: 'load' });
          await page.waitForTimeout(300);
          add(`${vp.name}.load`, 'page answers 200 as text/html', resp.status() === 200 && /text\/html/.test(resp.headers()['content-type'] || ''), `status ${resp.status()}, ${resp.headers()['content-type']}`);
          if (vp === VIEWPORTS[0]) {
            const served = sha256(await resp.body());
            add('served.matches-file', 'the bytes served are the bytes of docs/index.html in this checkout', served === sha256(indexBytes), `served ${served}; file ${sha256(indexBytes)}`);
          }
          // A request this harness blocked shows up in the console as a load error; that belongs to the next check, not this one.
          const ownConsole = blocked.length ? consoleErrors.filter(t => !/Failed to load resource/i.test(t)) : consoleErrors;
          add(`${vp.name}.no-errors`, 'no console error, page error, failed request or 4xx/5xx response', ownConsole.length + pageErrors.length + failed.length + badStatus.length === 0, JSON.stringify({ consoleErrors: ownConsole, pageErrors, failed, badStatus }));
          add(`${vp.name}.same-origin-only`, 'the page asks no other host for anything', blocked.length === 0 && requests.every(local), `${requests.length} request(s): ${JSON.stringify(requests.map(u => u.replace(origin, '')))}; to other hosts (blocked): ${JSON.stringify(blocked)}`);

          const measure = () => {
            const de = document.documentElement;
            const wide = [];
            for (const el of document.querySelectorAll('body *')) {
              if (el.classList.contains('skip')) continue; // parked off-screen until focused, by design
              const r = el.getBoundingClientRect();
              if (r.width > 0 && (r.right > de.clientWidth + 1 || r.left < -1)) wide.push(`${el.tagName.toLowerCase()}${el.id ? '#' + el.id : ''} [${Math.round(r.left)},${Math.round(r.right)}]`);
            }
            return { scrollWidth: de.scrollWidth, clientWidth: de.clientWidth, wide: wide.slice(0, 12) };
          };
          const closed = await page.evaluate(measure);
          add(`${vp.name}.no-sideways-overflow`, 'no horizontal page overflow and no element past the viewport edge', closed.scrollWidth <= closed.clientWidth + 1 && closed.wide.length === 0, `scrollWidth ${closed.scrollWidth}, clientWidth ${closed.clientWidth}, past the edge: ${JSON.stringify(closed.wide)}`);
          // Open every disclosure and look again: the long citation block lives inside one.
          await page.evaluate(() => document.querySelectorAll('details').forEach(d => { d.open = true; }));
          const opened = await page.evaluate(measure);
          add(`${vp.name}.no-sideways-overflow-open`, 'still no horizontal overflow with every disclosure opened', opened.scrollWidth <= opened.clientWidth + 1 && opened.wide.length === 0, `scrollWidth ${opened.scrollWidth}, clientWidth ${opened.clientWidth}, past the edge: ${JSON.stringify(opened.wide)}`);
          if (OUT) {
            fs.mkdirSync(OUT, { recursive: true });
            await page.screenshot({ path: path.join(OUT, `showcase-${vp.name}-open.png`), fullPage: true });
          }
        } finally {
          await context.close();
        }
      });
    }

    // ------------------------------------------------- once, at desktop width
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await context.newPage();
    await page.route('**/*', route => (local(route.request().url()) ? route.continue() : route.abort()));
    await page.goto(url, { waitUntil: 'load' });

    let dom = null;
    await group('doc', async () => {
      dom = await page.evaluate(() => {
        const q = s => Array.from(document.querySelectorAll(s));
        const ids = q('[id]').map(e => e.id);
        return {
          title: document.title, lang: document.documentElement.lang,
          h1: q('h1').map(h => h.innerText.replace(/\s+/g, ' ').trim()),
          headings: q('h1,h2,h3,h4,h5,h6').map(h => ({ level: Number(h.tagName[1]), text: h.textContent.replace(/\s+/g, ' ').trim() })),
          anchors: q('a[href^="#"]').map(a => ({ href: a.getAttribute('href'), text: a.textContent.trim(), ok: !!document.getElementById(a.getAttribute('href').slice(1)) })),
          duplicateIds: ids.filter((id, i) => ids.indexOf(id) !== i),
          counts: { script: q('script').length, img: q('img,picture,svg,canvas').length, frame: q('iframe,frame,object,embed').length, media: q('video,audio').length, form: q('form,input,textarea,select,button').length, link: q('link').length, base: q('base').length, metaRefresh: q('meta[http-equiv="refresh" i]').length },
          cssLoads: /@import|url\(/i.test(q('style').map(s => s.textContent).join('\n') + q('[style]').map(e => e.getAttribute('style')).join('\n')),
          inlineHandlers: q('*').filter(e => Array.from(e.attributes).some(a => /^on/i.test(a.name))).length,
          cookie: document.cookie, localKeys: localStorage.length, sessionKeys: sessionStorage.length,
          bars: q('.bar[role="img"]').map(b => ({ spoken: b.getAttribute('aria-label') || '', drawn: b.querySelector('i') ? b.querySelector('i').style.width : '', shown: b.previousElementSibling ? b.previousElementSibling.textContent.replace(/\s+/g, ' ').trim() : '' })),
          bigNumber: (document.querySelector('.big-number') || { textContent: '' }).textContent.replace(/\s+/g, ''),
          outside: q('a[href^="http"]').map(a => ({ href: a.href, target: a.target, rel: a.rel })),
          text: document.body.innerText,
          details: q('details').map(d => ({ id: d.id, open: d.open, summary: d.querySelector('summary') ? d.querySelector('summary').textContent.trim() : '' })),
          landmarks: { main: q('main').length, nav: q('nav').map(n => n.getAttribute('aria-label')), header: q('header').length, footer: q('footer').length },
        };
      });
      const sw = await page.evaluate(async () => ('serviceWorker' in navigator ? (await navigator.serviceWorker.getRegistrations()).length : 0));
      const cookies = await context.cookies();

      add('doc.title-and-lang', 'title and document language are set', dom.title.length > 0 && dom.lang === 'en', `title "${dom.title}", lang "${dom.lang}"`);
      add('doc.one-h1', 'exactly one h1', dom.h1.length === 1, JSON.stringify(dom.h1));
      const skippedLevels = [];
      for (let i = 1; i < dom.headings.length; i++) if (dom.headings[i].level > dom.headings[i - 1].level + 1) skippedLevels.push(`h${dom.headings[i - 1].level} "${dom.headings[i - 1].text}" then h${dom.headings[i].level} "${dom.headings[i].text}"`);
      add('doc.heading-order', 'no heading level is skipped', skippedLevels.length === 0, JSON.stringify(skippedLevels));
      add('doc.landmarks', 'one main, a labelled nav, a header and a footer', dom.landmarks.main === 1 && dom.landmarks.nav.length >= 1 && dom.landmarks.nav.every(Boolean) && dom.landmarks.header >= 1 && dom.landmarks.footer >= 1, JSON.stringify(dom.landmarks));
      add('doc.unique-ids', 'no duplicate id', dom.duplicateIds.length === 0, JSON.stringify(dom.duplicateIds));
      const broken = dom.anchors.filter(a => !a.ok);
      add('links.in-page-targets', 'every in-page link has a target on the page', dom.anchors.length > 0 && broken.length === 0, `${dom.anchors.length} in-page link(s); broken: ${JSON.stringify(broken)}`);
      add('privacy.nothing-embedded', 'no script, image, frame, media, form control, stylesheet link, CSS that loads something, or inline event handler', Object.values(dom.counts).every(n => n === 0) && !dom.cssLoads && dom.inlineHandlers === 0, JSON.stringify({ ...dom.counts, cssLoads: dom.cssLoads, inlineHandlers: dom.inlineHandlers }));
      add('privacy.nothing-stored', 'no cookie, no local or session storage, no service worker', cookies.length === 0 && dom.cookie === '' && dom.localKeys === 0 && dom.sessionKeys === 0 && sw === 0, JSON.stringify({ cookies: cookies.length, localKeys: dom.localKeys, sessionKeys: dom.sessionKeys, serviceWorkers: sw }));
      const insecure = dom.outside.filter(a => !a.href.startsWith('https://'));
      const unsafeBlank = dom.outside.filter(a => a.target === '_blank' && !/noopener/.test(a.rel));
      add('links.outside-https', 'every outside link is https and none opens a new tab without rel=noopener', dom.outside.length > 0 && insecure.length === 0 && unsafeBlank.length === 0, `${dom.outside.length} outside link(s) to ${new Set(dom.outside.map(a => new URL(a.href).host)).size} host(s); not https: ${JSON.stringify(insecure.map(a => a.href))}; new tab without noopener: ${JSON.stringify(unsafeBlank.map(a => a.href))}`);
      observations.push(`outside links (${dom.outside.length}): ${JSON.stringify(dom.outside.map(a => a.href))}`);
      observations.push(`in-page links (${dom.anchors.length}): ${JSON.stringify(dom.anchors.map(a => a.href))}`);
      observations.push(`disclosures and whether each starts open: ${JSON.stringify(dom.details.map(d => ({ summary: d.summary, open: d.open })))}`);
    });

    // The numbers drawn on the page must agree with each other.
    await group('numbers', async () => {
      const num = (s, re) => (String(s).match(re) || [])[1];
      const rows = dom.bars.map(b => ({ spoken: num(b.spoken, /([\d.]+)\s*percent/), drawn: num(b.drawn, /([\d.]+)%/), shown: num(b.shown, /([\d.]+)%/) }));
      add('numbers.bars-agree', 'each bar\'s label, drawn width and accessible name carry the same number', rows.length === 2 && rows.every(r => r.spoken && r.spoken === r.drawn && r.spoken === r.shown), JSON.stringify(dom.bars));
      const model = parseFloat(rows[0] && rows[0].shown), baseline = parseFloat(rows[1] && rows[1].shown);
      const stated = parseFloat(num(dom.text, /([\d.]+) percentage points/));
      add('numbers.difference', 'the stated gap equals model minus baseline', Number.isFinite(stated) && Math.round((model - baseline) * 10) === Math.round(stated * 10), `${model} - ${baseline} = ${(model - baseline).toFixed(1)}; page says ${stated}`);
      add('numbers.headline', 'the large figure is the model bar\'s number', parseFloat(dom.bigNumber) === model, `large figure ${dom.bigNumber}; model bar ${model}`);
    });

    // Links that point at a disclosure, and the navigation links.
    await group('links', async () => {
      const intoDetails = dom.anchors.filter(a => dom.details.some(d => d.id && '#' + d.id === a.href));
      const landing = [];
      for (const a of intoDetails) {
        await page.goto(url, { waitUntil: 'load' });
        await page.locator(`a[href="${a.href}"]`).nth(landing.filter(l => l.href === a.href).length).click();
        await settle(page);
        landing.push(await page.evaluate(h => { const d = document.querySelector(h); const r = d.getBoundingClientRect(); return { href: h, open: d.open, inView: r.top >= 0 && r.top < innerHeight }; }, a.href));
      }
      add('links.evidence-notes-come-into-view', 'a link to the evidence notes brings their disclosure into view', landing.length > 0 && landing.every(l => l.inView), `${intoDetails.length} link(s) point at a disclosure: ${JSON.stringify(landing)}`);
      if (landing.some(l => !l.open)) observations.push('a link to the evidence notes lands on their disclosure while it is still closed: one more activation shows the notes');

      const navHrefs = await page.evaluate(() => Array.from(document.querySelectorAll('nav a[href^="#"]')).map(a => a.getAttribute('href')));
      const navResults = [];
      for (const href of navHrefs) {
        await page.goto(url, { waitUntil: 'load' });
        await page.locator(`nav a[href="${href}"]`).first().click();
        await settle(page);
        navResults.push(await page.evaluate(h => {
          const target = document.getElementById(h.slice(1));
          if (!target) return { href: h, ok: false, hash: location.hash, note: 'no target' };
          const r = target.getBoundingClientRect();
          return { href: h, top: Math.round(r.top), ok: r.top >= -2 && r.top < innerHeight / 2, hash: location.hash };
        }, href));
      }
      add('links.navigation-lands', 'each navigation link brings its section to the top half of the window', navResults.length > 0 && navResults.every(r => r.ok && r.hash === r.href), JSON.stringify(navResults));
    });

    await group('keyboard', async () => {
      // The skip link comes first, shows itself when focused, and moves focus to main.
      await page.goto(url, { waitUntil: 'load' });
      await page.keyboard.press('Tab');
      const skip = await page.evaluate(() => { const e = document.activeElement; const r = e.getBoundingClientRect(); return { cls: e.className, text: e.textContent.trim(), top: Math.round(r.top), visible: r.top >= 0 && r.bottom <= innerHeight && r.width > 0 }; });
      await page.keyboard.press('Enter');
      await settle(page);
      const afterSkip = await page.evaluate(() => ({ hash: location.hash, active: document.activeElement.tagName.toLowerCase() + (document.activeElement.id ? '#' + document.activeElement.id : '') }));
      add('keyboard.skip-link', 'first Tab reaches a visible skip link and Enter moves focus to main', skip.cls === 'skip' && skip.visible && afterSkip.hash === '#main' && afterSkip.active === 'main#main', JSON.stringify({ skip, afterSkip }));

      // Every link and summary is reachable and shows a focus ring. A link inside a
      // closed disclosure cannot take focus, so the second walk opens them all first.
      const walk = async openAll => {
        await page.goto(url, { waitUntil: 'load' });
        if (openAll) await page.evaluate(() => document.querySelectorAll('details').forEach(d => { d.open = true; }));
        const expected = await page.evaluate(() => Array.from(document.querySelectorAll('a[href], summary')).filter(e => {
          if (e.tabIndex < 0) return true; // counted, so that taking a link out of the tab order is noticed
          const closed = e.parentElement && e.parentElement.closest('details:not([open])');
          return !(closed && !(e.tagName === 'SUMMARY' && e.parentElement === closed));
        }).length);
        const seen = [], noRing = [], stray = [];
        for (let i = 0; i < expected + 8; i++) {
          await page.keyboard.press('Tab');
          const f = await page.evaluate(() => {
            const e = document.activeElement;
            if (!e || e === document.body) return null;
            const cs = getComputedStyle(e);
            return { key: e.tagName + '|' + (e.getAttribute('href') || '') + '|' + e.textContent.trim().slice(0, 40) + '|' + Math.round(e.getBoundingClientRect().top + scrollY), ring: cs.outlineStyle !== 'none' && parseFloat(cs.outlineWidth) >= 2, expected: e.matches('a[href], summary') };
          });
          if (!f || seen.includes(f.key)) break;
          seen.push(f.key);
          if (!f.expected) stray.push(f.key);
          if (!f.ring) noRing.push(f.key);
        }
        return { expected, reached: seen.length - stray.length, noRing, stray };
      };
      const closedWalk = await walk(false);
      add('keyboard.reach-and-focus-ring', 'Tab reaches every link and summary and nothing else, each with a visible focus ring', closedWalk.reached === closedWalk.expected && closedWalk.noRing.length === 0 && closedWalk.stray.length === 0, `reached ${closedWalk.reached} of ${closedWalk.expected}; without a ring: ${JSON.stringify(closedWalk.noRing)}; other focus stops: ${JSON.stringify(closedWalk.stray)}`);
      const openWalk = await walk(true);
      add('keyboard.reach-and-focus-ring-open', 'the same with every disclosure opened, so the links inside are reachable too', openWalk.reached === openWalk.expected && openWalk.expected >= closedWalk.expected && openWalk.noRing.length === 0 && openWalk.stray.length === 0, `reached ${openWalk.reached} of ${openWalk.expected}; without a ring: ${JSON.stringify(openWalk.noRing)}; other focus stops: ${JSON.stringify(openWalk.stray)}`);

      // A disclosure opens and closes from its summary.
      await page.goto(url, { waitUntil: 'load' });
      const count = await page.locator('details > summary').count();
      const toggled = [];
      for (let i = 0; i < count; i++) {
        const before = await page.evaluate(n => document.querySelectorAll('details')[n].open, i);
        await page.locator('details > summary').nth(i).focus();
        await page.keyboard.press('Enter');
        const after = await page.evaluate(n => document.querySelectorAll('details')[n].open, i);
        toggled.push(before !== after);
      }
      add('keyboard.disclosure-toggles', 'Enter on each summary opens or closes its disclosure', count > 0 && toggled.every(Boolean), `${count} disclosure(s): ${JSON.stringify(toggled)}`);
    });

    await group('motion', async () => {
      const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
      try {
        const p = await ctx.newPage();
        await p.goto(url, { waitUntil: 'load' });
        const behaviour = await p.evaluate(() => getComputedStyle(document.documentElement).scrollBehavior);
        add('motion.reduced', 'smooth scrolling is off when the visitor asks for reduced motion', behaviour === 'auto', `scroll-behavior: ${behaviour}`);
      } finally { await ctx.close(); }
    });

    await group('nojs', async () => {
      const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, javaScriptEnabled: false });
      try {
        const p = await ctx.newPage();
        await p.goto(url, { waitUntil: 'load' });
        const heading = await p.locator('h1').first().innerText();
        const visibleLinks = await p.locator('a[href]:visible').count();
        add('nojs.renders', 'the page renders with JavaScript disabled', squash(heading).length > 0 && squash(heading) === squash(dom.h1[0]) && visibleLinks > 0, `h1 "${oneLine(heading)}", ${visibleLinks} visible link(s)`);
      } finally { await ctx.close(); }
    });

    // axe-core, when installed: WCAG 2.x A and AA rules plus best practices, with the disclosures opened.
    if (axeSource) {
      await group('a11y', async () => {
        await page.goto(url, { waitUntil: 'load' });
        await page.evaluate(() => document.querySelectorAll('details').forEach(d => { d.open = true; }));
        await page.evaluate(axeSource);
        const axe = await page.evaluate(async () => {
          const r = await window.axe.run(document, { runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa', 'best-practice'] } });
          return { version: window.axe.version, passes: r.passes.length, incomplete: r.incomplete.map(v => `${v.id} (${v.nodes.length})`), violations: r.violations.map(v => ({ id: v.id, impact: v.impact, help: v.help, nodes: v.nodes.map(n => ({ target: n.target.join(' '), summary: (n.failureSummary || '').replace(/\s+/g, ' ').slice(0, 300) })) })) };
        });
        add('a11y.axe', 'axe-core finds no WCAG A/AA or best-practice violation (disclosures opened)', axe.violations.length === 0, `axe ${axe.version}: ${axe.passes} rule(s) passed; violations: ${JSON.stringify(axe.violations)}; left for a human: ${JSON.stringify(axe.incomplete)}`);
      });
    } else {
      add('a11y.axe', 'axe-core accessibility rules', null, 'axe-core is not installed, so the rule check did not run');
    }
    await context.close();
  } finally {
    await browser.close();
    if (server) server.close();
  }

  const count = result => checks.filter(c => c.result === result).length;
  const report = {
    subject: 'docs/index.html',
    sha256: sha256(indexBytes),
    bytes: indexBytes.length,
    served_at: REMOTE ? `deployed copy at ${REMOTE}` : `loopback static server, base path ${BASE}`,
    browser: `Chromium ${version} (headless, Playwright)`,
    network: `every request to a host other than ${REMOTE ? new URL(REMOTE).host : 'the loopback server'} is blocked and counted`,
    viewports: VIEWPORTS.map(v => `${v.name} ${v.width}x${v.height}`),
    summary: { total: checks.length, passed: count('PASS'), failed: count('FAIL'), skipped: count('SKIPPED') },
    checks,
    observations,
    scope: (REMOTE ? 'Check of the deployed copy named above.' : 'Local check of the page file, not of the deployed site.') + ' Not a check of the linked repositories, of the cited publications, or of full WCAG conformance.',
  };
  if (OUT) {
    fs.mkdirSync(OUT, { recursive: true });
    fs.writeFileSync(path.join(OUT, 'showcase-verification.json'), JSON.stringify(report, null, 2) + '\n');
  }
  for (const c of checks) console.log(`${c.result.padEnd(7)} ${c.id.padEnd(42)} ${c.description}${c.result === 'PASS' ? '' : '\n        ' + c.detail}`);
  console.log(`\n${report.summary.passed} passed, ${report.summary.failed} failed, ${report.summary.skipped} skipped  (page sha256 ${report.sha256.slice(0, 16)}…, ${report.browser})`);
  process.exit(report.summary.failed ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(2); });
