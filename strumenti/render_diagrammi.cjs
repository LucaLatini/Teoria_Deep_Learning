// Render the original Mermaid source with a locally installed Mermaid bundle.
// No remote resources are loaded. Pass: job.json, mermaid.min.js, playwright module.
const fs = require('fs');
const path = require('path');
const { chromium } = require(process.argv[4]);
(async () => {
  const jobs = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const browser = await chromium.launch({
    executablePath: process.env.DISPENSA_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true,
    args: ['--disable-gpu', '--disable-dev-shm-usage'],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1800, height: 1400 }, deviceScaleFactor: 2 });
    await page.setContent('<html><head><style>html,body{margin:0;padding:0;background:white;}svg{display:block;max-width:none!important;}</style></head><body></body></html>');
    await page.addScriptTag({ path: process.argv[3] });
    await page.evaluate(() => mermaid.initialize({
      startOnLoad: false, securityLevel: 'loose', theme: 'base',
      themeVariables: { fontFamily: 'Arial', fontSize: '16px', primaryColor: '#eef4f6', primaryTextColor: '#142e3c', primaryBorderColor: '#648896', lineColor: '#50717f', secondaryColor: '#f8f4ea', tertiaryColor: '#f2f5f6' },
      flowchart: { htmlLabels: false, curve: 'linear', nodeSpacing: 28, rankSpacing: 35, padding: 12, useMaxWidth: false },
    }));
    for (const job of jobs) {
      const result = await page.evaluate(async ({ source, id }) => {
        document.body.innerHTML = '';
        let renderedSource = source;
        let { svg } = await mermaid.render(id, renderedSource);
        // Reorient wide chains for an A4 page. Nodes, labels and edges are unchanged.
        const probe = document.createElement('div');
        probe.innerHTML = svg;
        const probeView = probe.querySelector('svg').viewBox.baseVal;
        if (probeView.width > 900 && probeView.width / probeView.height > 3 && /^(flowchart|graph) LR/m.test(source)) {
          renderedSource = source.replace(/^(flowchart|graph) LR/m, '$1 TD');
          ({ svg } = await mermaid.render(id + 'verticale', renderedSource));
        }
        document.body.innerHTML = svg;
        const el = document.querySelector('svg');
        const v = el.viewBox.baseVal;
        el.setAttribute('width', v.width);
        el.setAttribute('height', v.height);
        el.style.width = v.width + 'px';
        el.style.height = v.height + 'px';
        return { svg: el.outerHTML, width: v.width, height: v.height, text: el.textContent, renderedSource };
      }, { source: job.source, id: job.id });
      fs.writeFileSync(job.svg, result.svg);
      await page.pdf({ path: job.pdf, width: `${result.width}px`, height: `${result.height}px`, printBackground: true, margin: { top: '0', bottom: '0', left: '0', right: '0' }, pageRanges: '1' });
      fs.writeFileSync(job.pdf + '.json', JSON.stringify(result));
      console.log(JSON.stringify({ id: job.id, width: result.width, height: result.height }));
    }
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
