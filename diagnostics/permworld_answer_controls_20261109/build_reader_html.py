"""Render this report's limited Markdown subset as standalone offline HTML."""
from pathlib import Path
import html
import re

HERE=Path(__file__).resolve().parent


def inline(text):
    text=html.escape(text)
    text=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',r'<a href="\2"><img src="\2" alt="\1" loading="lazy"></a>',text)
    text=re.sub(r'\[([^\]]*)\]\(([^)]+)\)',r'<a href="\2">\1</a>',text)
    text=re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',text)
    return re.sub(r'`([^`]+)`',r'<code>\1</code>',text)


def render(lines):
    out=[];paragraph=[];i=0
    def flush():
        if paragraph:out.append('<p>'+inline(' '.join(paragraph))+'</p>');paragraph.clear()
    while i<len(lines):
        line=lines[i]
        if not line.strip():flush();i+=1;continue
        if line.startswith('#'):
            flush();n=len(line)-len(line.lstrip('#'));out.append(f'<h{n}>'+inline(line[n:].strip())+f'</h{n}>');i+=1;continue
        if line.startswith('|'):
            flush();table=[]
            while i<len(lines) and lines[i].startswith('|'):
                table.append([x.strip() for x in lines[i].strip().strip('|').split('|')]);i+=1
            out.append('<div class="table"><table><thead><tr>'+''.join('<th>'+inline(x)+'</th>' for x in table[0])+'</tr></thead><tbody>')
            for row in table[2:]:out.append('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in row)+'</tr>')
            out.append('</tbody></table></div>');continue
        if line.startswith('- '):
            flush();out.append('<ul>')
            while i<len(lines) and lines[i].startswith('- '):out.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
            out.append('</ul>');continue
        paragraph.append(line);i+=1
    flush();return '\n'.join(out)


body=render((HERE/'READER_REPORT.md').read_text().splitlines())
style='''body{font:17px/1.6 system-ui,sans-serif;color:#16263a;background:#fafbfe;margin:0}main{max-width:1040px;margin:0 auto;padding:40px 26px 80px;background:white}h1{font-size:2.2rem;line-height:1.15}h2{margin-top:2.3em;line-height:1.25;color:#143a68}a{color:#105ca8}p,li{max-width:95ch}li{margin:.5em 0}code{font-size:.85em;background:#eef2f7;padding:2px 4px;overflow-wrap:anywhere}img{display:block;width:100%;height:auto;margin:22px 0;border:1px solid #dce3ec}.table{overflow-x:auto;margin:24px 0}table{border-collapse:collapse;font-size:.92em;width:100%}th,td{padding:10px 12px;border-bottom:1px solid #dce3ec;text-align:left}th{background:#edf3fa}tr:nth-child(even){background:#f8fafc}nav{font-size:.9em;color:#526477;border-bottom:1px solid #ddd;padding-bottom:15px}@media print{main{padding:0}img{max-height:90vh;object-fit:contain}h2{break-after:avoid}.table{overflow:visible}a{color:inherit}}'''
(HERE/'reader_report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PermWorld length and answer controls</title><style>'+style+'</style><main><nav>Exploratory existing-checkpoint diagnostic · No new training · <a href="READER_REPORT.md">Markdown report</a> · <a href="README.md">Reproduce</a></nav>'+body+'</main></html>\n')
print('Wrote offline reader_report.html; no scientific tables modified.')
