/**
 * Small Markdown subset → plain block/inline tokens. No HTML is ever produced
 * here; the caller builds DOM nodes from tokens, so note text cannot inject markup.
 */
const INLINE = /(`[^`\n]+`)|\*\*([^*\n]+)\*\*|(?<![\w*])\*([^*\s][^*\n]*?)\*(?![\w*])|\[\[([^\]\n]+)\]\]|\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g;

export function inline(text) {
  const out = [];
  let last = 0;
  for (const m of text.matchAll(INLINE)) {
    if (m.index > last) out.push({t: 'text', v: text.slice(last, m.index)});
    if (m[1]) out.push({t: 'code', v: m[1].slice(1, -1)});
    else if (m[2]) out.push({t: 'strong', v: m[2]});
    else if (m[3]) out.push({t: 'em', v: m[3]});
    else if (m[4]) {
      const [target, label] = m[4].split('|');
      out.push({t: 'wiki', target: target.split('#')[0].trim(), v: (label || target).trim()});
    } else out.push({t: 'link', href: m[6], v: m[5]});
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({t: 'text', v: text.slice(last)});
  return out;
}

export function parse(markdown) {
  const blocks = [];
  let paragraph = [], list = null, fence = null;
  const flush = () => {
    if (paragraph.length) blocks.push({type: 'p', inline: inline(paragraph.join(' '))});
    paragraph = [];
    if (list) blocks.push(list);
    list = null;
  };
  for (const line of markdown.replace(/\r\n/g, '\n').split('\n')) {
    if (fence) {
      if (/^\s{0,3}(`{3,}|~{3,})\s*$/.test(line) && line.trim()[0] === fence.marker) { blocks.push({type: 'code', text: fence.lines.join('\n')}); fence = null; }
      else fence.lines.push(line);
      continue;
    }
    const open = line.match(/^\s{0,3}(`{3,}|~{3,})/);
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    const item = line.match(/^\s*(?:[-*+]|(\d+)[.)])\s+(.*)$/);
    if (open) { flush(); fence = {marker: open[1][0], lines: []}; }
    else if (heading) { flush(); blocks.push({type: 'h', level: heading[1].length, inline: inline(heading[2])}); }
    else if (item) {
      if (paragraph.length) flush();
      const type = item[1] ? 'ol' : 'ul';
      if (list && list.type !== type) flush();
      list = list || (type === 'ol' ? {type, start: Number(item[1]), items: []} : {type, items: []});
      list.items.push(inline(item[2]));
    } else if (!line.trim()) flush();
    else { if (list) flush(); paragraph.push(line.trim()); }
  }
  if (fence) blocks.push({type: 'code', text: fence.lines.join('\n')});
  flush();
  return blocks;
}
