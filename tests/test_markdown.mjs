import assert from 'node:assert/strict';
import test from 'node:test';
import {inline, parse} from '../frontend/markdown.js';

test('inline tokens: code, strong, em, wiki, http links only', () => {
  assert.deepEqual(inline('a `x` **b** *c* [[Note|Label]] [d](https://e.org) [f](javascript:alert(1))'), [
    {t: 'text', v: 'a '}, {t: 'code', v: 'x'}, {t: 'text', v: ' '}, {t: 'strong', v: 'b'}, {t: 'text', v: ' '},
    {t: 'em', v: 'c'}, {t: 'text', v: ' '}, {t: 'wiki', target: 'Note', v: 'Label'}, {t: 'text', v: ' '},
    {t: 'link', href: 'https://e.org', v: 'd'}, {t: 'text', v: ' [f](javascript:alert(1))'},
  ]);
  assert.deepEqual(inline('2 * 3 * 4'), [{t: 'text', v: '2 * 3 * 4'}]);
});

test('blocks: headings, lists, paragraphs and fences', () => {
  const blocks = parse('# T\n\none\ntwo\n\n- a\n- b\n1. c\n\n```\n[[not a link]]\n```\n<script>x</script>');
  assert.deepEqual(blocks.map(b => b.type), ['h', 'p', 'ul', 'ol', 'code', 'p']);
  assert.equal(blocks[1].inline[0].v, 'one two');
  assert.equal(blocks[2].items.length, 2);
  assert.equal(blocks[4].text, '[[not a link]]');
  assert.deepEqual(blocks[5].inline, [{t: 'text', v: '<script>x</script>'}]);
  assert.equal(blocks[3].start, 1);
  assert.deepEqual(parse('1. a\n\nтекст\n\n2. b').map(b => b.start), [1, undefined, 2]);
});
