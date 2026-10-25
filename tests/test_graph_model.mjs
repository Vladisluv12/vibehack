import assert from 'node:assert/strict';
import test from 'node:test';
import {adjacency, communities, degrees, importanceTiers, localSubgraph, withPositions} from '../frontend/graph-model.js';

const graph = {
  nodes: ['a', 'b', 'c', 'd', 'e'].map(id => ({id, title: id.toUpperCase()})),
  links: [
    {source: 'a', target: 'b', relation: 'relates_to'},
    {source: 'c', target: 'b', relation: 'relates_to'},
    {source: 'c', target: 'd', relation: 'derived_from'},
    {source: 'a', target: 'a', relation: 'relates_to'},
  ],
};

test('degrees are undirected and count self-links once', () => {
  assert.deepEqual(Object.fromEntries(degrees(graph.links)), {a: 2, b: 2, c: 2, d: 1});
});

test('adjacency accepts force-graph object ends', () => {
  const objectLinks = graph.links.map(l => ({source: {id: l.source}, target: {id: l.target}}));
  assert.deepEqual([...adjacency(objectLinks).get('b')].sort(), ['a', 'c']);
});

test('local subgraph follows links both ways up to depth', () => {
  const one = localSubgraph(graph, 'b', 1);
  assert.deepEqual(one.nodes.map(n => [n.id, n.hops]), [['b', 0], ['a', 1], ['c', 1]]);
  assert.equal(one.links.length, 3);
  const two = localSubgraph(graph, 'b', 2);
  assert.deepEqual(two.nodes.map(n => n.id).sort(), ['a', 'b', 'c', 'd']);
  assert.ok(!two.nodes.some(n => n.id === 'e'));
  assert.deepEqual(localSubgraph(graph, 'missing', 2), {nodes: [], links: []});
  assert.deepEqual(localSubgraph(graph, 'e', 2).nodes.map(n => n.id), ['e']);
});

test('withPositions copies data, keeps layout and adds degree', () => {
  const first = withPositions(graph);
  first.nodes[0].x = 10; first.nodes[0].y = 20;
  first.links[0].source = first.nodes[0];
  const second = withPositions(graph, first.nodes);
  assert.deepEqual([second.nodes[0].x, second.nodes[0].y], [10, 20]);
  assert.equal(second.nodes[1].x, undefined);
  assert.equal(second.nodes[0].degree, 2);
  assert.equal(second.links[0].source, 'a');
  assert.equal(graph.nodes[0].x, undefined);
});

test('communities seed from projects and absorb unowned neighbours', () => {
  const nodes = [
    {id: 'p1', project_id: 'p1'}, {id: 'a', project_id: 'p1'}, {id: 'b', project_id: 'p1'},
    {id: 'p2', project_id: 'p2'}, {id: 'c', project_id: 'p2'}, {id: 'loose'}, {id: 'alone'},
  ];
  const links = [['a', 'p1'], ['b', 'p1'], ['c', 'p2'], ['loose', 'a'], ['loose', 'b'], ['loose', 'c']].map(([source, target]) => ({source, target}));
  assert.deepEqual(communities(nodes, links), [{key: 'p1', ids: ['p1', 'a', 'b', 'loose']}, {key: 'p2', ids: ['p2', 'c']}, {key: 'alone', ids: ['alone']}]);
  assert.deepEqual(communities([...nodes].reverse(), links).map(g => g.key), ['p1', 'p2', 'alone']);
});

test('importance tiers rank projects, then degree', () => {
  const nodes = Array.from({length: 10}, (_, i) => ({id: 'n' + i, degree: i}));
  nodes.push({id: 'proj', kind: 'project', degree: 0});
  const tiers = importanceTiers(nodes);
  assert.equal(tiers.get('proj'), 0);
  assert.equal(tiers.get('n9'), 1);
  assert.equal(tiers.get('n0'), 2);
});
