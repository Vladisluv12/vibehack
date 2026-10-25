/**
 * Pure graph helpers for the map: no DOM, no rendering, safe to unit-test in Node.
 * Links arrive as {source, target, relation} with string IDs; force-graph later
 * replaces the ends with node objects, so every helper accepts both forms.
 */
export const endId = end => (typeof end === 'object' && end !== null ? end.id : end);

/** Undirected degree per node ID. Self-links count once. */
export function degrees(links) {
  const result = new Map();
  for (const link of links) {
    const a = endId(link.source), b = endId(link.target);
    result.set(a, (result.get(a) || 0) + 1);
    if (b !== a) result.set(b, (result.get(b) || 0) + 1);
  }
  return result;
}

/** Map of node ID → Set of directly linked node IDs. */
export function adjacency(links) {
  const result = new Map();
  const add = (a, b) => { if (!result.has(a)) result.set(a, new Set()); result.get(a).add(b); };
  for (const link of links) {
    const a = endId(link.source), b = endId(link.target);
    add(a, b); add(b, a);
  }
  return result;
}

/**
 * Nodes within `depth` hops of `centerId`, following links in both directions,
 * and only the links between those nodes. Each node gets `hops` from the center.
 */
export function localSubgraph({nodes, links}, centerId, depth = 1) {
  const byId = new Map(nodes.map(n => [n.id, n]));
  if (!byId.has(centerId)) return {nodes: [], links: []};
  const neighbours = adjacency(links);
  const hops = new Map([[centerId, 0]]);
  let frontier = [centerId];
  for (let level = 1; level <= depth && frontier.length; level++) {
    const next = [];
    for (const id of frontier) {
      for (const other of neighbours.get(id) || []) {
        if (!hops.has(other) && byId.has(other)) { hops.set(other, level); next.push(other); }
      }
    }
    frontier = next;
  }
  return {
    nodes: [...hops].map(([id, h]) => ({...byId.get(id), hops: h})),
    links: links.filter(l => hops.has(endId(l.source)) && hops.has(endId(l.target))),
  };
}

/**
 * Fresh node/link objects for force-graph (it mutates what it receives), keeping
 * the layout position of nodes that were already on screen.
 */
export function withPositions({nodes, links}, previous = []) {
  const old = new Map(previous.map(n => [n.id, n]));
  const degree = degrees(links);
  return {
    nodes: nodes.map(n => {
      const prior = old.get(n.id);
      return {...n, degree: degree.get(n.id) || 0, ...(prior && Number.isFinite(prior.x) ? {x: prior.x, y: prior.y} : {})};
    }),
    links: links.map(l => ({...l, source: endId(l.source), target: endId(l.target)})),
  };
}

/**
 * Communities by label propagation. Notes with an explicit `project_id` keep it
 * as a fixed label; the rest adopt their neighbours' majority label. Ties keep
 * the current label, then prefer the smallest one, so the result is
 * deterministic. Returns groups sorted by size: [{key, ids}].
 */
export function communities(nodes, links, maxRounds = 30) {
  const neighbours = adjacency(links);
  const label = new Map(nodes.map(n => [n.id, n.project_id || n.id]));
  const fixed = new Set(nodes.filter(n => n.project_id).map(n => n.id));
  const order = nodes.map(n => n.id).filter(id => !fixed.has(id)).sort();
  for (let round = 0; round < maxRounds; round++) {
    let changed = false;
    for (const id of order) {
      const counts = new Map();
      for (const other of neighbours.get(id) || []) {
        if (label.has(other) && other !== id) counts.set(label.get(other), (counts.get(label.get(other)) || 0) + 1);
      }
      const current = label.get(id);
      let best = current, bestCount = counts.get(current) || 0;
      for (const [key, count] of [...counts].sort(([a], [b]) => (a < b ? -1 : 1))) {
        if (count > bestCount) { best = key; bestCount = count; }
      }
      if (best !== current) { label.set(id, best); changed = true; }
    }
    if (!changed) break;
  }
  const groups = new Map();
  for (const [id, key] of label) { if (!groups.has(key)) groups.set(key, []); groups.get(key).push(id); }
  return [...groups].map(([key, ids]) => ({key, ids})).sort((a, b) => b.ids.length - a.ids.length || (a.key < b.key ? -1 : 1));
}

/** 0 = hub (top ~10%), 1 = notable (next ~30%), 2 = the rest; by degree, projects first. */
export function importanceTiers(nodes) {
  const ranked = [...nodes].sort((a, b) => (b.kind === 'project') - (a.kind === 'project') || (b.degree || 0) - (a.degree || 0) || (a.id < b.id ? -1 : 1));
  const hubs = Math.max(1, Math.round(nodes.length * 0.1)), notable = Math.round(nodes.length * 0.4);
  return new Map(ranked.map((n, i) => [n.id, i < hubs ? 0 : i < notable ? 1 : 2]));
}
