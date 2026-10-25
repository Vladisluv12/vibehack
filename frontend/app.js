let checkpoints=null,checkpointAt=0,checkpointPromise=null;
import ForceGraph from 'force-graph';
import { adjacency, communities, degrees, endId, importanceTiers, localSubgraph, withPositions } from './graph-model.js';
import { parse } from './markdown.js';

const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="control-token"]').content;
let extension={connected:false};
const labels = {project:'Проект',research:'Исследование',source:'Источник',decision:'Решение',session:'Сессия',concept:'Понятие',note:'Заметка'};
const reviews = {proposed:'Предложено · ждёт проверки',unreviewed:'Не проверено',reviewed:'Проверено человеком'};
const reviewShort = {reviewed:'проверено',proposed:'предложено',unreviewed:'не проверено'};
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const TAU = Math.PI*2, FLASH_MS = 2600;
const PALETTE = ['#8ee07a','#5ea8f2','#f28b40','#a98bff','#ef6f8f','#4fd1c5','#f2c14e','#e879c9','#7dd3fc','#c3d36b'];
const LONE = '#6b7280';
const css = getComputedStyle(document.documentElement);
const kindColor = Object.fromEntries(Object.keys(labels).map(k=>[k,css.getPropertyValue('--k-'+k).trim()]));
const BG = css.getPropertyValue('--bg').trim();
let currentNote, request=0, noticeTimer;
const el=(name,text,cls)=>{const n=document.createElement(name);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
const rgba=(hex,a)=>{const n=parseInt(hex.slice(1),16);return `rgba(${n>>16&255},${n>>8&255},${n&255},${a})`};
async function api(path, body) {
  const response=await fetch('/api/'+path,{method:body?'POST':'GET',headers:{'X-Control-Token':token,...(body?{'Content-Type':'application/json'}:{})},body:body?JSON.stringify(body):undefined});
  const value=await response.json();if(!response.ok)throw new Error(value.error||'Ошибка загрузки. Обновите страницу.');return value;
}
function notify(message,error=false){clearTimeout(noticeTimer);$('notice').textContent=message;$('notice').classList.toggle('error',error);noticeTimer=setTimeout(()=>$('notice').textContent='',error?14000:5000)}
async function run(button,fn){button.disabled=true;try{return await fn()}catch(e){notify(e.message,true)}finally{button.disabled=false}}
function countUp(node,value,suffix=''){
  const format=v=>Math.round(v).toLocaleString('ru-RU')+suffix,from=Number(node.dataset.value||0);node.dataset.value=value;
  if(reduced){node.textContent=format(value);return}
  const start=performance.now(),step=now=>{const p=Math.min(1,(now-start)/1100);node.textContent=format(from+(value-from)*(1-(1-p)**3));if(p<1)requestAnimationFrame(step)};requestAnimationFrame(step);
}

// ── Model: the whole vault graph, its clusters and node importance ─────────
let semantic={links:[]}, semanticOn=true, linkCounts=new Map();
let full={nodes:[],links:[]}, total=0, byId=new Map(), groups=[], groupOf=new Map(), tiers=new Map(), projectTitles={}, space='human';
const spaceQuery=()=>space==='all'?{}:{space};
let graph, shown=new Map(), shownAdj=new Map(), scope='all', depth=1, colorMode='cluster', hover=null, selected=null;
let searchMatches=null, legendFocus=null, lit=null, fitPending=true;
const flashes=new Map(), runningByProject=new Map();
const knownTitles=new Map();
const titleOf=id=>byId.get(id)?.title||knownTitles.get(id)||projectTitles[id]||id;
const colorOf=id=>{const n=byId.get(id);if(!n)return LONE;return colorMode==='kind'?kindColor[n.kind]||kindColor.note:groups[groupOf.get(id)]?.color||LONE};
const radius=node=>{const n=byId.get(node.id)||node;return 2.2+Math.sqrt(n.degree||0)*1.3+(n.kind==='project'?1.6:0)};
const touches=(link,id)=>id&&(endId(link.source)===id||endId(link.target)===id);
const isHot=link=>hover?touches(link,hover.id):touches(link,selected);
const phase=id=>parseInt(id.slice(0,6),16)%628/100;
function buildModel(g){
  full=g;total=g.total;projectTitles=g.projects||{};const degree=degrees(g.links);
  byId=new Map(g.nodes.map(n=>[n.id,{...n,degree:degree.get(n.id)||0}]));
  groups=communities(g.nodes,g.links);groupOf=new Map();let colour=0;
  groups.forEach((group,i)=>{
    const lead=byId.get(group.key)?.kind==='project'?byId.get(group.key):[...group.ids].map(id=>byId.get(id)).sort((a,b)=>b.degree-a.degree)[0];
    group.name=projectTitles[group.key]||lead?.title||'Без названия';group.color=group.ids.length>1?PALETTE[colour++%PALETTE.length]:LONE;
    for(const id of group.ids)groupOf.set(id,i);
  });
  tiers=importanceTiers([...byId.values()]);
}
// One set decides what is lit: hover neighbourhood, then legend focus, then search hits.
function refreshLit(){lit=hover?new Set([hover.id,...(shownAdj.get(hover.id)||[])]):legendFocus?.ids||searchMatches;redraw()}
function redraw(){if(!graph)return;graph.linkDirectionalParticles(l=>!reduced&&isHot(l)?3:0);graph.nodeCanvasObject(drawNode)}

// ── Rendering ─────────────────────────────────────────────────────────────
function drawClusterGlow(ctx){
  if(colorMode!=='cluster')return;
  ctx.save();ctx.globalCompositeOperation='lighter';
  for(const group of groups){
    if(group.ids.length<3)continue;
    const members=group.ids.map(id=>shown.get(id)).filter(n=>n&&Number.isFinite(n.x));if(members.length<3)continue;
    const cx=members.reduce((s,n)=>s+n.x,0)/members.length,cy=members.reduce((s,n)=>s+n.y,0)/members.length;
    const spread=members.reduce((s,n)=>s+Math.hypot(n.x-cx,n.y-cy),0)/members.length*1.7+30;
    const quiet=lit&&!group.ids.some(id=>lit.has(id));
    const g=ctx.createRadialGradient(cx,cy,0,cx,cy,spread);g.addColorStop(0,rgba(group.color,quiet?.02:.085));g.addColorStop(1,rgba(group.color,0));
    ctx.fillStyle=g;ctx.beginPath();ctx.arc(cx,cy,spread,0,TAU);ctx.fill();
  }
  ctx.restore();
}
function drawNode(node,ctx,scale){
  if(!Number.isFinite(node.x)||!Number.isFinite(node.y))return;
  const r=radius(node),c=colorOf(node.id),tier=tiers.get(node.id)??2,dim=lit&&!lit.has(node.id),a=dim?.1:1,now=performance.now();
  const breath=!reduced&&tier===0?.82+.18*Math.sin(now/950+phase(node.id)):1;
  const glow=r*(tier===0?5.5:3.4)*(node.id===selected?1.5:1);
  ctx.save();ctx.globalCompositeOperation='lighter';
  const g=ctx.createRadialGradient(node.x,node.y,r*.3,node.x,node.y,glow);g.addColorStop(0,rgba(c,(tier===0?.5:.26)*a*breath));g.addColorStop(1,rgba(c,0));
  ctx.fillStyle=g;ctx.beginPath();ctx.arc(node.x,node.y,glow,0,TAU);ctx.fill();ctx.restore();
  const until=flashes.get(node.id);
  if(until>now){const left=(until-now)/FLASH_MS;ctx.globalAlpha=left;ctx.beginPath();ctx.arc(node.x,node.y,r+2+(1-left)*16,0,TAU);ctx.strokeStyle=c;ctx.lineWidth=2/scale;ctx.stroke()}
  ctx.beginPath();ctx.arc(node.x,node.y,r,0,TAU);
  if(node.review==='unreviewed'){ctx.globalAlpha=a;ctx.fillStyle=BG;ctx.fill();ctx.fillStyle=rgba(c,.22);ctx.fill();ctx.strokeStyle=c;ctx.lineWidth=Math.max(1/scale,r*.32);ctx.stroke()}
  else{ctx.globalAlpha=a*(node.review==='proposed'?.5:1);ctx.fillStyle=c;ctx.fill();if(node.review==='proposed'){ctx.globalAlpha=a;ctx.strokeStyle=c;ctx.lineWidth=Math.max(1/scale,r*.28);ctx.stroke()}}
  if(tier===0&&node.review!=='unreviewed'){ctx.globalAlpha=a*.55;ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(node.x,node.y,r*.38,0,TAU);ctx.fill()}
  if(node.id===selected){ctx.globalAlpha=1;ctx.strokeStyle='#fff';ctx.lineWidth=1.6/scale;ctx.beginPath();ctx.arc(node.x,node.y,r+4/scale+1.5,0,TAU);ctx.stroke()}
  const running=runningByProject.get(node.id);
  if(running){const bx=node.x+r*.9,by=node.y-r*.9,br=Math.max(5/scale,2.5);ctx.globalAlpha=1;ctx.fillStyle=kindColor.concept;ctx.beginPath();ctx.arc(bx,by,br,0,TAU);ctx.fill();ctx.fillStyle=BG;ctx.font=`700 ${br*1.2}px system-ui,sans-serif`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(running),bx,by)}
  ctx.globalAlpha=1;
}
// Floating panels the labels must not run under; re-measured at most twice a second.
const HUD=['.hud-left','.command','.actions','.kpis','#inspector','#drawer','#chat','.chat-fab','.legend','.toolbar','#activity-panel'];
let hudBoxes=[],hudAt=0;
function hud(){const now=performance.now();if(now-hudAt>500){hudAt=now;hudBoxes=HUD.map(s=>document.querySelector(s)).filter(e=>e&&!e.hidden&&!e.closest('[hidden]')).map(e=>{const r=e.getBoundingClientRect();return [r.left-6,r.top-6,r.right+6,r.bottom+6]}).filter(b=>b[2]>b[0])}return hudBoxes}
// Screen-space labels: hubs large and bright, the rest smaller and fainter; no overlaps.
function drawLabels(ctx,scale){
  const dpr=window.devicePixelRatio||1,placed=[...hud()];
  ctx.save();ctx.setTransform(dpr,0,0,dpr,0,0);ctx.textAlign='left';ctx.textBaseline='middle';ctx.lineJoin='round';ctx.lineWidth=3.5;ctx.strokeStyle='rgba(8,9,12,.85)';
  const order=[...shown.values()].filter(n=>Number.isFinite(n.x)).map(n=>({n,f:n.id===selected||n.id===hover?.id,p:!!lit?.has(n.id),t:tiers.get(n.id)??2}))
    .sort((a,b)=>b.f-a.f||b.p-a.p||a.t-b.t||(byId.get(b.n.id)?.degree||0)-(byId.get(a.n.id)?.degree||0));
  for(const {n,f,p:preferred,t} of order){
    const dim=lit&&!preferred;
    if(!f&&(dim||!(preferred||t===0||(t===1&&scale>.8)||scale>1.7||scope==='local')))continue;
    const size=t===0?15:t===1?12.5:11.5,text=n.title.length>30?n.title.slice(0,28)+'…':n.title;
    ctx.font=`${t===0||n.id===selected?600:450} ${size}px Inter,"Segoe UI",system-ui,sans-serif`;
    const p=graph.graph2ScreenCoords(n.x,n.y),w=ctx.measureText(text).width,gap=radius(n)*scale+7;
    const x=p.x+gap+w>innerWidth-8?p.x-gap-w:p.x+gap,box=[x-3,p.y-size/2-3,x+w+3,p.y+size/2+3];
    if(!f&&placed.some(b=>box[0]<b[2]&&box[2]>b[0]&&box[1]<b[3]&&box[3]>b[1]))continue;
    placed.push(box);
    ctx.globalAlpha=f||t===0?1:preferred?.95:t===1?.82:.55;ctx.strokeText(text,x,p.y);ctx.fillStyle=f||t===0?'#ffffff':'#d3d8e0';ctx.fillText(text,x,p.y);
  }
  ctx.restore();
}
// Pull each cluster toward its own centre so colours form readable regions, and
// keep disconnected components (e.g. a fresh GitHub import) near the rest.
function clusterForce(){let nodes=[];const force=alpha=>{const sums=new Map();for(const n of nodes){const g=groupOf.get(n.id),s=sums.get(g)||{x:0,y:0,c:0};s.x+=n.x;s.y+=n.y;s.c++;sums.set(g,s)}for(const n of nodes){n.vx-=n.x*.03*alpha;n.vy-=n.y*.03*alpha;const s=sums.get(groupOf.get(n.id));if(s.c<2)continue;n.vx+=(s.x/s.c-n.x)*.09*alpha;n.vy+=(s.y/s.c-n.y)*.09*alpha}};force.initialize=ns=>{nodes=ns};return force}
function tooltip(node){const n=byId.get(node.id)||node,tip=el('div',undefined,'graph-tooltip');tip.append(el('strong',n.title),el('small',`${labels[n.kind]} · ${reviewShort[n.review]} · связей: ${n.degree}`),el('small',groups[groupOf.get(n.id)]?.name||''));return tip}
function linkTooltip(link){const tip=el('div',undefined,'graph-tooltip');if(link.semantic){tip.append(el('span',`${titleOf(endId(link.source))} ≈ ${titleOf(endId(link.target))}`),el('small',`Смысловое сходство ${link.score.toFixed(2)} · явной связи нет`));return tip}tip.append(el('span',titleOf(endId(link.source))),el('small',`→ ${link.relation} →`),el('span',titleOf(endId(link.target))));return tip}
function initializeGraph(){
  try{
    graph=new ForceGraph($('graph')).backgroundColor('rgba(0,0,0,0)').nodeId('id').nodeRelSize(1).nodeVal(n=>radius(n)**2)
      .nodeCanvasObject(drawNode).nodePointerAreaPaint((node,color,ctx)=>{ctx.fillStyle=color;ctx.beginPath();ctx.arc(node.x,node.y,radius(node)+3,0,TAU);ctx.fill()})
      .nodeLabel(tooltip).linkLabel(linkTooltip).linkCurvature(l=>l.semantic?0:.22).linkLineDash(l=>l.semantic?[2,3]:null)
      .linkColor(l=>{const s=endId(l.source),t=endId(l.target),a=isHot(l)?.9:lit&&!(lit.has(s)&&lit.has(t))?.03:l.semantic?.3:.26;return rgba(l.semantic?'#e8ecf3':colorOf(s),a)})
      .linkWidth(l=>isHot(l)?(l.semantic?1.3:1.8):l.semantic?.6:.7).linkDirectionalParticleWidth(2.6).linkDirectionalParticleSpeed(.007).linkDirectionalParticleColor(l=>colorOf(endId(l.source)))
      .onNodeClick(node=>openNote(node.id)).onNodeHover(node=>{hover=node||null;$('graph').style.cursor=node?'pointer':'';refreshLit()})
      .onBackgroundClick(()=>{if(legendFocus){legendFocus=null;renderLegend();refreshLit();status()}})
      .onRenderFramePre(drawClusterGlow).onRenderFramePost(drawLabels)
      .onEngineStop(()=>{if(fitPending){fitPending=false;fit(reduced?0:900)}})
      .minZoom(.15).maxZoom(10).warmupTicks(reduced?300:0).cooldownTicks(reduced?0:260);
    graph.d3Force('charge').strength(-95).distanceMax(260);graph.d3Force('link').distance(38).strength(l=>l.semantic?.04:1/Math.max(1,Math.min(linkCounts.get(endId(l.source))||1,linkCounts.get(endId(l.target))||1)));graph.d3Force('cluster',clusterForce());
    new ResizeObserver(()=>graph.width(innerWidth).height(innerHeight)).observe(document.body);
  }catch(e){graph=null;notify('Граф недоступен в этом браузере: '+e.message,true)}
}
const fit=(ms=600,filter)=>graph?.zoomToFit(reduced?0:ms,innerWidth<760?40:120,filter);
function showGraph(refit=false){
  if(!graph)return;
  const local=scope==='local'&&selected,base=local?localSubgraph(full,selected,depth):full;
  // Suggested semantic links: dashed, barely pulling, never part of clusters or degrees.
  const extra=semanticOn&&!local?semantic.links.filter(l=>byId.has(l.source)&&byId.has(l.target)).map(l=>({...l,relation:'semantic',semantic:true})):[];
  linkCounts=degrees(base.links);
  const next=withPositions({nodes:base.nodes,links:[...base.links,...extra]},graph.graphData().nodes);
  shownAdj=adjacency(next.links);shown=new Map(next.nodes.map(n=>[n.id,n]));
  graph.autoPauseRedraw(reduced||next.nodes.length>300);
  if(refit){fitPending=true;if(!reduced)setTimeout(()=>fit(700),900)}
  graph.graphData(next);status();
}
function status(){
  let text=`${full.nodes.length} заметок · ${full.links.length} связей`+(full.limited?` · показаны ${full.nodes.length} из ${total}`:'');
  if(scope==='local'&&selected)text=`Окружение «${titleOf(selected)}» · ${shown.size} узлов`;
  else if(legendFocus)text=`«${legendFocus.name}» · ${legendFocus.ids.size} заметок`;
  else if(searchMatches)text=`Найдено ${searchMatches.size} из ${full.nodes.length}`;
  const suggested=semanticOn&&scope!=='local'?semantic.links.filter(l=>byId.has(l.source)&&byId.has(l.target)).length:0;
  $('graph-status').textContent=text+(suggested?` · ≈ ${suggested} смысловых`:'');
}
function renderKpis(){
  const nodes=[...byId.values()],reviewed=nodes.filter(n=>n.review==='reviewed').length;
  countUp($('kpi-notes'),total);countUp($('kpi-links'),full.links.length);
  countUp($('kpi-reviewed'),nodes.length?reviewed/nodes.length*100:0,'%');countUp($('kpi-proposed'),nodes.length-reviewed);
  countUp($('kpi-clusters'),groups.filter(g=>g.ids.length>1).length);
}
function renderLegend(){
  const items=colorMode==='cluster'
    ?groups.filter(g=>g.ids.length>1).slice(0,8).map(g=>({key:g.key,name:g.name,color:g.color,ids:g.ids}))
    :Object.keys(labels).map(k=>({key:k,name:labels[k],color:kindColor[k],ids:[...byId.values()].filter(n=>n.kind===k).map(n=>n.id)})).filter(i=>i.ids.length);
  $('legend').replaceChildren();
  for(const item of items){
    const b=el('button',item.name.length>26?item.name.slice(0,24)+'…':item.name);b.title=item.name;b.style.setProperty('--dot',item.color);b.append(el('small',String(item.ids.length)));
    b.setAttribute('aria-pressed',String(legendFocus?.key===item.key));
    b.onclick=()=>{legendFocus=legendFocus?.key===item.key?null:{key:item.key,name:item.name,ids:new Set(item.ids)};renderLegend();refreshLit();status();if(legendFocus)fit(700,n=>legendFocus.ids.has(n.id));else fit()};
    $('legend').append(b);
  }
}
function flash(ids){
  if(!graph)return;const until=performance.now()+FLASH_MS;
  for(const id of ids){if(!shown.has(id))continue;flashes.set(id,until);if(!reduced)for(const link of graph.graphData().links.filter(l=>touches(l,id)).slice(0,6))graph.emitParticle(link)}
  graph.autoPauseRedraw(false);clearTimeout(flash.timer);flash.timer=setTimeout(()=>{flashes.clear();graph.autoPauseRedraw(reduced||shown.size>300);redraw()},FLASH_MS+50);
}
async function load(refit=false){
  try{const g=await api('brain/graph?'+new URLSearchParams(spaceQuery()));$('count-human').textContent=g.spaces.human;$('count-agent').textContent=g.spaces.agent;buildModel(g);if(legendFocus)legendFocus=null;renderLegend();renderKpis();showGraph(refit);if(drawerOpen()){renderKindFilter();loadCatalog()}loadSemantic()}catch(e){notify(e.message,true)}
}
let semanticSerial=0;
async function loadSemantic(){
  const serial=++semanticSerial;
  try{const result=await api('brain/semantic');if(serial!==semanticSerial)return;semantic=result;showGraph();renderSimilar()}
  catch(e){if(serial===semanticSerial){semantic={links:[]};showGraph();renderSimilar();notify('Смысловые связи недоступны: '+e.message,true)}}
}

// ── Inspector ─────────────────────────────────────────────────────────────
function resolveWiki(target){const key=target.trim().toLowerCase();for(const n of byId.values())if(n.id===key||n.title.toLowerCase()===key||n.path?.replace(/\.md$/,'').toLowerCase()===key)return n.id;for(const [id,title] of knownTitles)if(id===key||title.toLowerCase()===key)return id;return null}
function renderInline(tokens,cite=null){
  const out=document.createDocumentFragment();
  for(const tok of tokens){
    if(tok.t==='text'&&cite){for(const part of tok.v.split(/(\[\d{1,2}\])/)){const n=part.match(/^\[(\d{1,2})\]$/);out.append(n&&cite(Number(n[1]))||part)}}
    else if(tok.t==='text')out.append(tok.v);
    else if(tok.t==='code')out.append(el('code',tok.v));
    else if(tok.t==='strong')out.append(el('strong',tok.v));
    else if(tok.t==='em')out.append(el('em',tok.v));
    else if(tok.t==='link'){const a=el('a',tok.v);a.href=tok.href;a.target='_blank';a.rel='noopener noreferrer';out.append(a)}
    else{const id=resolveWiki(tok.target),label=id&&tok.v===tok.target?titleOf(id):tok.v;if(id){const b=el('button',label,'wikilink');b.onclick=()=>openNote(id);out.append(b)}else out.append(el('span',label,'wikilink missing'))}
  }
  return out;
}
function renderMarkdown(text){
  const box=$('content');box.replaceChildren();
  for(const block of parse(text.split(/\n## Relations\s*\n/)[0])){
    if(block.type==='code'){const pre=el('pre');pre.append(el('code',block.text));box.append(pre)}
    else if(block.type==='ul'||block.type==='ol'){const list=el(block.type);if(block.start>1)list.start=block.start;for(const item of block.items){const li=el('li');li.append(renderInline(item));list.append(li)}box.append(list)}
    else{const node=el(block.type==='h'?'h4':'p');node.append(renderInline(block.inline));box.append(node)}
  }
}
function safeSource(source){try{const u=new URL(source);return ['https:','http:'].includes(u.protocol)?u.href:null}catch{return null}}
async function openNote(id){
  const selection=++request;
  try{
    const note=await api('brain/note?id='+encodeURIComponent(id));if(selection!==request)return;
    selected=id;currentNote=note;history.replaceState(null,'','#note='+id);
    for(const r of note.relations){if(r.source_title)knownTitles.set(r.source,r.source_title);if(r.target_title)knownTitles.set(r.target,r.target_title)}
    if(chatOpen())setChat(false);
    $('inspector').hidden=false;$('editing').hidden=true;$('content').hidden=false;
    $('note-kind').textContent=labels[note.kind];$('note-kind').style.color=kindColor[note.kind];$('note-title').textContent=note.title;
    $('review').textContent=reviews[note.review];$('review').className='review '+note.review;$('note-space').textContent=note.space==='agent'?'Агент':'Человек';$('note-space').className='space-badge '+note.space;
    const group=groups[groupOf.get(id)];$('note-cluster').hidden=!group||group.ids.length<2;if(group){$('note-cluster').textContent=group.name;$('note-cluster').style.setProperty('--dot',group.color)}
    $('metadata').replaceChildren();
    for(const [key,value] of [['Источник',note.meta.source],['Создано',note.meta.created?.slice(0,10)],['Проект',note.meta.project],['Раздел',note.meta.source_section],['Исполнение',note.meta.execution_status],['Сессия',note.meta.native_session]]){
      if(!value)continue;$('metadata').append(el('dt',key));const dd=el('dd'),url=key==='Источник'?safeSource(value):null;
      if(url){const a=el('a',value);a.href=url;a.target='_blank';a.rel='noopener noreferrer';dd.append(a)}else dd.textContent=String(value);$('metadata').append(dd);
    }
    $('obsidian').href='obsidian://open?vault=SecondBrain&file='+encodeURIComponent(note.path);
    $('note-tags').replaceChildren(...(note.tags||[]).map(tag=>{const b=el('button','#'+tag);b.title='Все заметки с этим тегом';b.onclick=()=>searchTag(tag);return b}));
    renderMarkdown(note.body);renderRelations(note);renderSimilar();showProjectMetrics(note,selection);catalogSelected();
    $('scope-local').disabled=false;$('scope-local').title='';
    if(scope==='local')showGraph(true);else{refreshLit();focusNode(id)}
  }catch(e){notify(e.message,true)}
}
function renderRelations(note){
  $('relations').replaceChildren();$('relations-count').textContent=note.relations.length||'';
  for(const relation of note.relations){
    const outgoing=relation.source===note.id,target=outgoing?relation.target:relation.source,b=el('button');
    b.style.setProperty('--dot',colorOf(target));b.append(el('span',titleOf(target)),el('small',(outgoing?'→ ':'← ')+relation.relation));
    b.onclick=()=>openNote(target);b.onmouseenter=()=>{hover=shown.get(target)||null;refreshLit()};b.onmouseleave=()=>{hover=null;refreshLit()};
    $('relations').append(b);
  }
  if(!note.relations.length)$('relations').append(el('p','Явных связей пока нет.','hint'));
}
function focusNode(id){
  const node=shown.get(id);if(!graph||!node||!Number.isFinite(node.x))return;
  const k=tourTimer?1.9:Math.max(graph.zoom(),1.6),ms=reduced?0:700,shift=innerWidth>760?210/k:0;
  graph.zoom(k,ms);graph.centerAt(node.x+shift,node.y,ms);
}
function renderSimilar(){
  const box=$('similar'),section=$('similar-section');
  if(!currentNote){section.hidden=true;return}
  const mine=semantic.links.filter(l=>l.source===currentNote.id||l.target===currentNote.id);
  section.hidden=false;box.replaceChildren();$('similar-count').textContent=mine.length||'';
  if(!(semantic.spaces||[]).includes(currentNote.space)){box.append(el('p',`Смысловые связи ищутся в пространстве «${(semantic.spaces||[]).map(s=>s==='agent'?'агенты':'люди').join(', ')}». Изменить можно в настройках ⚙.`,'hint'));return}
  if(!mine.length){box.append(el('p','Похожих заметок без явной связи не найдено.','hint'));return}
  for(const link of mine){
    const other=link.source===currentNote.id?link.target:link.source,item=el('div',undefined,'similar-item'),head=el('div',undefined,'similar-head');
    head.style.setProperty('--dot',colorOf(other));
    const open=el('button',titleOf(other));open.onclick=()=>openNote(other);open.onmouseenter=()=>{hover=shown.get(other)||null;refreshLit()};open.onmouseleave=()=>{hover=null;refreshLit()};
    const score=el('span',`${Math.round(link.score*100)}%`,'similar-score');if(link.duplicate)score.append(el('span','возможный дубль','dup'));
    head.append(open,score);item.append(head);
    if(link.why){const details=el('details');details.append(el('summary','похожие фрагменты'));const [a,b]=link.source===currentNote.id?link.why:[link.why[1],link.why[0]];details.append(el('blockquote',a),el('blockquote',b));item.append(details)}
    const actions=el('div',undefined,'similar-actions'),accept=el('button','Связать','pill primary'),reject=el('button','Скрыть','pill');
    accept.onclick=()=>run(accept,async()=>{const id=currentNote.id;await api('brain/link',{source:id,target:other});await load();await openNote(id);notify('Связь добавлена в Markdown, предыдущая версия сохранена.')});
    reject.onclick=()=>run(reject,async()=>{await api('brain/dismiss',{source:currentNote.id,target:other});await loadSemantic();notify('Предложение скрыто и больше не появится.')});
    actions.append(accept,reject);item.append(actions);box.append(item);
  }
}
function closeNote(){++request;selected=null;currentNote=null;hover=null;$('inspector').hidden=true;catalogSelected();history.replaceState(null,'',location.pathname);$('scope-local').disabled=true;$('scope-local').title='Сначала выберите заметку';if(scope==='local')setScope('all');else refreshLit()}
function setScope(next){scope=next;for(const name of ['all','local']){$('scope-'+name).classList.toggle('active',scope===name);$('scope-'+name).setAttribute('aria-pressed',String(scope===name))}$('depth-group').hidden=scope!=='local';showGraph(true)}
function setDepth(next){depth=next;for(const n of [1,2]){$('depth-'+n).classList.toggle('active',depth===n);$('depth-'+n).setAttribute('aria-pressed',String(depth===n))}showGraph(true)}
function setColorMode(next){colorMode=next;legendFocus=null;for(const name of ['cluster','kind']){$('color-'+name).classList.toggle('active',colorMode===name);$('color-'+name).setAttribute('aria-pressed',String(colorMode===name))}renderLegend();refreshLit();status();if(drawerOpen())loadCatalog()}

// Demo tour: walks the hubs of the real graph, one note card at a time.
let tourTimer=null,tourIndex=0;
function tourStep(){const hubs=[...byId.values()].filter(n=>tiers.get(n.id)===0).sort((a,b)=>b.degree-a.degree);if(!hubs.length)return stopTour();openNote(hubs[tourIndex++%hubs.length].id)}
function startTour(){if(scope==='local')setScope('all');legendFocus=null;searchMatches=null;$('search').value='';renderLegend();$('tour').setAttribute('aria-pressed','true');$('tour').textContent='■ Остановить';tourTimer=setInterval(tourStep,6000);tourStep()}
function stopTour(){if(!tourTimer)return;clearInterval(tourTimer);tourTimer=null;$('tour').setAttribute('aria-pressed','false');$('tour').textContent='▶ Демо-тур'}
$('tour').onclick=()=>tourTimer?stopTour():startTour();
for(const type of ['pointerdown','wheel','keydown'])document.addEventListener(type,e=>{if(!$('tour').contains(e.target))stopTour()},{capture:true,passive:true});

// Spaces: human notes and agent notes form separate graphs; «Всё» shows both.
function setSpace(next){
  if(next===space)return;space=next;stopTour();legendFocus=null;
  for(const name of ['human','agent','all']){$('space-'+name).classList.toggle('active',space===name);$('space-'+name).setAttribute('aria-pressed',String(space===name))}
  if(selected)closeNote();if(scope==='local')setScope('all');
  if($('search').value.trim())search();load(true);
}
for(const name of ['human','agent','all'])$('space-'+name).onclick=()=>setSpace(name);

// "#tag" in the search box filters by tag; anything else is full-text.
const queryParams=q=>q.startsWith('#')?{tag:q.slice(1)}:{q};
async function searchTag(tag){$('search').value='#'+tag;if(!drawerOpen())setDrawer(true);await search();hideResults()}

// Search mode: BM25 stays the default; semantic and hybrid use the embedding settings.
let searchMode='bm25';
const MODE_LABELS={bm25:'BM25',semantic:'По смыслу',hybrid:'Гибрид'};
function showSearchMode(){$('search-mode').textContent=MODE_LABELS[searchMode];$('search-mode').className='mode-pill '+searchMode;for(const b of document.querySelectorAll('.results-modes button'))b.setAttribute('aria-pressed',String(b.dataset.mode===searchMode))}
async function setSearchMode(mode){searchMode=mode;showSearchMode();try{await api('settings',{search:{mode}})}catch(e){notify(e.message,true)}if($('search').value.trim())search()}
$('search-mode').onclick=()=>setSearchMode({bm25:'semantic',semantic:'hybrid',hybrid:'bm25'}[searchMode]);
for(const b of document.querySelectorAll('.results-modes button'))b.onmousedown=e=>{e.preventDefault();setSearchMode(b.dataset.mode)};
const searchApi=(q,limit)=>q.startsWith('#')||searchMode==='bm25'
  ?api('brain/notes?'+new URLSearchParams({...queryParams(q),limit,...spaceQuery()}))
  :api('brain/search?'+new URLSearchParams({q,mode:searchMode,limit:Math.min(limit,50),...spaceQuery()}));

// ── Command search: results list plus highlighted matches on the graph ────
let searchTimer,searchSerial=0,results=[],active=-1;
function renderResults(){
  const box=$('results-list');box.replaceChildren();showSearchMode();
  results.slice(0,8).forEach((note,i)=>{
    const b=el('button',undefined,'result');b.setAttribute('role','option');b.setAttribute('aria-selected',String(i===active));b.style.setProperty('--dot',colorOf(note.id));
    const excerpt=(note.excerpt||'').split('## Relations')[0].replace(/\[\[[0-9a-f]{32}\]\]/g,'').replace(/\s+/g,' ').trim();
    const title=el('strong',note.title);if(searchMode==='semantic'&&note.score!==undefined)title.append(el('span',`≈ ${Math.round(note.score*100)}%`,'score'));
    b.append(title,el('small',excerpt&&excerpt!=='…'?excerpt:`${labels[note.kind]} · ${reviewShort[note.review]}`));
    b.onmousedown=e=>{e.preventDefault();pick(note.id)};box.append(b);
  });
  if(!results.length)box.append(el('p',searchMode==='bm25'?'Ничего не найдено. Попробуйте другое слово или поиск по смыслу.':'Ничего не найдено.','empty'));
  $('results').hidden=false;$('search').setAttribute('aria-expanded','true');
}
function hideResults(){$('results').hidden=true;$('search').setAttribute('aria-expanded','false')}
function pick(id){hideResults();openNote(id)}
async function search(){
  const q=$('search').value.trim(),serial=++searchSerial;
  if(drawerOpen())loadCatalog();
  if(!q||q==='#'){results=[];searchMatches=null;hideResults();refreshLit();status();return}
  try{const found=await searchApi(q,500);if(serial!==searchSerial)return;results=found.items;active=results.length?0:-1;searchMatches=new Set(results.map(n=>n.id));renderResults();refreshLit();status()}catch(e){notify(e.message,true)}
}
$('search').addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(search,180)});
$('search').addEventListener('focus',()=>{if(results.length&&$('search').value.trim())renderResults()});
$('search').addEventListener('blur',()=>setTimeout(hideResults,120));
$('search').addEventListener('keydown',e=>{
  const shownCount=Math.min(8,results.length);
  if(e.key==='ArrowDown'&&shownCount){e.preventDefault();active=(active+1)%shownCount;renderResults()}
  else if(e.key==='ArrowUp'&&shownCount){e.preventDefault();active=(active-1+shownCount)%shownCount;renderResults()}
  else if(e.key==='Enter'&&results[active]){e.preventDefault();pick(results[active].id)}
  else if(e.key==='Escape'){$('search').value='';search();$('search').blur()}
});
document.addEventListener('keydown',e=>{
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'||e.key==='/'&&!/INPUT|TEXTAREA/.test(document.activeElement.tagName)){e.preventDefault();$('search').focus();$('search').select();return}
  if(e.key==='Escape'&&!$('create-dialog').open&&!$('github-dialog').open&&!$('settings-dialog').open&&$('editing').hidden&&document.activeElement!==$('search')){if(chatOpen())setChat(false);else if(selected)closeNote();else if(drawerOpen())setDrawer(false)}
});

// ── Drawer: catalog of notes; follows the top search when it is active ────
let catalogKind='',catalogReview='',catalogNext=null,catalogSerial=0;
const drawerOpen=()=>!$('drawer').hidden;
function setDrawer(open){$('drawer').hidden=!open;$('drawer-toggle').setAttribute('aria-expanded',String(open));document.body.classList.toggle('drawer-open',open);if(open){renderKindFilter();loadCatalog()}}
function renderReviewFilter(){
  const counts={};for(const n of byId.values())counts[n.review]=(counts[n.review]||0)+1;
  $('review-filter').replaceChildren();
  for(const [review,label] of [['','Все статусы'],['unreviewed','Не проверено'],['proposed','Предложено'],['reviewed','Проверено']]){
    const b=el('button',review?`${label} ${counts[review]||0}`:label,review);b.setAttribute('aria-pressed',String(catalogReview===review));
    b.onclick=()=>{catalogReview=review;renderReviewFilter();loadCatalog()};$('review-filter').append(b);
  }
}
let tagSerial=0;
async function renderTagFilter(){
  const serial=++tagSerial;
  try{
    const {tags}=await api('brain/tags?'+new URLSearchParams(spaceQuery()));if(serial!==tagSerial)return;
    const current=$('search').value.trim().toLowerCase();$('tag-filter').replaceChildren();
    for(const {tag,count} of tags.slice(0,12)){const b=el('button','#'+tag);b.append(el('small',String(count)));b.setAttribute('aria-pressed',String(current==='#'+tag));b.onclick=()=>current==='#'+tag?($('search').value='',search()):searchTag(tag);$('tag-filter').append(b)}
  }catch{}
}
function renderKindFilter(){
  renderReviewFilter();
  const counts={};for(const n of byId.values())counts[n.kind]=(counts[n.kind]||0)+1;
  $('kind-filter').replaceChildren();
  for(const kind of ['',...Object.keys(labels).filter(k=>counts[k])]){
    const b=el('button',kind?`${labels[kind]} ${counts[kind]}`:'Все');if(kind)b.style.setProperty('--dot',kindColor[kind]);
    b.setAttribute('aria-pressed',String(catalogKind===kind));b.onclick=()=>{catalogKind=kind;renderKindFilter();loadCatalog()};$('kind-filter').append(b);
  }
}
function catalogSelected(){for(const b of $('catalog').querySelectorAll('.catalog-item'))b.classList.toggle('selected',b.dataset.id===selected)}
async function loadCatalog(append=false){
  const serial=++catalogSerial,q=$('search').value.trim()==='#'?'':$('search').value.trim();
  try{
    const semanticList=q&&!q.startsWith('#')&&searchMode!=='bm25';
    let result=semanticList?await searchApi(q,50):await api('brain/notes?'+new URLSearchParams({...queryParams(q),kind:catalogKind,review:catalogReview,offset:append?catalogNext:0,limit:50,...spaceQuery()}));if(serial!==catalogSerial)return;
    if(semanticList){const items=result.items.filter(n=>(!catalogKind||n.kind===catalogKind)&&(!catalogReview||n.review===catalogReview));result={items,total:items.length,next:null}}
    if(!append)$('catalog').replaceChildren();
    for(const note of result.items){
      const b=el('button',undefined,'catalog-item');b.dataset.id=note.id;b.style.setProperty('--dot',colorOf(note.id));
      b.append(el('span',note.title),el('small',`${labels[note.kind]} · ${reviewShort[note.review]}`,note.review));
      b.onclick=()=>openNote(note.id);b.onmouseenter=()=>{hover=shown.get(note.id)||null;refreshLit()};b.onmouseleave=()=>{hover=null;refreshLit()};
      $('catalog').append(b);
    }
    if(!result.total)$('catalog').append(el('p',q?'По этому запросу заметок нет.':'Заметок с такими фильтрами нет.','hint'));
    catalogNext=result.next;$('more').hidden=catalogNext===null;$('catalog-count').textContent=result.total;
    $('catalog-query').hidden=!q;$('catalog-query').textContent=q.startsWith('#')?`Заметки с тегом ${q}`:`${MODE_LABELS[searchMode]}: «${q}»`;catalogSelected();renderTagFilter();
  }catch(e){notify(e.message,true)}
}
$('drawer-new').onclick=()=>$('create-dialog').showModal();
$('drawer-toggle').onclick=()=>setDrawer(!drawerOpen());$('drawer-close').onclick=()=>setDrawer(false);
$('more').onclick=()=>run($('more'),()=>loadCatalog(true));

// ── GitHub import ─────────────────────────────────────────────────────────
function openGitHub(){$('github-status').textContent='';$('github-status').classList.remove('error');$('github-dialog').showModal();$('github-url').focus()}
$('github').onclick=openGitHub;$('github-drawer').onclick=openGitHub;$('close-github').onclick=()=>$('github-dialog').close();
$('github-form').onsubmit=async e=>{
  e.preventDefault();const button=e.target.querySelector('[type=submit]'),state=$('github-status');
  button.disabled=true;state.classList.remove('error');state.textContent='Скачиваю архив репозитория и разбираю Markdown…';
  try{
    const result=await api('brain/import-github',{url:$('github-url').value});
    $('github-dialog').close();e.target.reset();await load();await openNote(result.project);flash([result.project,...result.sources]);
    notify(`Импортировано из ${result.repository.replace('https://github.com/','')}: ${result.imported} Markdown-файлов`+(result.skipped?`, пропущено ${result.skipped}`:'')+(result.omitted?`, сверх лимита ${result.omitted}`:'')+'. Статус — «Не проверено».');
  }catch(error){state.classList.add('error');state.textContent=error.message}
  finally{button.disabled=false}
};

// ── Settings: embedding endpoint and semantic suggestion parameters ──────
const URL_HINTS={openai:'http://localhost:1234/v1 или https://api.openai.com/v1',ollama:'http://localhost:11434',tei:'http://localhost:8080'};
function settingsStatus(text,error=false){$('settings-status').textContent=text;$('settings-status').classList.toggle('error',error)}
function providerFields(){const p=$('set-provider').value;$('endpoint-fields').hidden=p==='local';$('set-url').placeholder=URL_HINTS[p]||'';$('set-model').placeholder=p==='tei'?'задаётся при запуске TEI':'bge-m3'}
function thresholdLabel(){const auto=$('set-auto').checked;$('set-threshold').disabled=auto;$('threshold-value').textContent=auto?(semantic.auto&&semantic.threshold?`сейчас ${semantic.threshold.toFixed(2)}`:'авто'):Number($('set-threshold').value).toFixed(2)}
function fillSettings(values){
  const e=values.embeddings,s=values.semantic,f=$('settings-form');
  f.provider.value=e.provider;f.url.value=e.url;f.model.value=e.model;f.document_prefix.value=e.document_prefix;f.query_prefix.value=e.query_prefix;f.api_key.value='';
  const c=values.chat;f.chat_provider.value=c.provider;f.chat_url.value=c.url;f.chat_model.value=c.model;f.chat_notes.value=c.context_notes;f.chat_key.value='';
  f.chat_key.placeholder=c.api_key_set?'Ключ сохранён — оставьте пустым, чтобы не менять':'Не нужен для Ollama';
  for(const r of f.querySelectorAll('[name=search_mode]'))r.checked=r.value===values.search.mode;
  f.api_key.placeholder=e.api_key_set?'Ключ сохранён — оставьте пустым, чтобы не менять':'Не нужен для локальных серверов';
  for(const box of f.querySelectorAll('[name=space]'))box.checked=s.spaces.includes(box.value);
  $('set-auto').checked=s.threshold==='auto';f.threshold.value=s.threshold==='auto'?(semantic.threshold||.5):s.threshold;f.top_k.value=s.top_k;providerFields();thresholdLabel();
}
function settingsPayload(){
  const f=$('settings-form'),e={provider:f.provider.value,url:f.url.value,model:f.model.value,document_prefix:f.document_prefix.value,query_prefix:f.query_prefix.value};
  if(f.api_key.value)e.api_key=f.api_key.value;
  const chat={provider:f.chat_provider.value,url:f.chat_url.value,model:f.chat_model.value,context_notes:Number(f.chat_notes.value)};if(f.chat_key.value)chat.api_key=f.chat_key.value;
  const mode=f.querySelector('[name=search_mode]:checked')?.value||searchMode;
  return {embeddings:e,chat,search:{mode},semantic:{spaces:[...f.querySelectorAll('[name=space]:checked')].map(b=>b.value),threshold:$('set-auto').checked?'auto':Number(f.threshold.value),top_k:Number(f.top_k.value)}};
}
$('settings').onclick=async()=>{try{applySettings(await api('settings'));settingsStatus('');$('settings-dialog').showModal()}catch(e){notify(e.message,true)}};
$('close-settings').onclick=()=>$('settings-dialog').close();$('set-provider').onchange=providerFields;$('set-threshold').oninput=thresholdLabel;$('set-auto').onchange=thresholdLabel;
$('settings-test').onclick=()=>run($('settings-test'),async()=>{settingsStatus('Сохраняю и запрашиваю тестовый вектор…');try{applySettings(await api('settings',settingsPayload()));const r=await api('settings/test',{});settingsStatus(`Работает: ${r.provider}${r.model?' · '+r.model:''} · размерность ${r.dims}`)}catch(e){settingsStatus(e.message,true)}});
$('settings-form').onsubmit=e=>{e.preventDefault();const button=$('settings-form').querySelector('[type=submit]');run(button,async()=>{try{applySettings(await api('settings',settingsPayload()));settingsStatus('Сохранено. Считаю смысловые связи…');await loadSemantic();$('settings-dialog').close();notify(`Смысловые связи пересчитаны: ${semantic.links.length} предложений`+(semantic.fetched?`, запрошено векторов: ${semantic.fetched}`:'')+'.')}catch(err){settingsStatus(err.message,true)}})};
$('chat-test').onclick=()=>run($('chat-test'),async()=>{settingsStatus('Сохраняю и спрашиваю модель…');try{applySettings(await api('settings',settingsPayload()));const r=await api('settings/test-chat',{});settingsStatus(`Чат работает: ${r.model} ответила «${r.reply}»`)}catch(e){settingsStatus(e.message,true)}});
function applySettings(values){fillSettings(values);searchMode=values.search.mode;chatConfigured=!!values.chat.url;$('chat-model').textContent=chatConfigured?`${values.chat.model} · ответы только по вашим заметкам`:'Модель не подключена';showSearchMode();if(chatOpen())renderChat()}
$('semantic-toggle').onclick=()=>{semanticOn=!semanticOn;$('semantic-toggle').setAttribute('aria-pressed',String(semanticOn));showGraph()};

// ── Chat with the knowledge base ──────────────────────────────────────────
let chatMessages=[],chatBusy=false,chatConfigured=false;
const chatOpen=()=>!$('chat').hidden;
function setChat(open){$('chat').hidden=!open;$('chat-toggle').setAttribute('aria-expanded',String(open));document.body.classList.toggle('chat-open',open);if(open){renderChat();$('chat-input').focus()}}
function chatEmpty(){
  const box=el('div',undefined,'chat-empty');
  if(!chatConfigured){box.append(el('p','Чтобы общаться с LLM, укажите модель в настройках: например, Ollama с qwen3 или любой OpenAI-совместимый сервер.'));const b=el('button','Открыть настройки ⚙');b.onclick=()=>$('settings').click();box.append(b);return box}
  box.append(el('p','Спросите что-нибудь — ответ будет построен по вашим заметкам со ссылками на них.'));
  for(const example of ['Как мы защищаемся от того, что агент выдумывает факты?','Какие решения приняты по хранению данных?','Что мешает новой команде начать пользоваться?']){const b=el('button',example);b.onclick=()=>{$('chat-input').value=example;sendChat()};box.append(b)}
  return box;
}
function renderAnswer(message){
  const box=el('div',undefined,'msg bot'),byNumber=new Map((message.sources||[]).map(s=>[s.n,s]));
  const cite=n=>{const source=byNumber.get(n);if(!source)return null;const b=el('button',String(n),'cite');b.title=source.title;b.onclick=()=>citeOpen(source.id);b.onmouseenter=()=>{hover=shown.get(source.id)||null;refreshLit()};b.onmouseleave=()=>{hover=null;refreshLit()};return b};
  for(const block of parse(message.content)){
    if(block.type==='code'){const pre=el('pre');pre.append(el('code',block.text));box.append(pre)}
    else if(block.type==='ul'||block.type==='ol'){const list=el(block.type);if(block.start>1)list.start=block.start;for(const item of block.items){const li=el('li');li.append(renderInline(item,cite));list.append(li)}box.append(list)}
    else{const node=el(block.type==='h'?'h4':'p');node.append(renderInline(block.inline,cite));box.append(node)}
  }
  const cited=(message.sources||[]).filter(s=>s.cited),other=(message.sources||[]).filter(s=>!s.cited);
  const sourceButton=s=>{const b=el('button',undefined,'source');b.append(el('b',`[${s.n}]`),el('span',s.title),el('small',reviewShort[s.review],s.review));b.onclick=()=>citeOpen(s.id);return b};
  if(message.sources?.length){
    const list=el('div',undefined,'sources');cited.forEach(s=>list.append(sourceButton(s)));
    if(other.length){const more=el('details');more.append(el('summary',cited.length?`ещё найдено: ${other.length}`:`найдено заметок: ${other.length}, в ответе не процитированы`));other.forEach(s=>more.append(sourceButton(s)));list.append(more)}
    box.append(list);
  }
  const actions=el('div',undefined,'msg-actions'),save=el('button','Сохранить как черновик','pill');
  save.title='Ответ модели станет заметкой «предложено» в пространстве агентов со связями на процитированные заметки';
  save.onclick=()=>run(save,async()=>{const question=message.question||'';const result=await api('brain/create',{title:('Ответ: '+question).slice(0,190),body:`Вопрос: ${question}\n\n${message.content}`,kind:'note',origin:'chat',links:cited.map(s=>s.id),tags:['chat']});save.textContent='Сохранено';save.disabled=true;await load();notify('Ответ сохранён черновиком в пространство агентов.');void result});
  actions.append(save);box.append(actions);
  return box;
}
function citeOpen(id){setChat(false);openNote(id)}
function renderChat(){
  const log=$('chat-log');log.replaceChildren();
  if(!chatMessages.length)log.append(chatEmpty());
  for(const message of chatMessages){
    if(message.role==='user')log.append(el('div',message.content,'msg user'));
    else if(message.error)log.append(el('div',message.content,'msg error'));
    else log.append(renderAnswer(message));
  }
  if(chatBusy)log.append(el('div','Ищу подходящие заметки и спрашиваю модель','msg thinking'));
  log.scrollTop=log.scrollHeight;
}
async function sendChat(){
  const text=$('chat-input').value.trim();if(!text||chatBusy)return;
  chatMessages.push({role:'user',content:text});$('chat-input').value='';chatBusy=true;renderChat();
  try{
    const history=chatMessages.filter(m=>!m.error).map(m=>({role:m.role,content:m.content})).slice(-12);
    const result=await api('brain/chat',{messages:history,space:space==='all'?'':space});
    chatMessages.push({role:'assistant',content:result.answer,sources:result.sources,question:text});
    const ids=result.sources.filter(s=>s.cited).map(s=>s.id);if(ids.length)flash(ids);
  }catch(e){chatMessages.push({role:'assistant',content:'Не получилось ответить: '+e.message,error:true})}
  finally{chatBusy=false;renderChat()}
}
$('chat-toggle').onclick=()=>setChat(!chatOpen());$('chat-close').onclick=()=>setChat(false);
$('chat-clear').onclick=()=>{chatMessages=[];renderChat()};
$('chat-form').onsubmit=e=>{e.preventDefault();sendChat()};
$('chat-input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendChat()}});

// ── Agent Control extension ──────────────────────────────────────────────
let activityCursor=null,activityBusy=false,activityFeed=[],callTargets=new Map();
const activityLabels={'task.start':'Начало работы','task.complete':'Исполнение завершено','task.error':'Остановка / ошибка','tool.start':'Вызов функции','tool.finish':'Ответ функции','memory.search':'Поиск в памяти','memory.read':'Чтение памяти','memory.draft':'Черновик в памяти','agent.message':'Обращение к агенту'};
const connection=text=>{$('connection').querySelector('span').textContent=text};
function renderActivity(){
  $('activity-feed').replaceChildren();
  for(const item of activityFeed.slice(0,5)){
    const row=el('div',undefined,'activity-row '+(item.type==='task.error'||item.failed?'activity-error':''));
    row.append(el('time',new Date(item.ts*1000).toLocaleTimeString('ru-RU',{hour:'2-digit',minute:'2-digit',second:'2-digit'})),el('span',`${item.provider} · ${activityLabels[item.type]||item.label}${item.tool?' / '+item.tool:''}`));
    row.title=item.label+(item.live?'':' · событие из очереди после перерыва');
    $('activity-feed').append(row);
  }
  if(!activityFeed.length)$('activity-feed').append(el('p','Новые вызовы появятся во время работы агентов.','hint'));
}
function applyExtension(){
  const link=$('queue-link'),name=extension.name||'Agent Control';
  if(extension.connected){$('activity-panel').hidden=false;$('activity-state').textContent='Подключение к '+name+'…';return}
  link.removeAttribute('href');link.setAttribute('aria-disabled','true');link.title='Agent Control не подключён';
  connection('Локально · без сети');
}
function setRunning(tasks){runningByProject.clear();for(const task of tasks||[])if(task.project_id)runningByProject.set(task.project_id,(runningByProject.get(task.project_id)||0)+1);redraw()}
async function pollActivity(){
  if(!extension.connected||activityBusy)return;activityBusy=true;
  try{
    const result=await api('activity'+(activityCursor===null?'':'?after='+activityCursor));
    if(result.initial){activityFeed=[];callTargets.clear()}
    activityCursor=result.cursor;setRunning(result.tasks);
    const touched=[];
    for(const raw of result.items){
      const item={...raw};const callKey=item.task_id+':'+item.attempt+':'+item.call_id;
      if(item.type==='tool.finish'&&item.call_id){const started=callTargets.get(callKey);if(started){item.target_id=started.target_id;item.tool=started.tool;item.label='Ответ: '+started.tool+(item.failed?' · ошибка':'');callTargets.delete(callKey)}}
      else if(item.call_id){callTargets.set(callKey,item);if(callTargets.size>256)callTargets.delete(callTargets.keys().next().value)}
      activityFeed.unshift(item);if(item.live&&item.target_id)touched.push(String(item.target_id));
    }
    activityFeed=activityFeed.slice(0,40);if(touched.length)flash(touched);renderActivity();
    $('activity-state').textContent=result.running?`${result.running} в работе`:'Нет активных задач';
    connection(result.running?`${result.running} агентов в работе`:'Локально · агенты на связи');
    if(result.omitted)$('activity-state').textContent+=' · поток сокращён';
  }catch{setRunning([]);connection('Нет соединения');$('activity-state').textContent='Поток недоступен'}
  finally{activityBusy=false}
}

// ── Actions ───────────────────────────────────────────────────────────────
$('sync').onclick=()=>run($('sync'),async()=>{const result=await api('brain/sync',{});await load();notify(`Обновлено: ${result.notes} заметок, ${result.links} связей.`+(result.warning_count?` Замечаний к связям: ${result.warning_count}.`:''))});
$('backup').onclick=()=>run($('backup'),async()=>{const result=await api('brain/backup',{});notify('Копия Markdown и версий сохранена: '+result.path)});
$('close-note').onclick=closeNote;$('focus-local').onclick=()=>setScope('local');
$('edit').onclick=()=>{$('editor').value=currentNote.content;$('content').hidden=true;$('editing').hidden=false;$('editor').focus()};$('cancel-edit').onclick=()=>{$('editing').hidden=true;$('content').hidden=false};
$('save').onclick=()=>run($('save'),async()=>{const note=await api('brain/edit',{id:currentNote.id,content:$('editor').value,hash:currentNote.hash});await load();await openNote(note.id);notify('Изменения сохранены. Предыдущая версия оставлена в vault.')});
$('new-note').onclick=()=>$('create-dialog').showModal();$('close-dialog').onclick=()=>$('create-dialog').close();
$('create-form').onsubmit=e=>{e.preventDefault();const button=e.target.querySelector('[type=submit]');run(button,async()=>{const form=Object.fromEntries(new FormData(e.target));form.tags=form.tags.split(/[,\s]+/).map(t=>t.replace(/^#/,'')).filter(Boolean);const result=await api('brain/create',form);$('create-dialog').close();e.target.reset();await load();await openNote(result.id);notify('Заметка создана со статусом «Предложено».')})};
$('color-cluster').onclick=()=>setColorMode('cluster');$('color-kind').onclick=()=>setColorMode('kind');
$('scope-all').onclick=()=>setScope('all');$('scope-local').onclick=()=>setScope('local');$('depth-1').onclick=()=>setDepth(1);$('depth-2').onclick=()=>setDepth(2);
$('fit').onclick=()=>fit();

initializeGraph();
extension=await api('extension').catch(()=>({connected:false}));applyExtension();
await api('settings').then(applySettings).catch(()=>{});
await load(true);await pollActivity();
const initial=new URLSearchParams(location.hash.slice(1)).get('note');if(initial)openNote(initial);
setInterval(()=>{if(!document.hidden)pollActivity()},1800);

async function showProjectMetrics(note,selection){
  $('project-checkpoint').hidden=note.kind!=='project'||!extension.connected;if(note.kind!=='project'||!extension.connected)return;
  $('project-values').replaceChildren(el('p','Читаю локальный checkpoint…','hint'));
  try{
    if(!checkpoints||Date.now()-checkpointAt>60000){
      if(!checkpointPromise)checkpointPromise=api('project-metrics').then(value=>{checkpoints=value;checkpointAt=Date.now();return value}).finally(()=>checkpointPromise=null);
      await checkpointPromise;
    }
    if(selection!==request||currentNote?.id!==note.id)return;
    const m=checkpoints.projects.find(p=>p.project_id===note.id),box=$('project-values');box.replaceChildren();
    if(!m){box.append(el('p','Нет явно настроенного checkpoint для проекта.','hint'));return}
    const number=value=>Number.isFinite(value)?new Intl.NumberFormat('ru-RU').format(value):'неизвестно';
    const date=value=>Number.isFinite(value)&&value>0?new Date(value*1000).toLocaleString('ru-RU'):'неизвестно';
    const rows=[['Материалы в памяти',number(m.memory_materials)],['Последний Git checkpoint',m.git?m.git.head.slice(0,8)+' · '+date(m.git.committed_at):'неизвестно'],['Задачи Agent Control в работе',number(m.managed_running.length)]];
    if(m.project.endsWith('/news-grid'))rows.push(['Записи корпуса',number(m.corpus?.items)],['Записи с датированным body',number(m.corpus?.body_items)],['Источники в реестре',number(m.polling?.sources)],['Последняя попытка опроса',date(m.polling?.last_attempt)],['Последняя запись результата опроса',date(m.polling?.last_ok)]);
    for(const [label,value] of rows){const row=el('div');row.append(el('span',label),el('strong',value));box.append(row)}
    box.append(el('p',`Срез: ${date(m.observed_at)} · кэш до ${m.cache_ttl_s} с. Извлечённый body не гарантирует полноту статьи.`,'hint'));
    box.append(el('p','Внешние сессии Qoder не считаются задачами этой очереди; время polling показывает наблюдение, а не состояние процесса.','hint'));
    if(m.errors.length)box.append(el('p',m.errors.join(' · '),'hint'));
  }catch(error){if(selection===request)$('project-values').replaceChildren(el('p','Checkpoint недоступен: '+error.message,'hint'))}
}
setInterval(()=>{if(!document.hidden&&currentNote?.kind==='project')showProjectMetrics(currentNote,request)},60000);
