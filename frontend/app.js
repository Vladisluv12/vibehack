let checkpoints=null,checkpointAt=0,checkpointPromise=null;
import ForceGraph3D from '3d-force-graph';
import * as THREE from 'three';
import { ParticleField } from './particle-field.js';

const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="control-token"]').content;
let extension={connected:false};
const colors = {project:'#ffc978',research:'#ae96ff',source:'#6dafff',session:'#64d9d2',decision:'#f5a5c6',concept:'#a5d89c',note:'#bdc9df'};
const labels = {project:'Проект',research:'Исследование',source:'Источник',session:'Сессия',decision:'Решение',concept:'Понятие',note:'Заметка'};
const reviews = {proposed:'Предложено · требует проверки',unreviewed:'Не проверено',reviewed:'Проверено'};
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
let fitPending=true;
let flow, rendererMode='flow';
let graph, data={nodes:[],links:[]}, selected, currentNote, nextPage=null, request=0, paused=reduced, motionOptIn=false, view='knowledge', noticeTimer, searchTimer;
const el=(name,text,cls)=>{const n=document.createElement(name);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
async function api(path, body) {
  const response=await fetch('/api/'+path,{method:body?'POST':'GET',headers:{'X-Control-Token':token,...(body?{'Content-Type':'application/json'}:{})},body:body?JSON.stringify(body):undefined});
  const value=await response.json();if(!response.ok)throw new Error(value.error||'Ошибка загрузки. Обновите страницу.');return value;
}
function notify(message,error=false){clearTimeout(noticeTimer);$('notice').textContent=message;$('notice').style.borderColor=error?'#ba7e72':'#526986';noticeTimer=setTimeout(()=>$('notice').textContent='',error?14000:6000)}
async function run(button,fn){button.disabled=true;try{return await fn()}catch(e){notify(e.message,true)}finally{button.disabled=false}}
function query(offset=0){return new URLSearchParams({q:$('search').value,kind:$('kind').value,offset,limit:50}).toString()}
function selectedStyle(){for(const b of $('catalog').querySelectorAll('button'))b.classList.toggle('selected',b.dataset.id===selected)}
function renderList(result,append=false){if(!append)$('catalog').replaceChildren();for(const note of result.items){const b=el('button');b.dataset.id=note.id;b.style.setProperty('--dot',colors[note.kind]);const text=el('span');text.append(el('span',note.title,'item-title'),el('small',labels[note.kind]+' · '+(note.review==='reviewed'?'проверено':note.review==='proposed'?'предложено':'не проверено')));b.append(text);b.onclick=()=>openNote(note.id);$('catalog').append(b)}nextPage=result.next;$('more').hidden=nextPage===null;$('catalog-count').textContent=result.total;if(!result.total)$('catalog').append(el('p','Заметок не найдено. Попробуйте другой запрос.','hint'));selectedStyle()}
function safeSource(source){try{const u=new URL(source);return ['https:','http:'].includes(u.protocol)?u.href:null}catch{return null}}
async function openNote(id){const selection=++request;try{const note=await api('brain/note?id='+encodeURIComponent(id));if(selection!==request)return;selected=id;currentNote=note;history.replaceState(null,'','#note='+id);$('welcome').hidden=true;$('note').hidden=false;$('editing').hidden=true;$('content').hidden=false;$('note-title').textContent=note.title;$('note-kind').textContent=labels[note.kind];$('note-kind').style.color=colors[note.kind];$('review').textContent=reviews[note.review];$('review').className='review '+note.review;$('content').textContent=note.body;$('metadata').replaceChildren();showProjectMetrics(note,selection);for(const [key,value] of [['Источник',note.meta.source],['Проект',note.meta.project],['Раздел',note.meta.source_section],['Исполнение',note.meta.execution_status],['Сессия',note.meta.native_session],['ID',note.id]]){if(!value)continue;$('metadata').append(el('dt',key));const dd=el('dd');const url=key==='Источник'?safeSource(value):null;if(url){const a=el('a',value);a.href=url;a.target='_blank';a.rel='noopener noreferrer';dd.append(a)}else dd.textContent=String(value);$('metadata').append(dd)}$('obsidian').href='obsidian://open?vault=SecondBrain&file='+encodeURIComponent(note.path);$('relations').replaceChildren();for(const relation of note.relations){const target=relation.source===id?relation.target:relation.source;const found=data.nodes.find(n=>n.id===target);const b=el('button',found?.title||target);b.onclick=()=>openNote(target);$('relations').append(b)}if(!note.relations.length)$('relations').append(el('p','Явных связей пока нет.','hint'));selectedStyle();flow?.focus(id);if(graph&&rendererMode==='graph'){const node=data.nodes.find(n=>n.id===id);if(node&&Number.isFinite(node.x)&&!paused){const distance=170;const ratio=1+distance/Math.max(1,Math.hypot(node.x,node.y,node.z||0));graph.cameraPosition({x:node.x*ratio,y:node.y*ratio,z:(node.z||40)*ratio},node,reduced?0:850)}graph.nodeThreeObject(nodeObject);graph.linkColor(linkColor);if(paused)drawGraphOnce()} }catch(e){notify(e.message,true)}}
function nodeObject(node){const group=new THREE.Group();const radius=node.kind==='project'?6:node.kind==='decision'?3.6:2.5;const color=colors[node.kind]||'#bdc9df';const sphere=new THREE.Mesh(new THREE.SphereGeometry(radius,12,10),new THREE.MeshBasicMaterial({color}));group.add(sphere);const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:glow,color,transparent:true,blending:THREE.AdditiveBlending,depthWrite:false,opacity:selected===node.id?.9:.55}));sprite.scale.set(radius*9,radius*9,1);group.add(sprite);if(node.kind==='project'||selected===node.id){const canvas=document.createElement('canvas');canvas.width=768;canvas.height=96;const ctx=canvas.getContext('2d');ctx.font='30px "Segoe UI",sans-serif';ctx.fillStyle='#d8e4f7';ctx.textAlign='center';const label=node.title.length>40?node.title.slice(0,38)+'…':node.title;ctx.fillText(label,384,57);const tag=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(canvas),transparent:true,depthWrite:false}));tag.scale.set(96,12,1);tag.position.y=radius+9;group.add(tag)}return group}
const glowCanvas=document.createElement('canvas');glowCanvas.width=glowCanvas.height=128;const glowCtx=glowCanvas.getContext('2d');const gradient=glowCtx.createRadialGradient(64,64,0,64,64,64);gradient.addColorStop(0,'rgba(255,255,255,1)');gradient.addColorStop(.12,'rgba(255,255,255,.7)');gradient.addColorStop(.38,'rgba(255,255,255,.12)');gradient.addColorStop(1,'rgba(255,255,255,0)');glowCtx.fillStyle=gradient;glowCtx.fillRect(0,0,128,128);const glow=new THREE.CanvasTexture(glowCanvas);
const endId=x=>typeof x==='object'?x.id:x;
function linkColor(link){if(selected&&(endId(link.source)===selected||endId(link.target)===selected))return '#96b7fa';return '#3d527d'}
function initializeGraph(){try{graph=new ForceGraph3D($('graph'),{controlType:'orbit',rendererConfig:{antialias:true,alpha:true}}).backgroundColor('#060b15').nodeId('id').nodeThreeObject(nodeObject).nodeLabel(node=>{const tip=el('div',node.title,'graph-tooltip');return tip}).linkColor(linkColor).linkOpacity(.42).linkWidth(.45).onNodeClick(node=>openNote(node.id)).onBackgroundClick(()=>{}).onEngineStop(()=>{if(fitPending&&!paused){fitPending=false;graph.zoomToFit(reduced?0:700,100)}}).showNavInfo(false).warmupTicks(reduced?80:20).cooldownTicks(reduced?0:120);graph.d3Force('charge').strength(-85);graph.d3Force('link').distance(link=>endId(link.source)===endId(link.target)?20:50);graph.cameraPosition({x:80,y:60,z:390});const controls=graph.controls();controls.enableDamping=!reduced;controls.minDistance=35;controls.maxDistance=1200;new ResizeObserver(entries=>{const {width,height}=entries[0].contentRect;graph.width(width).height(height)}).observe($('graph'));}catch(e){$('graph-empty').hidden=false;$('graph-empty').textContent='3D недоступен в этом браузере. Все знания доступны в каталоге слева.';$('map-title').textContent='Вся память — рядом.';for(const id of ['three','two'])$(id).disabled=true;notify('WebGL недоступен. Используйте каталог заметок.',true)}}
let loadSerial=0;
async function load(){const serial=++loadSerial;try{const [list,g]=await Promise.all([api('brain/notes?'+query()),api('brain/graph?'+query())]);if(serial!==loadSerial)return;renderList(list);const old=new Map(data.nodes.map(n=>[n.id,n]));data={nodes:g.nodes.map((n,i)=>({...n,...(old.has(n.id)?{x:old.get(n.id).x,y:old.get(n.id).y,z:old.get(n.id).z}:{x:Math.cos(i*2.4)*90,y:Math.sin(i*2.4)*70,z:Math.sin(i*.71)*60})})),links:g.links.map(x=>({...x}))};const picker=$('project-focus'),prior=picker.value;picker.replaceChildren();const all=el('option','Все проекты');all.value='';picker.append(all);for(const project of g.nodes.filter(n=>n.kind==='project')){const o=el('option',`${project.title} · ${project.project_materials||1} материалов`);o.value=project.id;picker.append(o)}if([...picker.options].some(o=>o.value===prior))picker.value=prior;$('graph-count').textContent=`${g.nodes.length} узлов · ${g.links.length} связей`+(g.limited?` · из ${g.total} найденных`:'');$('graph-empty').hidden=g.nodes.length>0;flow?.setData(data);if(graph){fitPending=true;graph.graphData(data);if(paused||rendererMode!=='graph')graph.pauseAnimation();if(paused&&rendererMode==='graph')drawGraphOnce()}$('summary').replaceChildren(el('div',`${g.total} материалов в текущем разделе`),el('div',`${g.nodes.filter(n=>n.review==='reviewed').length} проверенных записей на карте`),el('div','Ссылки ведут к исходным материалам'));}catch(e){notify(e.message,true)}}
let activityCursor=null,activityBusy=false,activityFeed=[],callTargets=new Map();
const activityLabels={'task.start':'Начало работы','task.complete':'Исполнение завершено','task.error':'Остановка / ошибка','tool.start':'Вызов функции','tool.finish':'Ответ функции','memory.search':'Поиск в памяти','memory.read':'Чтение памяти','memory.draft':'Черновик в памяти','agent.message':'Обращение к агенту'};
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
  if(extension.connected){$('activity-state').textContent='Подключение к '+name+'…';return}
  link.removeAttribute('href');link.setAttribute('aria-disabled','true');link.title='Agent Control не подключён';
  $('activity-state').textContent='Agent Control не подключён';$('connection').textContent='Локальная база подключена';
  $('activity-feed').replaceChildren(el('p','Лента вызовов появится, когда панель запущена через Agent Control.','hint'));
}
async function status(){
  if(!extension.connected||activityBusy)return;activityBusy=true;
  try{
    const result=await api('activity'+(activityCursor===null?'':'?after='+activityCursor));
    if(result.initial){activityFeed=[];callTargets.clear()}
    activityCursor=result.cursor;flow?.setActivity({running:result.running,tasks:result.tasks});
    const events=[];
    for(const raw of result.items){
      const item={...raw};const callKey=item.task_id+':'+item.attempt+':'+item.call_id;
      if(item.type==='tool.finish'&&item.call_id){const started=callTargets.get(callKey);if(started){item.target_id=started.target_id;item.tool=started.tool;item.label='Ответ: '+started.tool+(item.failed?' · ошибка':'');callTargets.delete(callKey)}}
      else if(item.call_id){callTargets.set(callKey,item);if(callTargets.size>256)callTargets.delete(callTargets.keys().next().value)}
      activityFeed.unshift(item);if(item.live)events.push(item);
    }
    activityFeed=activityFeed.slice(0,40);if(events.length)flow?.pushEvents(events);renderActivity();
    $('activity-state').textContent=result.running?`${result.running} в работе`:'Нет активных задач';
    $('connection').textContent=result.running?`${result.running} задач в работе`:'Локальная база подключена';
    if(result.omitted)$('activity-state').textContent+=' · поток сокращён';
  }catch{flow?.setActivity({running:0,tasks:[]});$('connection').textContent='Нет соединения';$('activity-state').textContent='Поток недоступен'}
  finally{activityBusy=false}
}

function setView(next){view=next;$('kind').value=next==='sessions'?'session':'';$('view-title').textContent=next==='sessions'?'История сессий':'Ваши знания';$('map-title').textContent=next==='sessions'?'Работа остаётся в памяти.':'Связи имеют значение.';$('knowledge').classList.toggle('active',next==='knowledge');$('sessions').classList.toggle('active',next==='sessions');load()}
$('knowledge').onclick=()=>setView('knowledge');$('sessions').onclick=()=>setView('sessions');$('search').addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(load,250)});$('kind').onchange=load;
$('more').onclick=()=>run($('more'),async()=>{const result=await api('brain/notes?'+query(nextPage));renderList(result,true)});
$('sync').onclick=()=>run($('sync'),async()=>{const result=await api('brain/sync',{});await load();notify(`Обновлено: ${result.notes} заметок, ${result.links} связей.`+(result.warning_count?` Замечаний к связям: ${result.warning_count}.`:''))});
$('backup').onclick=()=>run($('backup'),async()=>{const result=await api('brain/backup',{});notify('Копия Markdown и версий сохранена: '+result.path)});
$('close-note').onclick=()=>{++request;selected=null;currentNote=null;flow?.focus(null);$('note').hidden=true;$('welcome').hidden=false;history.replaceState(null,'',location.pathname);selectedStyle();if(graph){graph.nodeThreeObject(nodeObject);graph.linkColor(linkColor)}};
$('edit').onclick=()=>{$('editor').value=currentNote.content;$('content').hidden=true;$('editing').hidden=false;$('editor').focus()};$('cancel-edit').onclick=()=>{$('editing').hidden=true;$('content').hidden=false};
$('save').onclick=()=>run($('save'),async()=>{const note=await api('brain/edit',{id:currentNote.id,content:$('editor').value,hash:currentNote.hash});await load();await openNote(note.id);notify('Изменения сохранены. Предыдущая версия оставлена в vault.')});
$('new-note').onclick=()=>$('create-dialog').showModal();$('close-dialog').onclick=()=>$('create-dialog').close();$('create-form').onsubmit=e=>{e.preventDefault();const button=e.target.querySelector('[type=submit]');run(button,async()=>{const result=await api('brain/create',Object.fromEntries(new FormData(e.target)));$('create-dialog').close();e.target.reset();await load();await openNote(result.id);notify('Заметка создана со статусом «Предложено».')})};
function dimensions(n){if(!graph)return;graph.numDimensions(n);for(const [id,num] of [['three',3],['two',2]]){$(id).classList.toggle('active',num===n);$(id).setAttribute('aria-pressed',String(num===n))}if(!paused&&rendererMode==='graph')graph.resumeAnimation();else if(paused&&rendererMode==='graph')drawGraphOnce()}
function flowStatus(info){$('flow-status').textContent=info.error||`${Number(info.particles||0).toLocaleString('ru')} частиц оформления · ${info.quality||'auto'}${info.fps?` · ${Math.round(info.fps)} FPS`:''}${info.reducedMotion?' · снижение движения в системе':info.paused?' · движение остановлено':''}${info.running?' · '+info.running+' агентов в работе':''}${info.effects?' · '+info.effects+' событий':''}`;if(info.error)notify(info.error,true)}
let graphDrawTimer, graphFirstDraw=true;
function drawGraphOnce(attempt=0){clearTimeout(graphDrawTimer);graphDrawTimer=setTimeout(()=>{if(!graph||rendererMode!=='graph'||document.hidden||!paused)return;if(data.nodes.some(n=>!n.__threeObj)&&attempt<4){drawGraphOnce(attempt+1);return}graph.pauseAnimation();graph.resumeAnimation();graph.pauseAnimation();if(graphFirstDraw){graph.zoomToFit(0,85);graphFirstDraw=false}graph.renderer().render(graph.scene(),graph.camera())},40)}
function updateMotion(){flow?.setPaused(paused);flow?.setVisible(rendererMode==='flow'&&!document.hidden);if(graph){if(paused||rendererMode!=='graph'||document.hidden){graph.pauseAnimation();if(paused&&rendererMode==='graph'&&!document.hidden)drawGraphOnce()}else graph.resumeAnimation()}}
function setRenderer(mode){
  rendererMode=mode;$('graph').hidden=mode!=='graph';$('flow').hidden=mode!=='flow';
  for(const name of ['graph','flow']){$('mode-'+name).classList.toggle('active',mode===name);$('mode-'+name).setAttribute('aria-pressed',String(mode===name))}
  $('graph-dimensions').hidden=mode!=='graph';$('flow-controls').hidden=mode!=='flow';$('flow-status').hidden=mode!=='flow';
  $('map-eyebrow').textContent=mode==='flow'?'FLOW / ОРБИТЫ ПАМЯТИ':'GRAPH / КАРТА СВЯЗЕЙ';
  if(mode==='graph'&&!graph){initializeGraph();if(graph){graph.graphData(data);graph.zoomToFit(0,85)}}
  if(mode==='flow'&&!flow){try{flow=new ParticleField({container:$('flow'),onSelect:node=>{if(node)openNote(node.id)},onStatus:flowStatus});flow.setQuality($('quality').value);flow.setPreset($('preset').value);flow.setData(data);if(selected)flow.focus(selected)}catch(e){$('flow-status').textContent='Flow недоступен. Переключитесь на Graph или используйте каталог.';notify(e.message,true)}}
  updateMotion();flow?.resize();
}
$('mode-graph').onclick=()=>setRenderer('graph');$('mode-flow').onclick=()=>setRenderer('flow');
$('quality').onchange=()=>flow?.setQuality($('quality').value);$('preset').onchange=()=>flow?.setPreset($('preset').value);$('glow').oninput=()=>flow?.setGlow(Number($('glow').value));
$('three').onclick=()=>dimensions(3);$('two').onclick=()=>dimensions(2);
$('fit').onclick=()=>{if(rendererMode==='flow')flow?.fit();else if(graph)graph.zoomToFit(reduced?0:800,70)};
$('pause').textContent=reduced?'Включить движение':'Пауза';$('pause').setAttribute('aria-pressed',String(paused));
$('pause').onclick=()=>{if(reduced&&!motionOptIn){motionOptIn=true;flow?.setMotionOverride(true)}paused=!paused;$('pause').textContent=paused?'Продолжить':'Пауза';$('pause').setAttribute('aria-pressed',String(paused));updateMotion()};
document.addEventListener('visibilitychange',updateMotion);
window.addEventListener('pagehide',event=>{if(event.persisted){flow?.setVisible(false);graph?.pauseAnimation()}else{flow?.dispose();graph?._destructor();}});
window.addEventListener('pageshow',updateMotion);
extension=await api('extension').catch(()=>({connected:false}));applyExtension();
setRenderer('flow');await load();await status();const initial=new URLSearchParams(location.hash.slice(1)).get('note');if(initial)openNote(initial);setInterval(()=>{if(!document.hidden)status()},1800);

$('project-focus').onchange=()=>{const id=$('project-focus').value;if(id)openNote(id);else if(rendererMode==='flow')flow?.fit();else graph?.zoomToFit(0,70)};
$('map-expand').onclick=()=>{const expanded=document.querySelector('.workspace').classList.toggle('map-expanded');$('map-expand').setAttribute('aria-pressed',String(expanded));$('map-expand').textContent=expanded?'Вернуть панель ⛶':'Карта проекта ⛶';flow?.resize();if(graph){const box=$('graph').getBoundingClientRect();graph.width(box.width).height(box.height)}if(rendererMode==='flow'){const id=$('project-focus').value;if(id)flow?.focus(id);else flow?.fit()}else graph?.zoomToFit(0,70)};


async function showProjectMetrics(note,selection){
  $('project-checkpoint').hidden=note.kind!=='project';if(note.kind!=='project')return;
  if(!extension.connected){$('project-values').replaceChildren(el('p','Метрики проекта появятся при подключении Agent Control.','hint'));return}
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
    box.append(el('p',`Срез: ${date(m.observed_at)} · кэш до ${m.cache_ttl_s} с. Размер корпуса не задаёт размер галактики. Извлечённый body не гарантирует полноту статьи.`,'hint'));
    box.append(el('p','Внешние сессии Qoder не считаются задачами этой очереди; время polling показывает наблюдение, а не состояние процесса.','hint'));
    if(m.errors.length)box.append(el('p',m.errors.join(' · '),'hint'));
  }catch(error){if(selection===request)$('project-values').replaceChildren(el('p','Checkpoint недоступен: '+error.message,'hint'))}
}
setInterval(()=>{if(!document.hidden&&currentNote?.kind==='project')showProjectMetrics(currentNote,request)},60000);
