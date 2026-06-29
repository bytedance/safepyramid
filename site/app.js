/* ============ SafePyramid · app.js ============ */
(function(){
const D = window.SP_DATA;
const DOMAINS = D.DOMAINS;
const LEVELS = ['L0','L1','L2','Avg'];
const LIDX = {L0:0,L1:1,L2:2,Avg:3};

// short codes + org colors for the model tag
const CODE = {
  'GPT-5.5':'5.5','Claude-Opus-4.7':'Cl','Kimi-K2.6':'Ki','DeepSeek-V4-Pro':'DS',
  'Doubao-Seed-2.0-Pro':'豆','Gemini-3.5-Flash':'Gm','Hunyuan-HY3-Preview':'HY',
  'Qwen-3.6-Max-Preview':'Qw','GLM-5.1':'GL','Grok-4.3':'Gr',
  'GPT-OSS-Safeguard-120B':'120','GPT-OSS-Safeguard-20B':'20','FlexGuard-Qwen3-8B':'FG',
  'DynaGuard-8B':'Dy','ShieldLM-14B-Qwen':'SL'
};
const ORGC = {
  'OpenAI':'#0B8C72','Anthropic':'#C2683B','Google':'#3B7DD8','DeepSeek':'#3A56C8',
  'Moonshot AI':'#5B49C9','ByteDance':'#2C73BD','Tencent':'#1FA4C9','Alibaba':'#D9772E',
  'Zhipu AI':'#4A6FA5','xAI':'#363B43','HK PolyU':'#6E7C36','University of Maryland':'#9A2F3A','Tsinghua University':'#7E3417'
};
const ORGLOGO = {
  'OpenAI':'openai','Anthropic':'claude','Moonshot AI':'kimi','DeepSeek':'deepseek',
  'ByteDance':'doubao','Google':'gemini','Tencent':'hunyuan','Alibaba':'qwen',
  'Zhipu AI':'zai','xAI':'grok'
};

const state = { protocol:'policy', metric:'RMR', level:'Avg', domain:'All' };

// ---- value lookup ----
function lower(){ return state.metric==='RDR'; }
function getLevels(model){
  // returns {L0,L1,L2,Avg} numbers or nulls for current protocol/metric/domain
  const overall = (state.protocol==='policy'?D.overallPolicy:D.overallRule)[model];
  if(state.domain==='All'){
    const a = overall[state.metric];
    return {L0:a[0],L1:a[1],L2:a[2],Avg:a[3]};
  }
  const pd = (state.protocol==='policy'?D.perDomain.policy:D.perDomain.rule)[model];
  if(!pd){ return {L0:null,L1:null,L2:null,Avg:null}; }
  const di = DOMAINS.indexOf(state.domain);
  const m = pd[state.metric];
  const l0=m.L0[di], l1=m.L1[di], l2=m.L2[di];
  return {L0:l0,L1:l1,L2:l2,Avg:Math.round(((l0+l1+l2)/3)*10)/10};
}
function models(){ return Object.keys(state.protocol==='policy'?D.overallPolicy:D.overallRule); }

// ---- controls ----
function seg(id,key,cb){
  const el=document.getElementById(id);
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b)return;
    [...el.children].forEach(x=>x.classList.remove('on'));
    b.classList.add('on');
    state[key]=b.dataset.v; cb&&cb();
  });
}
function paintMetricSeg(){
  const el=document.getElementById('seg-metric');
  [...el.children].forEach(b=>b.classList.toggle('clay', b.dataset.v==='RDR' && b.classList.contains('on')));
}
seg('seg-protocol','protocol',()=>{ ensureDomainValid(); render(); });
seg('seg-metric','metric',()=>{ paintMetricSeg(); render(); });
seg('seg-level','level',()=>{ render(); });

const sel = document.getElementById('dom-select');
['All'].concat(DOMAINS).forEach(d=>{
  const o=document.createElement('option'); o.value=d; o.textContent=(d==='All'?'All domains (overall)':d); sel.appendChild(o);
});
sel.addEventListener('change',()=>{ state.domain=sel.value; render(); });
function ensureDomainValid(){ /* domains identical across protocols */ }

// ---- sorting via header ----
function setLevel(lv){
  state.level=lv;
  const el=document.getElementById('seg-level');
  [...el.children].forEach(b=>b.classList.toggle('on', b.dataset.v===lv));
  render();
}

// ---- render ----
function render(){
  paintMetricSeg();
  const metric=state.metric, lv=state.level;
  let rows = models().map(m=>({ model:m, vals:getLevels(m), meta:D.META[m] }));
  // when a specific domain is selected, drop models with no per-domain data (e.g. GPT-OSS guards under per-policy)
  if(state.domain!=='All'){
    rows = rows.filter(r=>r.vals.L0!=null || r.vals.L1!=null || r.vals.L2!=null);
  }

  // sort by active level; nulls last
  rows.sort((a,b)=>{
    const av=a.vals[lv], bv=b.vals[lv];
    if(av==null && bv==null) return 0;
    if(av==null) return 1; if(bv==null) return -1;
    return lower()? av-bv : bv-av;
  });

  // best value for highlight + bar scaling
  const present = rows.map(r=>r.vals[lv]).filter(v=>v!=null);
  const best = present.length? (lower()?Math.min(...present):Math.max(...present)) : 0;

  // head
  const head=document.getElementById('lb-head');
  head.innerHTML = `<tr>
    <th class="crank">#</th>
    <th class="cmodel">Model</th>
    ${LEVELS.map(L=>`<th class="sortable ${L===lv?'act':''} ${L==='Avg'?'':'hide-sm'}" data-lv="${L}">${L==='Avg'?'Avg':L}<span class="car">${lower()?'▲':'▼'}</span></th>`).join('')}
  </tr>`;
  head.querySelectorAll('th.sortable').forEach(th=>th.addEventListener('click',()=>setLevel(th.dataset.lv)));

  // body
  const body=document.getElementById('lb-body');
  body.innerHTML = rows.map((r,i)=>{
    const c = ORGC[r.meta.org]||'#555';
    const isGuard = r.meta.type==='guard';
    const cells = LEVELS.map(L=>{
      const v=r.vals[L];
      const active = L===lv;
      if(v==null) return `<td class="${L==='Avg'?'':'hide-sm'} dim"><span class="dash">—</span></td>`;
      if(active){
        const w = Math.max(2, Math.min(100, v))+'%';
        const barClass = lower()?'clay':'';
        const isBest = Math.abs(v-best)<0.001;
        return `<td><div class="bar-wrap">
          <div class="bar-track"><div class="bar-fill ${barClass}" style="width:${w}"></div></div>
          <span class="bar-val ${isBest?'best':''}">${v.toFixed(1)}</span>
        </div></td>`;
      }
      return `<td class="${L==='Avg'?'':'hide-sm'}">${v.toFixed(1)}</td>`;
    }).join('');
    const rankCls = i===0?'rank top':'rank';
    const logo = ORGLOGO[r.meta.org];
    const tag = logo
      ? `<span class="mtag logo"><img src="assets/logos/${logo}.svg" alt="${r.meta.org}" loading="lazy" /></span>`
      : `<span class="mtag" style="background:${c}">${CODE[r.model]||''}</span>`;
    return `<tr>
      <td class="${rankCls}">${i+1}</td>
      <td class="cmodel"><div class="modelcell">
        ${tag}
        <span class="minfo"><span class="mn">${r.model}<span class="typechip ${isGuard?'tc-guard':'tc-frontier'}">${isGuard?'guard':'LLM'}</span></span><span class="mo">${r.meta.org}</span></span>
      </div></td>
      ${cells}
    </tr>`;
  }).join('');

  // meta + footnote
  const pretty = {RMR:'RMR',['RMR@1.0']:'Exact match',RDR:'RDR'}[metric];
  document.getElementById('lb-meta').innerHTML =
    `${state.protocol==='policy'?'Per-policy':'Per-rule'} · ${pretty}<br>${rows.length} models`;

  const notes = {
    RMR:'<b>RMR</b> — rule matching rate, averaged over thresholds τ∈{0.7,0.8,0.9,1.0}. Higher is better.',
    'RMR@1.0':'<b>Exact match (RMR@1.0)</b> — share of cases where the predicted violated-rule set is exactly correct. Higher is better.',
    RDR:'<b>RDR</b> — rule disagreement rate (micro-averaged FP+FN over the union). Lower is better.'
  };
  let extra='';
  if(state.domain!=='All'){
    extra=' &nbsp;·&nbsp; Per-domain values extracted from the paper figures; <b>Avg</b> is the mean of L0–L2 for this domain.';
    if(state.protocol==='policy') extra+=' The GPT-OSS-Safeguard models are shown only under the per-rule protocol, where the paper provides their per-domain breakdowns.';
  } else {
    extra=' &nbsp;·&nbsp; <b>Avg</b> is the published case-weighted aggregate.';
  }
  document.getElementById('lb-foot').innerHTML = notes[metric]+extra;
}

// ============ mini charts ============
function bars(id, items, opt){
  opt=opt||{};
  const max = opt.max || Math.max(...items.map(i=>Math.abs(i.val)))*1.1;
  document.getElementById(id).innerHTML = items.map(it=>{
    const w = Math.max(3, Math.abs(it.val)/max*100)+'%';
    const txt = (it.prefix||'')+it.val.toFixed(1)+(opt.suffix||'');
    return `<div class="mbar-row">
      <div class="ml">${it.label}</div>
      <div class="mbar-track"><div class="mbar-fill" data-w="${w}" style="width:0;background:${it.color}">${txt}</div></div>
    </div>`;
  }).join('');
}
function buildCharts(){
  // 02 per-policy vs per-rule (avg RMR)
  bars('chart-perrule',[
    {label:'GPT-5.5 · policy',val:54.2,color:'var(--clay)'},
    {label:'GPT-5.5 · rule',val:55.5,color:'var(--blue)'},
    {label:'OSS-120B · policy',val:23.6,color:'var(--clay)'},
    {label:'OSS-120B · rule',val:52.4,color:'var(--blue)'},
    {label:'OSS-20B · policy',val:22.5,color:'var(--clay)'},
    {label:'OSS-20B · rule',val:44.5,color:'var(--blue)'},
  ],{max:60,suffix:'%'});

  // 03 reasoning effort ΔRMR (xhigh − low)
  bars('chart-effort',[
    {label:'L1 · ΔRMR',val:5.5,color:'var(--blue)',prefix:'+'},
    {label:'L2 · ΔRMR',val:14.5,color:'var(--clay)',prefix:'+'},
  ],{max:16,suffix:' pts'});

  // 05 dominant error source per level (qualitative)
  const errRows=[
    {lv:'L0',name:'Decisive rules',note:'often exceed 90% of errors',c:'var(--blue)',w:'90%'},
    {lv:'L1',name:'Exception rules',note:'models over-trigger exceptions',c:'var(--clay)',w:'66%'},
    {lv:'L2',name:'Conditional rules',note:'treated as violated rules themselves',c:'var(--green)',w:'58%'},
  ];
  document.getElementById('chart-errors').innerHTML =
    `<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px">`+
    errRows.map(r=>`<div style="border:1px solid var(--line);border-radius:10px;padding:16px 16px 14px;background:var(--bg-panel)">
      <div style="font-family:var(--mono);font-size:11px;letter-spacing:.1em;color:var(--ink-faint)">${r.lv} · DOMINANT ERROR</div>
      <div style="font-family:var(--serif);font-size:18px;font-weight:600;margin-top:6px">${r.name}</div>
      <div class="mbar-track" style="margin-top:12px;height:8px"><div class="mbar-fill" data-w="${r.w}" style="width:0;background:${r.c}"></div></div>
      <div style="font-size:12.5px;color:var(--ink-soft);margin-top:10px">${r.note}</div>
    </div>`).join('')+`</div>`;
}

// ============ reveal + animate bars on view ============
function animBars(scope){
  scope.querySelectorAll('.mbar-fill[data-w]').forEach(el=>{
    requestAnimationFrame(()=>{ el.style.width=el.dataset.w; });
  });
}
const io = new IntersectionObserver((es)=>{
  es.forEach(e=>{ if(e.isIntersecting){ e.target.classList.add('in'); animBars(e.target); io.unobserve(e.target); } });
},{threshold:0.12});

// ============ cite copy ============
document.getElementById('cite-copy').addEventListener('click',function(){
  const t=document.getElementById('cite-pre').innerText;
  navigator.clipboard.writeText(t).then(()=>{ this.textContent='✓ Copied'; setTimeout(()=>this.textContent='⧉ Copy',1600); });
});

// ============ init ============
render();
buildCharts();
document.querySelectorAll('.reveal').forEach(el=>io.observe(el));
})();
