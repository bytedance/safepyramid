/* ============ SafePyramid · jev.js — the Jev-1.13.0 spotlight section ============ */
(function(){
const J = window.JEV_DATA; if(!J) return;
const M = J.metrics, R = J.run;
// GPT-5.5 baseline = its per-rule leaderboard row (the paper's numbers), so the comparison matches the table above
const PR = window.SP_DATA.overallRule['GPT-5.5'];
const G = {L0:{}, L1:{}, L2:{}, ALL:{}};
[['L0',0],['L1',1],['L2',2],['ALL',3]].forEach(([k,i])=>{ G[k].rmr_avg=PR.RMR[i]; G[k].rmr_10=PR['RMR@1.0'][i]; G[k].rdr=PR.RDR[i]; });
const INK='#15171C', SOFT='#52565F', CAP='#6B6F78', LINE='rgba(21,23,28,0.12)', LINE_SOFT='rgba(21,23,28,0.07)';
const BLUE='#1C548F', GOLD='#B7822E', GOLD_LIGHT='#D6B77A';
const MONO="'IBM Plex Mono', ui-monospace, monospace", SANS="'IBM Plex Sans', system-ui, sans-serif";
const f1 = v => v.toFixed(1);

// ---- headline numbers ----
document.querySelectorAll('[data-jev]').forEach(el=>{
  const [lv,key]=el.dataset.jev.split('.'); el.textContent = f1(M[lv][key])+'%';
});
// cost, calls, tokens and wall-clock in metrics.json cover all K=3 sampling passes; show one pass, note the total
const perRun = R.wallclock_min / R.K, tot = R.wallclock_min;
const costPer = R.cost_usd / R.K, callsPer = R.calls / R.K, tokPer = R.input_tokens / R.K;
document.getElementById('jev-cost').textContent = '$'+costPer.toFixed(2);
document.getElementById('jev-cost-lab').innerHTML = `API cost for one pass over the entire benchmark &mdash; ${R.n_cases.toLocaleString()} conversation&ndash;policy cases, ${callsPer.toLocaleString()} calls, ${(tokPer/1e6).toFixed(1)}M input tokens. The reported K&nbsp;=&nbsp;${R.K} result takes ${R.K} passes: $${R.cost_usd.toFixed(2)}.`;
const fmtMin = m => m>=60 ? `${Math.floor(m/60)} h ${Math.round(m%60)} m` : `${Math.round(m)} min`;
document.getElementById('jev-stats').innerHTML = [
  [fmtMin(perRun), `wall-clock per pass · ×${R.K} = ${fmtMin(tot)}`],
  [f1(M.ALL.rmr_avg), 'RMR · per-rule, all levels'],
  [f1(M.ALL.refusal)+'%', 'refusal rate'],
].map(([v,k])=>`<div class="jev-stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join('');
document.getElementById('jev-exact').innerHTML =
  `Exact match: L0 ${f1(M.L0.rmr_10)} · L1 ${f1(M.L1.rmr_10)} · L2 ${f1(M.L2.rmr_10)} · all ${f1(M.ALL.rmr_10)} (95% CI ${f1(J.ci.ALL[0])}–${f1(J.ci.ALL[1])})`;
document.getElementById('jev-h2h-foot').textContent =
  `Against GPT-5.5's per-rule leaderboard row, Jev lands ${f1(+f1(G.ALL.rmr_avg)-(+f1(M.ALL.rmr_avg)))} RMR behind overall (${f1(M.ALL.rmr_avg)} vs ${f1(G.ALL.rmr_avg)}) — ahead on L0 and L2, behind on L1, where rule dependencies must be resolved.`;

// ---- tooltip ----
const tip = document.createElement('div'); tip.className='jev-tip'; document.body.appendChild(tip);
function bindTips(scope){
  scope.querySelectorAll('[data-tip]').forEach(el=>{
    el.addEventListener('mousemove',e=>{ tip.innerHTML=el.dataset.tip; tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY-10)+'px'; tip.classList.add('on'); });
    el.addEventListener('mouseleave',()=>tip.classList.remove('on'));
  });
}

// ---- chart helpers ----
const barPath=(x,yt,w,h,r=4)=>{ if(h<r) r=Math.max(h,.5); const yb=yt+h;
  return `M${x} ${yb} V${yt+r} Q${x} ${yt} ${x+r} ${yt} H${x+w-r} Q${x+w} ${yt} ${x+w} ${yt+r} V${yb} Z`; };
const svgOpen=(W,H,label)=>`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${label}">`;
const txt=(x,y,s,o={})=>`<text x="${x}" y="${y}" text-anchor="${o.a||'start'}" font-family="${o.f||MONO}" font-size="${o.s||10.5}" font-weight="${o.w||400}" fill="${o.c||SOFT}">${s}</text>`;

// 01 head-to-head grouped bars
function chartH2H(){
  const lv=['L0','L1','L2','ALL'], lab=['L0','L1','L2','Avg'];
  const W=500,H=230,top=26,bot=30,ph=H-top-bot,y0=top+ph,gw=(W-16)/4,bw=36,gap=2;
  let s=svgOpen(W,H,'RMR by level, Jev versus GPT-5.5 per-rule leaderboard numbers');
  [25,50,75].forEach(t=>{ const y=y0-ph*t/100; s+=`<line x1="8" x2="${W-8}" y1="${y}" y2="${y}" stroke="${LINE_SOFT}"/>`; });
  lv.forEach((k,i)=>{
    const j=M[k].rmr_avg, g=G[k].rmr_avg, cx=8+gw*i+gw/2, xj=cx-bw-gap/2, xg=cx+gap/2, hj=ph*j/100, hg=ph*g/100;
    s+=`<path d="${barPath(xj,y0-hj,bw,hj)}" fill="${GOLD}" data-tip="Jev-1.13.0 · ${lab[i]}<br>RMR ${f1(j)} · exact ${f1(M[k].rmr_10)} · RDR ${f1(M[k].rdr)}"/>`;
    s+=`<path d="${barPath(xg,y0-hg,bw,hg)}" fill="${BLUE}" data-tip="GPT-5.5 (per-rule leaderboard) · ${lab[i]}<br>RMR ${f1(g)} · exact ${f1(G[k].rmr_10)} · RDR ${f1(G[k].rdr)}"/>`;
    s+=txt(xj+bw/2,y0-hj-6,f1(j),{a:'middle',w:600,c:INK});
    s+=txt(xg+bw/2,y0-hg-6,f1(g),{a:'middle'});
    s+=txt(cx,y0+19,lab[i],{a:'middle',s:11});
  });
  s+=`<line x1="8" x2="${W-8}" y1="${y0}" y2="${y0}" stroke="${LINE}"/></svg>`;
  return s;
}
// 02 theta sweep
function chartTheta(){
  const pts=Object.entries(J.theta_sweep).map(([t,v])=>[parseFloat(t),v.rmr_10]).sort((a,b)=>a[0]-b[0]);
  const W=500,H=230,L=34,Rr=14,T=26,B=30;
  const X=t=>L+(t-0.2)/0.6*(W-L-Rr), Y=v=>T+(40-v)/30*(H-T-B);
  let s=svgOpen(W,H,'Exact match versus confidence threshold theta; peak at theta 0.5');
  [10,20,30,40].forEach(v=>{ s+=`<line x1="${L}" x2="${W-Rr}" y1="${Y(v)}" y2="${Y(v)}" stroke="${LINE_SOFT}"/>`+txt(L-6,Y(v)+3.5,v,{a:'end',s:10,c:CAP}); });
  [0.2,0.3,0.4,0.5,0.6,0.7,0.8].forEach(t=>{ s+=txt(X(t),H-10,t,{a:'middle'}); });
  s+=`<line x1="${X(0.5)}" x2="${X(0.5)}" y1="${T-4}" y2="${Y(10)}" stroke="${GOLD}" stroke-dasharray="3 3" opacity=".7"/>`;
  s+=`<path d="${pts.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)).join(' ')}" fill="none" stroke="${GOLD}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
  pts.forEach(([t,v])=>{ const big=Math.abs(t-0.5)<1e-9; const d=J.theta_sweep[String(t)];
    s+=`<circle cx="${X(t)}" cy="${Y(v)}" r="${big?5.5:4}" fill="${GOLD}" stroke="#fff" stroke-width="2"/>`;
    s+=`<circle cx="${X(t)}" cy="${Y(v)}" r="12" fill="transparent" data-tip="θ = ${t}<br>exact ${f1(d.rmr_10)} · RMR ${f1(d.rmr_avg)} · RDR ${f1(d.rdr)}"/>`; });
  s+=txt(X(0.5)+9,Y(33.8)-8,`${f1(J.theta_sweep['0.5'].rmr_10)} · θ = 0.5`,{s:11,w:600,c:INK});
  s+=txt(X(0.2)+2,Y(14.87)+16,f1(J.theta_sweep['0.2'].rmr_10),{s:10});
  s+=txt(X(0.8)-2,Y(16)+16,f1(J.theta_sweep['0.8'].rmr_10),{a:'end',s:10});
  return s+'</svg>';
}

document.getElementById('jev-chart-h2h').innerHTML = chartH2H();
document.getElementById('jev-chart-theta').innerHTML = chartTheta();
bindTips(document.getElementById('jev'));
})();
