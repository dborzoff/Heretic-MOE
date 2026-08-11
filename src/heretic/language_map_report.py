"""Dependency-free interactive HTML for frozen geometry trajectories."""

from __future__ import annotations

import base64
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


_SENSITIVE_KEYS = {"prompt", "response", "answer", "question", "text"}
_VERDICT_KEYS = {"trial_number", "row_id", "status", "confidence", "finalist"}
_VERDICT_STATES = {"success", "failure", "borderline", "unknown"}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _assert_no_sensitive_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _SENSITIVE_KEYS:
                raise ValueError(f"sensitive key in public geometry report: {key}")
            _assert_no_sensitive_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_sensitive_keys(child)


def _load_verdicts(path: Path) -> list[dict[str, object]]:
    verdicts = _read_jsonl(path)
    output: list[dict[str, object]] = []
    for verdict in verdicts:
        _assert_no_sensitive_keys(verdict)
        extra = set(verdict) - _VERDICT_KEYS
        if extra:
            raise ValueError(f"unsupported verdict keys: {sorted(extra)}")
        status = str(verdict.get("status", "unknown")).lower()
        if status not in _VERDICT_STATES:
            raise ValueError(f"unsupported verdict status: {status}")
        row: dict[str, object] = {
            "trial_number": int(verdict["trial_number"]),
            "row_id": str(verdict["row_id"]),
            "status": status,
        }
        if "confidence" in verdict:
            confidence = float(verdict["confidence"])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("verdict confidence must be within [0, 1]")
            row["confidence"] = confidence
        if "finalist" in verdict:
            row["finalist"] = str(verdict["finalist"])
        output.append(row)
    return output


def _data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return "data:application/octet-stream;base64," + encoded


def _checked_f32(path: Path, expected_values: int) -> str:
    if path.stat().st_size != expected_values * 4:
        raise ValueError(f"coordinate byte-size mismatch: {path.name}")
    values = np.fromfile(path, dtype="<f4")
    if values.size != expected_values or not bool(np.isfinite(values).all()):
        raise ValueError(f"invalid coordinate data: {path.name}")
    return _data_uri(path)


def _safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def write_interactive_geometry_report(
    package_dir: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Write one offline report with base points, trial points, and filters."""

    package_dir = Path(package_dir)
    output_path = Path(output_path) if output_path is not None else package_dir / "report.html"
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    base_index = json.loads((package_dir / "base_index.json").read_text(encoding="utf-8"))
    _assert_no_sensitive_keys(base_index)
    rows = int(manifest["rows"])
    layers = int(manifest["layers"])
    if len(base_index) != rows:
        raise ValueError("base index row count mismatch")
    base_path = package_dir / "base_points.f32"
    expected_hash = manifest["files"]["base_points.f32"]["sha256"]
    if _sha256(base_path) != expected_hash:
        raise ValueError("base coordinate hash mismatch")
    base_uri = _checked_f32(base_path, rows * layers * 3)

    trajectory_manifest_path = package_dir / "trajectory_manifest.json"
    trajectory_manifest = (
        json.loads(trajectory_manifest_path.read_text(encoding="utf-8"))
        if trajectory_manifest_path.is_file()
        else {"anchor_count": 0, "anchor_rows": [], "captured_trials": 0}
    )
    anchor_rows = [int(value) for value in trajectory_manifest.get("anchor_rows", [])]
    timeline = _read_jsonl(package_dir / "journal_trials.jsonl")
    trial_index = _read_jsonl(package_dir / "trial_index.jsonl")
    verdicts = _load_verdicts(package_dir / "verdicts.jsonl")
    _assert_no_sensitive_keys(timeline)
    _assert_no_sensitive_keys(trial_index)

    trial_uris: dict[str, str] = {}
    evaluation_uris: dict[str, str] = {}
    evaluation_indexes: dict[str, list[dict[str, object]]] = {}
    for entry in trial_index:
        trial_number = str(int(entry["trial_number"]))
        path = package_dir / str(entry["file"])
        if _sha256(path) != str(entry["sha256"]):
            raise ValueError(f"trial coordinate hash mismatch: {trial_number}")
        shape = [int(value) for value in entry["shape"]]
        if shape != [len(anchor_rows), layers, 3]:
            raise ValueError(f"trial coordinate shape mismatch: {trial_number}")
        trial_uris[trial_number] = _checked_f32(path, int(np.prod(shape)))
        if "evaluation_file" in entry:
            evaluation_path = package_dir / str(entry["evaluation_file"])
            if _sha256(evaluation_path) != str(entry["evaluation_sha256"]):
                raise ValueError(
                    f"evaluation coordinate hash mismatch: {trial_number}"
                )
            evaluation_shape = [int(value) for value in entry["evaluation_shape"]]
            if evaluation_shape[1:] != [layers, 3]:
                raise ValueError(
                    f"evaluation coordinate shape mismatch: {trial_number}"
                )
            evaluation_uris[trial_number] = _checked_f32(
                evaluation_path, int(np.prod(evaluation_shape))
            )
            evaluation_index_path = package_dir / str(entry["evaluation_index_file"])
            if _sha256(evaluation_index_path) != str(
                entry["evaluation_index_sha256"]
            ):
                raise ValueError(f"evaluation index hash mismatch: {trial_number}")
            evaluation_indexes[trial_number] = json.loads(
                evaluation_index_path.read_text(encoding="utf-8")
            )
            _assert_no_sensitive_keys(evaluation_indexes[trial_number])

    languages = sorted({str(row["language"]) for row in base_index})
    groups = sorted({str(row["group"]) for row in base_index})
    categories = sorted({str(row["category_id"]) for row in base_index})
    phases = sorted({str(row.get("phase", "search")) for row in timeline})
    finalists = sorted(
        {
            str(verdict["finalist"])
            for verdict in verdicts
            if verdict.get("finalist")
        }
    )
    metadata = {
        "rows": rows,
        "layers": layers,
        "base_index": base_index,
        "anchor_rows": anchor_rows,
        "timeline": timeline,
        "trial_index": trial_index,
        "verdicts": verdicts,
        "evaluation_indexes": evaluation_indexes,
        "languages": languages,
        "groups": groups,
        "categories": categories,
        "phases": phases,
        "finalists": finalists,
    }
    _assert_no_sensitive_keys(metadata)
    document = (
        _HTML_TEMPLATE.replace("__METADATA__", _safe_json(metadata))
        .replace("__BASE_URI__", _safe_json(base_uri))
        .replace("__TRIAL_URIS__", _safe_json(trial_uris))
        .replace("__EVALUATION_URIS__", _safe_json(evaluation_uris))
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(document, encoding="utf-8")
    os.replace(temporary, output_path)
    return {
        "status": "PASS",
        "base_rows": rows,
        "layers": layers,
        "captured_trials": len(trial_index),
        "bytes": output_path.stat().st_size,
        "sha256": _sha256(output_path),
        "output": str(output_path.resolve()),
    }


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Heretic-MOE 3D geometry</title>
<style>
:root{color-scheme:dark;font:14px system-ui;background:#080b12;color:#e8edf7}*{box-sizing:border-box}body{margin:0;display:grid;grid-template-columns:310px 1fr;height:100vh;overflow:hidden}aside{padding:14px;overflow:auto;background:#101522;border-right:1px solid #273149}h1{font-size:17px;margin:0 0 12px}h2{font-size:12px;color:#91a2bf;text-transform:uppercase;letter-spacing:.08em;margin:16px 0 6px}.row{display:flex;gap:8px;align-items:center;margin:6px 0;flex-wrap:wrap}label{display:inline-flex;gap:5px;align-items:center}input[type=range],select{width:100%}button,select{background:#182136;color:#e8edf7;border:1px solid #34415e;border-radius:5px;padding:6px}button{cursor:pointer}.swatch{width:10px;height:10px;border-radius:50%}.small{font-size:11px;color:#9aa8bd}main{position:relative;min-width:0}canvas{width:100%;height:100%;display:block;background:radial-gradient(circle at 50% 45%,#151e31,#070910 70%)}#hud{position:absolute;left:12px;top:10px;padding:7px 9px;background:#0a0e17cc;border:1px solid #29334a;border-radius:5px;pointer-events:none}.filter-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px}.legend{display:grid;gap:5px}.legend>div{display:flex;gap:7px;align-items:center}.line{width:24px;border-top:2px solid #d8dfef}.dashed{border-top-style:dashed}
</style></head><body>
<aside><h1>Heretic-MOE 3D geometry</h1><div class="small">Frozen Original-model axes. Drag to rotate, wheel to zoom.</div>
<h2>Layer</h2><input id="layer-filter" type="range" min="0" value="0"><div id="layer-value" class="small"></div>
<h2>Languages</h2><div id="language-filters" class="filter-grid"></div>
<h2>Groups</h2><div id="group-filters" class="filter-grid"></div>
<h2>Categories</h2><div id="category-filters" class="filter-grid"></div>
<h2>Search passes</h2><div id="phase-filters" class="filter-grid"></div>
<h2>Stage</h2><select id="stage-filter"><option value="all">Original + trials</option><option value="original">Original only</option><option value="trials">Trials only</option><option value="finalists">Finalists only</option></select>
<h2>Trial</h2><input id="trial-filter" type="range" min="0" max="0" value="0"><div id="trial-value" class="small"></div><button id="animate-trials">Animate trials</button>
<div class="row"><label><input id="show-evaluation" type="checkbox" checked>Evaluation points</label></div>
<h2>Finalist</h2><select id="finalist-filter"><option value="all">All finalists</option></select>
<h2>Verdict</h2><select id="verdict-filter"><option value="all">All</option><option value="success">Verified success</option><option value="failure">Failure</option><option value="borderline">Borderline</option><option value="unknown">Unverified</option></select>
<h2>Arrows</h2><select id="arrow-filter"><option value="centroids">Centroids + largest shifts</option><option value="all">All Original → Trial</option><option value="path">Dashed optimizer path only</option><option value="none">Hidden</option></select>
<div class="row"><label>Point size <input id="point-size" type="range" min="1" max="8" step=".5" value="3"></label></div><div class="row"><label>Opacity <input id="opacity" type="range" min=".1" max="1" step=".05" value=".72"></label></div>
<div class="row"><button id="reset-original">Original-only reset</button><button id="reset-camera">Reset camera</button></div>
<h2>Legend</h2><div class="legend"><div><i class="swatch" style="background:#4d9cff"></i>Original group A</div><div><i class="swatch" style="background:#ff5364"></i>Original group B</div><div><i class="swatch" style="background:#35df8d"></i>Verified success</div><div><i class="swatch" style="background:#f3c84b"></i>Unverified/borderline</div><div><i class="line"></i>Original → Trial</div><div><i class="line dashed"></i>optimizer path (independent edits)</div></div>
</aside><main><canvas id="scene"></canvas><div id="hud"></div></main>
<script>
"use strict";
const META=__METADATA__,BASE_URI=__BASE_URI__,TRIAL_URIS=__TRIAL_URIS__,EVALUATION_URIS=__EVALUATION_URIS__;
function floats(uri){const raw=atob(uri.slice(uri.indexOf(',')+1)),u=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)u[i]=raw.charCodeAt(i);return new Float32Array(u.buffer)}
const BASE=floats(BASE_URI),TRIALS={},EVALUATIONS={};for(const [k,v] of Object.entries(TRIAL_URIS))TRIALS[k]=floats(v);for(const [k,v] of Object.entries(EVALUATION_URIS))EVALUATIONS[k]=floats(v);
const canvas=document.getElementById('scene'),ctx=canvas.getContext('2d'),hud=document.getElementById('hud');let yaw=-.55,pitch=.35,zoom=1,panX=0,panY=0,drag=null,timer=null;
const byId=id=>document.getElementById(id), layer=byId('layer-filter'),trial=byId('trial-filter');layer.max=Math.max(0,META.layers-1);const captured=META.trial_index.map(x=>x.trial_number).sort((a,b)=>a-b);trial.max=Math.max(0,captured.length-1);
function addChecks(id,values){const root=byId(id);for(const value of values){const label=document.createElement('label'),box=document.createElement('input');box.type='checkbox';box.checked=true;box.value=value;box.addEventListener('change',render);label.append(box,document.createTextNode(value));root.append(label)}}
addChecks('language-filters',META.languages);addChecks('group-filters',META.groups);addChecks('category-filters',META.categories);addChecks('phase-filters',META.phases);
for(const value of META.finalists){const o=document.createElement('option');o.value=o.textContent=value;byId('finalist-filter').append(o)}
function checked(id){return new Set([...byId(id).querySelectorAll('input:checked')].map(x=>x.value))}function currentTrial(){return captured[+trial.value]??null}function coord(array,row,l){const p=(row*META.layers+l)*3;return[array[p],array[p+1],array[p+2]]}
function verdict(t,row){const hit=META.verdicts.find(v=>v.trial_number===t&&v.row_id===row);return hit?.status||'unknown'}
function visible(row,langs,groups,cats){return langs.has(row.language)&&groups.has(row.group)&&cats.has(row.category_id)}
function rotate(p){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);const x=p[0]*cy-p[2]*sy,z=p[0]*sy+p[2]*cy;return[x,z*sp+p[1]*cp,z*cp-p[1]*sp]}
function resize(){const d=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;if(canvas.width!==w*d||canvas.height!==h*d){canvas.width=w*d;canvas.height=h*d;ctx.setTransform(d,0,0,d,0,0)}}
function render(){resize();const w=canvas.clientWidth,h=canvas.clientHeight,l=+layer.value,t=currentTrial(),langs=checked('language-filters'),groups=checked('group-filters'),cats=checked('category-filters'),phases=checked('phase-filters'),stage=byId('stage-filter').value,vfilter=byId('verdict-filter').value,arrows=byId('arrow-filter').value,size=+byId('point-size').value,alpha=+byId('opacity').value;ctx.clearRect(0,0,w,h);byId('layer-value').textContent=`Layer ${l+1} / ${META.layers}`;byId('trial-value').textContent=t===null?'No captured trials':`Trial ${t}`;
let scale=55*zoom,ox=w/2+panX,oy=h/2+panY;const project=p=>{const q=rotate(p);return[ox+q[0]*scale,oy-q[1]*scale,q[2]]};const basePoints=[],trialPoints=[];
if(stage==='all'||stage==='original')for(let i=0;i<META.rows;i++){const row=META.base_index[i];if(visible(row,langs,groups,cats)){const q=project(coord(BASE,i,l));basePoints.push({q,color:row.group==='A'?'#4d9cff':'#ff5364',row})}}
const entry=META.trial_index.find(x=>x.trial_number===t),phase=entry?.phase||META.timeline.find(x=>x.trial_number===t)?.phase||'search';if(t!==null&&TRIALS[t]&&phases.has(phase)&&(stage==='all'||stage==='trials')){const arr=TRIALS[t];for(let a=0;a<META.anchor_rows.length;a++){const baseRow=META.anchor_rows[a],row=META.base_index[baseRow];if(!visible(row,langs,groups,cats))continue;const status=verdict(t,row.row_id);if(vfilter!=='all'&&status!==vfilter)continue;const from=project(coord(BASE,baseRow,l)),to=project(coord(arr,a,l));trialPoints.push({from,to,row,status,color:status==='success'?'#35df8d':'#f3c84b'});}}
if(t!==null&&EVALUATIONS[t]&&byId('show-evaluation').checked&&phases.has(phase)&&(stage==='all'||stage==='trials')){const arr=EVALUATIONS[t],rows=META.evaluation_indexes[t]||[];for(let i=0;i<rows.length;i++){const status=verdict(t,rows[i].prompt_sha256);if(vfilter!=='all'&&status!==vfilter)continue;const to=project(coord(arr,i,l));trialPoints.push({from:to,to,row:{row_id:rows[i].prompt_sha256},status,color:status==='success'?'#35df8d':'#f3c84b',evaluation:true})}}
ctx.globalAlpha=.45;if(arrows==='all'||arrows==='centroids'){ctx.setLineDash([]);for(const p of trialPoints){if(p.evaluation)continue;ctx.strokeStyle=p.color;ctx.beginPath();ctx.moveTo(p.from[0],p.from[1]);ctx.lineTo(p.to[0],p.to[1]);ctx.stroke()}}
if((arrows==='path'||arrows==='centroids')&&captured.length>1){ctx.setLineDash([6,5]);ctx.strokeStyle='#d8dfef';ctx.beginPath();let first=true;for(const n of captured){const e=META.trial_index.find(x=>x.trial_number===n);if(!phases.has(e?.phase||'search'))continue;const arr=TRIALS[n];let c=[0,0,0];for(let a=0;a<META.anchor_rows.length;a++){const p=coord(arr,a,l);c[0]+=p[0];c[1]+=p[1];c[2]+=p[2]}c=c.map(x=>x/META.anchor_rows.length);const q=project(c);first?(ctx.moveTo(q[0],q[1]),first=false):ctx.lineTo(q[0],q[1])}ctx.stroke();ctx.setLineDash([])}
ctx.globalAlpha=alpha;const points=[...basePoints,...trialPoints.map(p=>({q:p.to,color:p.color,row:p.row}))].sort((a,b)=>a.q[2]-b.q[2]);for(const p of points){ctx.fillStyle=p.color;ctx.beginPath();ctx.arc(p.q[0],p.q[1],size,0,Math.PI*2);ctx.fill()}ctx.globalAlpha=1;hud.textContent=`Layer ${l+1} · Trial ${t??'Original'} · ${points.length} visible points`;}
for(const id of ['layer-filter','trial-filter','stage-filter','finalist-filter','verdict-filter','arrow-filter','point-size','opacity','show-evaluation'])byId(id).addEventListener('input',render);
byId('animate-trials').onclick=()=>{if(timer){clearInterval(timer);timer=null;byId('animate-trials').textContent='Animate trials';return}byId('animate-trials').textContent='Stop animation';timer=setInterval(()=>{trial.value=(+trial.value+1)%Math.max(captured.length,1);render()},700)};
byId('reset-original').onclick=()=>{byId('stage-filter').value='original';if(timer){clearInterval(timer);timer=null}render()};byId('reset-camera').onclick=()=>{yaw=-.55;pitch=.35;zoom=1;panX=panY=0;render()};
canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,button:e.button};canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;if(drag.button===2){panX+=dx;panY+=dy}else{yaw+=dx*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*.008))}render()};canvas.onpointerup=()=>drag=null;canvas.oncontextmenu=e=>e.preventDefault();canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.15,Math.min(12,zoom*Math.exp(-e.deltaY*.001)));render()};window.onresize=render;render();
</script></body></html>
'''
