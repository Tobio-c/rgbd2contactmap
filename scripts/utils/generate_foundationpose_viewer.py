#!/usr/bin/env python3
import ast
import base64
import json
import math
import pathlib
import struct


SRC = pathlib.Path("/tmp/contactmap_cjq_repro_20260603_200212/stage_k_foundationpose_object_pose")
OUT = pathlib.Path("/home/originflow/project/contactmap_cjq/foundationpose_object_pose_viewer.html")


def read_npy_matrix(path: pathlib.Path):
    data = path.read_bytes()
    if not data.startswith(b"\x93NUMPY"):
        raise ValueError("not a npy file")
    major = data[6]
    if major == 1:
        header_len = struct.unpack("<H", data[8:10])[0]
        offset = 10
    else:
        header_len = struct.unpack("<I", data[8:12])[0]
        offset = 12
    header = ast.literal_eval(data[offset : offset + header_len].decode("latin1").strip())
    dtype = header["descr"]
    shape = header["shape"]
    if dtype in ("<f8", "|f8", "f8"):
        fmt = "d"
        width = 8
    elif dtype in ("<f4", "|f4", "f4"):
        fmt = "f"
        width = 4
    else:
        raise ValueError(f"unsupported dtype {dtype}")
    count = math.prod(shape)
    values = struct.unpack(
        "<" + fmt * count,
        data[offset + header_len : offset + header_len + width * count],
    )
    rows, cols = shape
    return [list(values[i * cols : (i + 1) * cols]) for i in range(rows)]


def ply_scalar_size(kind: str) -> int:
    return {
        "char": 1,
        "uchar": 1,
        "int8": 1,
        "uint8": 1,
        "short": 2,
        "ushort": 2,
        "int16": 2,
        "uint16": 2,
        "int": 4,
        "uint": 4,
        "int32": 4,
        "uint32": 4,
        "float": 4,
        "float32": 4,
        "double": 8,
        "float64": 8,
    }[kind]


def unpack_scalar(data: bytes, offset: int, kind: str):
    fmts = {
        "char": "b",
        "uchar": "B",
        "int8": "b",
        "uint8": "B",
        "short": "h",
        "ushort": "H",
        "int16": "h",
        "uint16": "H",
        "int": "i",
        "uint": "I",
        "int32": "i",
        "uint32": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
        "float64": "d",
    }
    return struct.unpack_from("<" + fmts[kind], data, offset)[0]


def read_ply_mesh(path: pathlib.Path):
    data = path.read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii", errors="replace").splitlines()
    elements = []
    current = None
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current)
        elif parts[0] == "property" and current is not None:
            if parts[1] == "list":
                current["properties"].append(("list", parts[2], parts[3], parts[4]))
            else:
                current["properties"].append(("scalar", parts[1], parts[2]))

    offset = header_end
    vertices = []
    faces = []

    for element in elements:
        for _ in range(element["count"]):
            record = {}
            for prop in element["properties"]:
                if prop[0] == "scalar":
                    _, kind, name = prop
                    value = unpack_scalar(data, offset, kind)
                    offset += ply_scalar_size(kind)
                    record[name] = value
                else:
                    _, count_kind, item_kind, name = prop
                    n = int(unpack_scalar(data, offset, count_kind))
                    offset += ply_scalar_size(count_kind)
                    values = []
                    for _ in range(n):
                        values.append(unpack_scalar(data, offset, item_kind))
                        offset += ply_scalar_size(item_kind)
                    record[name] = values
            if element["name"] == "vertex":
                vertices.append([record["x"], record["y"], record["z"]])
            elif element["name"] == "face":
                indices = record.get("vertex_indices", [])
                if len(indices) >= 3:
                    first = int(indices[0])
                    for i in range(1, len(indices) - 1):
                        faces.append([first, int(indices[i]), int(indices[i + 1])])

    return vertices, faces, header


def bounds(vertices):
    mins = [min(v[i] for v in vertices) for i in range(3)]
    maxs = [max(v[i] for v in vertices) for i in range(3)]
    center = [(mins[i] + maxs[i]) / 2 for i in range(3)]
    size = [maxs[i] - mins[i] for i in range(3)]
    return {"min": mins, "max": maxs, "center": center, "size": size}


def matrix_info(matrix):
    rot = [row[:3] for row in matrix[:3]]
    trans = [matrix[i][3] for i in range(3)]
    det = (
        rot[0][0] * (rot[1][1] * rot[2][2] - rot[1][2] * rot[2][1])
        - rot[0][1] * (rot[1][0] * rot[2][2] - rot[1][2] * rot[2][0])
        + rot[0][2] * (rot[1][0] * rot[2][1] - rot[1][1] * rot[2][0])
    )
    return {"translation": trans, "determinant": det}


def main():
    image_path = SRC / "rgb_sam_mask_foundationpose_bbox_overlay.png"
    ply_path = SRC / "foundationpose_posed_cad_object_mesh_ref.ply"
    npy_path = SRC / "object_pose_cam.npy"

    vertices, faces, header = read_ply_mesh(ply_path)
    matrix = read_npy_matrix(npy_path)
    payload = {
        "sourceDir": str(SRC),
        "imageData": "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii"),
        "vertices": vertices,
        "faces": faces,
        "pose": matrix,
        "poseInfo": matrix_info(matrix),
        "meshInfo": {
            "vertexCount": len(vertices),
            "faceCount": len(faces),
            "bounds": bounds(vertices),
            "header": header,
        },
    }

    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    OUT.write_text(html, encoding="utf-8")
    print(OUT)


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FoundationPose Object Pose Viewer</title>
<style>
:root{color-scheme:dark;--bg:#101214;--panel:#181b1f;--panel2:#20242a;--text:#eef2f5;--muted:#a9b1ba;--line:#343a42;--accent:#43b3a6;--warn:#e6b450}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{height:56px;display:flex;align-items:center;justify-content:space-between;padding:0 18px;border-bottom:1px solid var(--line);background:#121519}
h1{font-size:18px;margin:0;font-weight:650;letter-spacing:0}.src{color:var(--muted);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:58vw}
main{display:grid;grid-template-columns:minmax(320px,38vw) minmax(360px,1fr) 360px;min-height:calc(100vh - 56px)}
section{border-right:1px solid var(--line);min-width:0}.panel-head{height:44px;display:flex;align-items:center;justify-content:space-between;padding:0 14px;border-bottom:1px solid var(--line);background:var(--panel)}
.panel-head h2{font-size:14px;margin:0;font-weight:650}.meta{color:var(--muted);font-size:12px}.image-wrap{height:calc(100vh - 100px);display:grid;place-items:center;padding:14px;background:#0d0f12}
.image-wrap img{max-width:100%;max-height:100%;object-fit:contain;border:1px solid var(--line);background:#000}
#viewer{height:calc(100vh - 100px);position:relative;background:radial-gradient(circle at 50% 35%,#252b31 0,#111418 56%,#0b0d10 100%)}
canvas{display:block;width:100%;height:100%}.toolbar{position:absolute;left:12px;top:12px;display:flex;gap:8px;z-index:2}
button{height:32px;border:1px solid var(--line);background:#242a31;color:var(--text);border-radius:6px;padding:0 10px;cursor:pointer}button:hover{border-color:var(--accent)}
.hint{position:absolute;left:12px;bottom:10px;color:#c2c9d0;font-size:12px;background:rgba(16,18,20,.72);padding:6px 8px;border:1px solid rgba(255,255,255,.08);border-radius:6px}
aside{min-width:0;background:#121519}.content{padding:14px;display:grid;gap:14px;max-height:calc(100vh - 100px);overflow:auto}
.block{border:1px solid var(--line);background:var(--panel);border-radius:8px;overflow:hidden}.block h3{margin:0;padding:10px 12px;font-size:13px;border-bottom:1px solid var(--line);background:var(--panel2)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{border-bottom:1px solid var(--line);padding:8px 10px;text-align:right}th{color:var(--muted);font-weight:600}td:first-child,th:first-child{text-align:left;color:var(--muted)}
.kv{display:grid;grid-template-columns:1fr auto;gap:6px 12px;padding:10px 12px;font-variant-numeric:tabular-nums}.kv div:nth-child(odd){color:var(--muted)}pre{margin:0;padding:10px 12px;white-space:pre-wrap;color:#d7dde3;font-size:12px}
@media (max-width:1100px){main{grid-template-columns:1fr}section,aside{border-right:0;border-bottom:1px solid var(--line)}.image-wrap,#viewer{height:58vh}.src{display:none}}
</style>
</head>
<body>
<header><h1>FoundationPose Object Pose Viewer</h1><div class="src" id="src"></div></header>
<main>
  <section>
    <div class="panel-head"><h2>RGB / SAM Mask / BBox Overlay</h2><span class="meta">640 x 480 PNG</span></div>
    <div class="image-wrap"><img id="overlay" alt="rgb_sam_mask_foundationpose_bbox_overlay.png"></div>
  </section>
  <section>
    <div class="panel-head"><h2>Posed CAD Object Mesh</h2><span class="meta" id="meshMeta"></span></div>
    <div id="viewer">
      <div class="toolbar"><button id="reset">Reset</button><button id="toggleWire">Wire</button></div>
      <canvas id="gl"></canvas>
      <div class="hint">Drag rotate · Wheel zoom · Shift+drag pan</div>
    </div>
  </section>
  <aside>
    <div class="panel-head"><h2>Object Pose in Camera</h2><span class="meta">object_pose_cam.npy</span></div>
    <div class="content">
      <div class="block"><h3>4 x 4 Matrix</h3><div id="poseTable"></div></div>
      <div class="block"><h3>Summary</h3><div class="kv" id="summary"></div></div>
      <div class="block"><h3>PLY Header</h3><pre id="headerText"></pre></div>
    </div>
  </aside>
</main>
<script>
const DATA=__PAYLOAD__;
document.getElementById('src').textContent=DATA.sourceDir;
document.getElementById('overlay').src=DATA.imageData;
document.getElementById('meshMeta').textContent=`${DATA.meshInfo.vertexCount} vertices · ${DATA.meshInfo.faceCount} faces`;
document.getElementById('headerText').textContent=DATA.meshInfo.header.join('\n');

function fmt(n){return Number(n).toFixed(6)}
function renderPose(){
  const rows=DATA.pose.map((r,i)=>`<tr><th>r${i}</th>${r.map(v=>`<td>${fmt(v)}</td>`).join('')}</tr>`).join('');
  document.getElementById('poseTable').innerHTML=`<table><tbody>${rows}</tbody></table>`;
  const b=DATA.meshInfo.bounds, p=DATA.poseInfo;
  document.getElementById('summary').innerHTML=[
    ['tx',fmt(p.translation[0])],['ty',fmt(p.translation[1])],['tz',fmt(p.translation[2])],
    ['rotation det',fmt(p.determinant)],['mesh min',b.min.map(fmt).join(', ')],['mesh max',b.max.map(fmt).join(', ')],
    ['mesh size',b.size.map(fmt).join(', ')]
  ].map(([k,v])=>`<div>${k}</div><div>${v}</div>`).join('');
}
renderPose();

const canvas=document.getElementById('gl'), gl=canvas.getContext('webgl',{antialias:true});
if(!gl){throw new Error('WebGL unavailable')}
const vs=`attribute vec3 p;attribute vec3 n;uniform mat4 mvp;uniform mat4 model;varying vec3 vn;varying vec3 wp;void main(){wp=(model*vec4(p,1.0)).xyz;vn=mat3(model)*n;gl_Position=mvp*vec4(p,1.0);}`;
const fs=`precision mediump float;varying vec3 vn;varying vec3 wp;uniform bool wire;void main(){vec3 N=normalize(vn);vec3 L=normalize(vec3(.45,.75,.55));float d=max(dot(N,L),0.0);vec3 base=wire?vec3(.95,.78,.28):vec3(.27,.70,.65);vec3 col=base*(0.28+0.72*d)+vec3(.05,.06,.07);gl_FragColor=vec4(col,1.0);}`;
function shader(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s}
const prog=gl.createProgram();gl.attachShader(prog,shader(gl.VERTEX_SHADER,vs));gl.attachShader(prog,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(prog);gl.useProgram(prog);

const verts=DATA.vertices, faces=DATA.faces, pos=[], norms=[], wire=[];
for(const f of faces){
  const a=verts[f[0]],b=verts[f[1]],c=verts[f[2]];
  const ux=b[0]-a[0],uy=b[1]-a[1],uz=b[2]-a[2],vx=c[0]-a[0],vy=c[1]-a[1],vz=c[2]-a[2];
  let nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx; const l=Math.hypot(nx,ny,nz)||1; nx/=l;ny/=l;nz/=l;
  for(const q of [a,b,c]){pos.push(q[0],q[1],q[2]);norms.push(nx,ny,nz)}
  for(const e of [[a,b],[b,c],[c,a]]) wire.push(...e[0],...e[1]);
}
function buffer(data,attr,size){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.STATIC_DRAW);const loc=gl.getAttribLocation(prog,attr);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,0,0);return b}
const posBuf=buffer(pos,'p',3), normBuf=buffer(norms,'n',3), wireBuf=gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER,wireBuf);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(wire),gl.STATIC_DRAW);
const pLoc=gl.getAttribLocation(prog,'p'), nLoc=gl.getAttribLocation(prog,'n'), mvpLoc=gl.getUniformLocation(prog,'mvp'), modelLoc=gl.getUniformLocation(prog,'model'), wireLoc=gl.getUniformLocation(prog,'wire');
let rx=-0.65, ry=0.75, zoom=2.4, pan=[0,0], showWire=false, dragging=false, last=[0,0], mode='rot';
document.getElementById('reset').onclick=()=>{rx=-0.65;ry=0.75;zoom=2.4;pan=[0,0];draw()};
document.getElementById('toggleWire').onclick=()=>{showWire=!showWire;draw()};
canvas.addEventListener('pointerdown',e=>{dragging=true;last=[e.clientX,e.clientY];mode=e.shiftKey?'pan':'rot';canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{if(!dragging)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];last=[e.clientX,e.clientY];if(mode==='pan'){pan[0]+=dx/canvas.clientWidth*2;pan[1]-=dy/canvas.clientHeight*2}else{ry+=dx*.01;rx+=dy*.01}draw()});
canvas.addEventListener('pointerup',()=>dragging=false);
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom*=Math.exp(e.deltaY*.001);zoom=Math.max(.25,Math.min(20,zoom));draw()},{passive:false});

function matMul(a,b){const r=new Array(16).fill(0);for(let c=0;c<4;c++)for(let r0=0;r0<4;r0++)for(let k=0;k<4;k++)r[c*4+r0]+=a[k*4+r0]*b[c*4+k];return r}
function persp(fovy,asp,n,f){const t=1/Math.tan(fovy/2);return [t/asp,0,0,0,0,t,0,0,0,0,(f+n)/(n-f),-1,0,0,2*f*n/(n-f),0]}
function trans(x,y,z){return [1,0,0,0,0,1,0,0,0,0,1,0,x,y,z,1]}
function rotx(a){const c=Math.cos(a),s=Math.sin(a);return [1,0,0,0,0,c,s,0,0,-s,c,0,0,0,0,1]}
function roty(a){const c=Math.cos(a),s=Math.sin(a);return [c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1]}
function scale(s){return [s,0,0,0,0,s,0,0,0,0,s,0,0,0,0,1]}
function resize(){const dpr=Math.min(devicePixelRatio||1,2),w=Math.floor(canvas.clientWidth*dpr),h=Math.floor(canvas.clientHeight*dpr);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;gl.viewport(0,0,w,h)}}
function draw(){
  resize(); gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST); gl.enable(gl.CULL_FACE);
  const b=DATA.meshInfo.bounds, maxSize=Math.max(...b.size)||1, s=1.8/maxSize, c=b.center;
  const model=matMul(matMul(roty(ry),rotx(rx)),matMul(scale(s),trans(-c[0],-c[1],-c[2])));
  const view=trans(pan[0],pan[1],-zoom), proj=persp(Math.PI/4,canvas.width/canvas.height,.01,100), mvp=matMul(proj,matMul(view,model));
  gl.uniformMatrix4fv(modelLoc,false,new Float32Array(model)); gl.uniformMatrix4fv(mvpLoc,false,new Float32Array(mvp));
  if(showWire){
    gl.disable(gl.CULL_FACE); gl.uniform1i(wireLoc,1); gl.bindBuffer(gl.ARRAY_BUFFER,wireBuf); gl.vertexAttribPointer(pLoc,3,gl.FLOAT,false,0,0); gl.disableVertexAttribArray(nLoc); gl.vertexAttrib3f(nLoc,0,0,1); gl.drawArrays(gl.LINES,0,wire.length/3); gl.enableVertexAttribArray(nLoc);
  } else {
    gl.uniform1i(wireLoc,0); gl.bindBuffer(gl.ARRAY_BUFFER,posBuf); gl.vertexAttribPointer(pLoc,3,gl.FLOAT,false,0,0); gl.bindBuffer(gl.ARRAY_BUFFER,normBuf); gl.vertexAttribPointer(nLoc,3,gl.FLOAT,false,0,0); gl.drawArrays(gl.TRIANGLES,0,pos.length/3);
  }
}
addEventListener('resize',draw); draw();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
