"""Rebuild README figures from editable FigureSpec JSON without loading a model.

Requires the ARIS figure-spec renderer (pass --renderer); optional --edge creates
PNG/PDF previews with an isolated headless Edge profile, not the user's profile.
Project extensions: edges.via (orthogonal waypoints), edges.arrow (bool), panels.
No training inputs, model files or active run contracts are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
INK = '#19191D'
MUTED = '#626773'
BLUE = ('#BDD7EC', '#477AB5')
GREEN = ('#DEEDCE', '#648C45')
PURPLE = ('#DCC8ED', '#8355AD')
GOLD = ('#FFE7A1', '#AE812D')
PEACH = ('#F5BC96', '#BA7550')
GRAY = ('#DCDDDD', '#93989D')
SERIF = 'Times New Roman, Georgia, Noto Serif, serif'
SANS = 'Arial, Microsoft YaHei, Noto Sans CJK SC, sans-serif'


class Figure:
    def __init__(self, title, subtitle, width=1800, height=1100):
        self.spec = dict(title=title, canvas=dict(width=width, height=height),
            style=dict(font_family=SANS,
                       font_size=22, bg_color='#FFFFFF',
                       palette=[INK, MUTED, BLUE[1], GREEN[1], PURPLE[1], GOLD[1], PEACH[1], GRAY[1]]),
            nodes=[], edges=[], groups=[], labels=[], panels=[])
        self.label(title, 42, 52, size=36)
        self.label(subtitle, 42, 91, color=MUTED)

    def node(self, name, text, x, y, width=260, height=66, color=BLUE, shape='rounded'):
        self.spec['nodes'].append(dict(id=name, label=text, x=x, y=y, width=width,
            height=height, shape=shape, fill=color[0], stroke=color[1], text_color=INK, font_size=22))
        return name

    def label(self, text, x, y, size=22, color=INK, anchor='start'):
        self.spec['labels'].append(dict(text=text, x=x, y=y, font_size=size, color=color, anchor=anchor))

    def edge(self, source, target, via=None, dashed=False, color=INK, arrow=True):
        item = dict(**{'from': source, 'to': target}, color=color,
                    style='dashed' if dashed else 'solid', thickness=3, arrow=arrow)
        if via:
            item['via'] = via
        self.spec['edges'].append(item)

    def chain(self, *names):
        for a, b in zip(names, names[1:]):
            self.edge(a, b)

    def group(self, ids, color=GRAY, padding=28):
        tint={BLUE[0]:'#EDF4FA', GREEN[0]:'#EFF6E9', PURPLE[0]:'#F2EDF8',
              GOLD[0]:'#FFF7DC', PEACH[0]:'#FBEDE1', GRAY[0]:'#F4F4F4'}[color[0]]
        self.spec['groups'].append(dict(node_ids=ids, fill=tint, stroke=tint, padding=padding))

    def panel(self, x, y, width, height, fill='#F4F4F4', shadow=True):
        self.spec['panels'].append(dict(x=x,y=y,width=width,height=height,fill=fill,shadow=shadow))

    def footer(self, text):
        self.label(text, 42, self.spec['canvas']['height']-30, color=MUTED)


def dense():
    f = Figure('EvoMind · Dense language model',
               'Current model: 63.91M parameters · 8 layers · hidden 768 · vocabulary 6,400', 1840, 1160)
    f.label('(a) Autoregressive LM', 48, 145)
    # Whole model: bottom-to-top, matching the reference figure's reading order.
    for name, text, y, col in [
        ('text','Input text',995,GOLD),('tok','BPE Tokenizer',893,GOLD),
        ('emb','Input Embedding\n6,400 × 768',778,BLUE),
        ('blocks','Transformer Block\n× 8',650,PURPLE),
        ('finalnorm','Final RMSNorm',522,GRAY),('head','LM Head\n768 → 6,400',407,BLUE),
        ('sample','Softmax + sampling',292,PEACH),('decode','Decode next token',190,GOLD)]:
        f.node(name,text,205,y,width=282,height=80,color=col)
    f.chain('text','tok','emb','blocks','finalnorm','head','sample','decode')
    f.edge('head','emb',via=[[370,407],[370,778]],dashed=True,arrow=False,color=MUTED)
    f.label('Embedding ↔ LM Head: tied weights',48,1068,color=MUTED)

    f.label('(b) Pre-Norm Block',435,145)
    for name,text,y,col in [('x','x',990,GRAY),('n1','RMSNorm 1',870,GRAY),
                           ('att','GQA attention',750,GREEN),('a1','+',635,GRAY),
                           ('n2','RMSNorm 2',515,GRAY),('ffn','SwiGLU FFN',395,PURPLE),
                           ('a2','+',280,GRAY),('out','Next block',190,GRAY)]:
        is_add=name in ('a1','a2')
        f.node(name,text,565,y,width=46 if is_add else 230,height=46 if is_add else 64,
               color=col,shape='circle' if is_add else 'rounded')
    f.chain('x','n1','att','a1','n2','ffn','a2','out')
    f.edge('x','a1',via=[[738,990],[738,635]])
    f.edge('a1','a2',via=[[565,595],[710,595],[710,280]])
    f.label('skip x',724,940,anchor='end',color=MUTED)
    f.label('skip h',697,580,anchor='end',color=MUTED)

    f.label('(c) GQA',805,145)
    f.node('ga_in','Normalized input',1050,995,width=350,color=GRAY)
    f.node('q','Q\n8 × 96',865,875,width=140,height=82)
    f.node('k','K\n4 × 96',1045,875,width=140,height=82)
    f.node('v','V\n4 × 96',1225,875,width=140,height=82)
    for x,name,text in [(865,'qrope','Q Norm\n+ RoPE'),(1045,'krope','K Norm\n+ RoPE')]:
        f.node(name,text,x,746,width=140,height=82,color=GRAY)
    f.node('kr','Repeat K × 2',1045,628,width=156,color=GREEN)
    f.node('vr','Repeat V × 2',1225,628,width=156,color=GREEN)
    f.node('scores','QKᵀ / √96\n+ causal mask',995,510,width=270,height=82,color=GREEN)
    f.node('soft','Softmax over keys',995,389,width=270,color=PEACH)
    f.node('av','Attention weights × V',1050,287,width=350,color=GREEN)
    f.node('op','Concat → Linear',1050,190,width=395)
    f.edge('ga_in','q',via=[[865,950]])
    f.edge('ga_in','k');f.edge('ga_in','v',via=[[1225,950]])
    f.chain('q','qrope');f.chain('k','krope','kr');f.edge('v','vr')
    f.edge('qrope','scores',via=[[865,562]])
    f.edge('kr','scores');f.chain('scores','soft','av','op')
    f.edge('vr','av',via=[[1225,574],[1282,574],[1282,287]])
    f.label('Cache: 4 KV heads before repeat',803,1068,color=MUTED)

    f.label('(d) SwiGLU FFN',1374,145)
    f.node('ff_in','Normalized input\n768 dimensions',1590,995,width=330,height=80,color=GRAY)
    f.node('up','Up projection\n768 → 2,432',1478,818,width=198,height=82)
    f.node('gate','Gate projection\n768 → 2,432',1702,818,width=198,height=82)
    f.node('silu','SiLU',1702,652,width=198,color=GOLD)
    f.node('multiply','⊙',1590,499,width=56,height=56,shape='circle',color=GRAY)
    f.node('down','Down projection\n2,432 → 768',1590,346,width=330,height=82)
    f.node('ff_out','FFN output\n768 dimensions',1590,190,width=330,height=80,color=PURPLE)
    f.edge('ff_in','up');f.edge('ff_in','gate');f.chain('gate','silu','multiply','down','ff_out')
    f.edge('up','multiply',via=[[1478,499]])
    f.label('⊙  element-wise multiplication',1375,1068,color=MUTED)
    f.group(['ga_in','q','k','v','qrope','krope','kr','vr','scores','soft','av','op'],GREEN,18)
    f.group(['ff_in','up','gate','silu','multiply','down','ff_out'],PURPLE,18)
    f.panel(43,239,335,600,fill='#F5F5F5')
    f.panel(416,160,345,880,fill='#F0E5F8')
    f.footer('Architecture schematic · training returns logits (CE applies log-softmax internally); sampling is the inference path.')
    return f.spec


def moe():
    f=Figure('EvoMind · Mixture-of-Experts Architecture',
             'Four routed SwiGLU experts · Top-1 activation · same GQA attention backbone.',1600,900)
    f.label('(a) Transformer block',60,150)
    for name,text,y,col in [('x','Input x',764,GRAY),('att','RMSNorm → GQA',645,GREEN),
                           ('add1','+ x',533,GRAY),('norm','RMSNorm',426,GRAY),
                           ('moe','MoE FFN',318,PURPLE),('add2','+ residual',208,GRAY)]:
        f.node(name,text,240,y,width=320,height=72,color=col)
    f.chain('x','att','add1','norm','moe','add2')
    f.edge('x','add1',via=[[445,764],[445,533]])
    f.edge('add1','add2',via=[[240,480],[422,480],[422,208]])
    f.label('x',456,744,color=MUTED)
    f.label('h',433,466,color=MUTED)
    f.label('(b) Routed Experts',540,150)
    f.node('tokens','Input token features\n[B × T, 768]',1010,764,width=425,height=80,color=GRAY)
    f.node('router','Router\nLinear(768 → 4) + Softmax',1010,645,width=830,height=80,color=GRAY)
    f.node('top','Top-1 selection\nnormalize selected weight',1010,520,width=425,height=80,color=GOLD)
    for i,x in enumerate([650,890,1130,1370]):
        f.node('e'+str(i),'Expert '+str(i+1)+'\nSwiGLU · 2,432',x,370,width=210,height=100,
               color=BLUE if i==1 else GRAY)
        f.edge('top','e'+str(i),dashed=i!=1,color=BLUE[1] if i==1 else GRAY[1])
    f.node('merge','Weighted Scatter-Add\n[B × T, 768]',1010,208,width=860,height=80,color=PURPLE)
    f.chain('tokens','router','top')
    f.edge('tokens','e1',via=[[550,764],[550,452],[890,452]],color=INK)
    f.label('token features',565,715,color=MUTED)
    for i in range(4):
        f.edge('e'+str(i),'merge',dashed=i!=1,color=BLUE[1] if i==1 else GRAY[1])
    f.label('One token shown: Expert 2 is active; dashed routes are not selected for this token.',535,834,color=MUTED)
    f.group(['tokens','router','top','e0','e1','e2','e3','merge'],PURPLE,24)
    f.panel(48,163,425,654,fill='#F0E5F8')
    f.footer('No shared expert. The compatible 768-dim MoE checkpoint is PUBLIC upstream SFT, not a completed EvoMind training result.')
    return f.spec


def text_data():
    f=Figure('EvoMind · Data & Training Pipeline',
             'Source: jingyaogong/minimind_dataset · pinned revision 312afb4f… · counts below are records / preference pairs.',1780,980)
    f.label('(a) Pretrain',55,160);f.label('(b) Supervised Fine-Tuning',390,160)
    f.label('(c) Preference Optimization',795,160);f.label('(d) Online RL',1210,160)
    f.panel(30,116,337,443,fill='#EDF3FA',shadow=False)
    f.panel(383,116,370,443,fill='#ECF4E5',shadow=False)
    f.panel(770,116,420,443,fill='#FFF4CD',shadow=False)
    f.panel(1205,116,549,591,fill='#FAE7D8',shadow=False)
    f.node('pre_data','pretrain_t2t_mini.jsonl\n1,270,238 records',195,259,width=302,height=92)
    f.node('sft_data','sft_t2t_mini.jsonl\n905,718 records',565,259,width=330,height=92,color=GREEN)
    f.node('dpo_data','dpo.jsonl\n17,166 preference pairs',973,259,width=330,height=92,color=GOLD)
    f.node('rl_data','rlaif.jsonl\n19,502 prompts',1460,259,width=486,height=92,color=PEACH)
    f.node('pre','Pretrain · 2 epochs\nNext-token CE\nRandom initialization',195,440,width=302,height=110)
    f.node('sft','SFT · 2 epochs\nAssistant-only CE',565,440,width=330,height=110,color=GREEN)
    f.node('dpo','DPO · 1 epoch\nPreference pairs + reference',973,440,width=390,height=110,color=GOLD)
    f.node('cispo','CISPO · 1 epoch\nSelected text base',1460,440,width=390,height=110,color=PEACH)
    f.node('grpo','GRPO · 1 epoch\nSame-DPO comparison',1460,645,width=390,height=92,color=PEACH)
    f.edge('pre_data','pre',dashed=True);f.edge('sft_data','sft',dashed=True)
    f.edge('dpo_data','dpo',dashed=True);f.edge('rl_data','cispo',dashed=True)
    f.edge('rl_data','grpo',via=[[1734,259],[1734,645]],dashed=True)
    f.chain('pre','sft','dpo','cispo');f.edge('dpo','grpo',via=[[1180,440],[1180,645]])
    f.node('agent_data','agent_rl.jsonl\n39,988 tool-use records',565,645,width=360,height=92,color=GRAY)
    f.node('agent','Agentic-RL / CISPO\nPrepared · not yet trained',565,823,width=390,height=92,color=GRAY)
    f.node('vlm','EvoMind-V\nDense single-image training',1460,823,width=390,height=92,color=PURPLE)
    f.edge('agent_data','agent',dashed=True)
    f.edge('cispo','vlm',via=[[1690,440],[1690,823]],color=PURPLE[1])
    f.edge('cispo','agent',via=[[1460,558],[1220,558],[1220,748],[850,748],[850,823]],dashed=True,color=GRAY[1])
    f.label('GRPO / CISPO: G=6 · frozen reward model + rules',865,919,color=MUTED)
    f.label('Same DPO parent · 1 epoch / 9,751 updates each',865,950,color=MUTED)
    f.label('Solid arrows: actual weight lineage',48,865,color=MUTED)
    f.label('Dashed: data feed / planned extension',48,901,color=MUTED)
    return f.spec


def vision_arch():
    f=Figure('EvoMind-V · Single-Image Architecture',
             'Frozen SigLIP2 vision tower + trainable projector + selected 768-dim CISPO text backbone.',1680,1080)
    f.label('(a) Image features',45,156)
    f.label('(b) Token Alignment',670,156)
    f.label('(c) Language Model',1190,156)
    f.panel(43,177,500,757,fill='#EDF4FA')
    f.panel(586,177,550,757,fill='#FFF8E6')
    f.panel(1168,165,493,825,fill='#F2EDF8')
    for name,text,y,col in [('image','One RGB image',236,BLUE),
        ('resize','Resize + normalize\n256 × 256 pixels',357,BLUE),
        ('patch','32 × 32 patches\n8 × 8 = 64 tokens',491,BLUE),
        ('vit','SigLIP2 vision tower\n12 layers · hidden 768\nFROZEN',636,GRAY),
        ('project','MLP Projector\nLayerNorm → Linear\n→ GELU → Linear\nTRAINABLE · 768 → 768 → 768',839,PURPLE)]:
        f.node(name,text,295,y,width=455,height=142 if name=='project' else 103,color=col)
    f.chain('image','resize','patch','vit','project')
    f.label('[B, 64, 768]',325,726,color=MUTED)
    f.node('prompt','User question + chat template\n64 image-pad positions',860,236,width=422,height=100,color=GOLD)
    f.node('token','BPE Tokenizer\nText token IDs',860,378,width=340,height=92,color=GOLD)
    f.node('embed','Text Embedding\nFROZEN · hidden 768',860,519,width=375,height=92,color=GRAY)
    f.node('replace','Visual Token Injection\nReplace 64 image-pad embeddings',860,688,width=520,height=100,color=PURPLE)
    f.node('mixed','Mixed visual + text sequence\n[B, T, 768] · training T ≤ 768',860,870,width=470,height=100,color=PURPLE)
    f.chain('prompt','token','embed','replace','mixed')
    f.edge('project','replace',via=[[565,839],[565,688]])
    for name,text,y,col in [('b1','LLM block 1\nTRAINABLE',236,PURPLE),
        ('middle','LLM Blocks 2–7\nFROZEN parameters\nGradients pass through',410,GRAY),
        ('b8','LLM block 8\nTRAINABLE',587,PURPLE),
        ('head','Final RMSNorm + LM Head\nFROZEN parameters',754,GRAY),
        ('loss','Assistant-only CE\nIgnore prompt / image / padding',923,PEACH)]:
        f.node(name,text,1414,y,width=445,height=114,color=col)
    f.edge('mixed','b1',via=[[1128,870],[1128,236]])
    f.chain('b1','middle','b8','head','loss')
    f.label('Gray: frozen parameters    Purple: updated parameters / learned features',45,1003,color=MUTED)
    f.footer('Current training: 2 epochs · batch 4 · BF16 · LR 5e-6. No SigLIP text tower, cross-attention module or video encoder is used.')
    return f.spec


def vision_data(counts):
    f=Figure('EvoMind-V · Visual Data Pipeline',
             'Dataset provenance and split isolation are separate from final model-quality evaluation.',1800,970)
    f.label('Publisher-described sources: ALLaVA-4V caption / instruction (LAION + VFLAN) + supplementary examples',45,133,color=MUTED)
    f.node('source','minimind-v_dataset\nsft_i2t.parquet\n2,904,511 source records',243,250,width=395,height=142,color=BLUE)
    f.node('clean','Validate & Filter\nImages · roles · visual markers\nRequire supervised answer',735,250,width=435,height=142,color=GREEN)
    f.node('accepted','2,654,826 accepted records\n249,685 rejected',1295,250,width=570,height=142,color=GREEN)
    f.chain('source','clean','accepted')
    f.node('group','Image-Level Split\nGroup by original image-byte SHA256',1295,455,width=630,height=110,color=GOLD)
    f.edge('accepted','group')
    f.node('train','TRAIN\n2,544,979 records',740,640,width=385,height=110,color=BLUE)
    f.node('val','VALIDATION\n'+f"{counts['val']:,}"+' records',1200,640,width=355,height=110,color=GRAY)
    f.node('test','TEST\n'+f"{counts['test']:,}"+' records',1583,640,width=345,height=110,color=GRAY)
    f.edge('group','train',via=[[740,455]])
    f.edge('group','val');f.edge('group','test')
    f.node('native','Train-only Parquet\n2,544,979 rows',265,640,width=390,height=110,color=BLUE)
    f.edge('train','native')
    f.node('fit','Dense Visual SFT\n2 epochs · 1,272,490 updates',265,838,width=445,height=116,color=PURPLE)
    f.edge('native','fit')
    f.node('eval','Final Checkpoint → Six Reference Images\nDescriptions · EOS · repetition · latency · memory\nQualitative inspection, not a benchmark score',1090,838,width=1100,height=116,color=PEACH)
    f.edge('fit','eval')
    f.label('Original image-byte groups: 645,226',45,500,color=MUTED)
    f.label('Only the TRAIN split enters this run.',45,540,color=MUTED)
    f.panel(24,160,1749,188,fill='#EEF4EB')
    f.panel(525,382,1248,334,fill='#FFF7DF')
    f.panel(23,757,1637,153,fill='#F3EBF8')
    f.footer('Byte-hash grouping is not semantic near-deduplication; the six evaluation images are a separate reference set, not these val/test splits.')
    return f.spec


def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extended_render(spec, renderer):
    """Use the supplied FigureSpec renderer; route declared orthogonal edges.

    The SVG is rendered from JSON as one deterministic build. Generated SVGs
    are never hand-edited. Unknown project extensions are harmless to upstream.
    """
    issues=renderer.validate_spec(spec)
    if issues:
        raise ValueError('\n'.join(issues))
    root=ET.fromstring(renderer.render_svg(spec))
    ns='{http://www.w3.org/2000/svg}'
    defs=root.find(ns+'defs')
    for name,opacity,dy,blur in [('module-shadow','0.19','5','5'),('panel-shadow','0.12','4','7')]:
        shadow=ET.SubElement(defs,ns+'filter',dict(id=name,x='-25%',y='-50%',width='150%',height='210%'))
        ET.SubElement(shadow,ns+'feDropShadow',{'dx':'0','dy':dy,'stdDeviation':blur,
            'flood-color':'#282432','flood-opacity':opacity})
    # Fixed physical arrowheads avoid huge heads on thicker publication strokes.
    for marker in defs.findall(ns+'marker'):
        marker.attrib.update(markerWidth='14',markerHeight='12',refX='13',refY='6',markerUnits='userSpaceOnUse')
        marker.find(ns+'polygon').set('points','0 0, 14 6, 0 12, 3 6')
    # Broad, quiet backgrounds group the same conceptual units as the reference.
    for panel in reversed(spec.get('panels',[])):
        attrs={k:str(panel[k]) for k in ('x','y','width','height','fill')}
        attrs.update(rx='24',stroke='none')
        if panel.get('shadow',True):
            attrs['filter']='url(#panel-shadow)'
        root.insert(list(root).index(defs)+1,ET.Element(ns+'rect',attrs))
    nodes={n['id']:n for n in spec['nodes']}
    children=list(root)
    shape_tags={ns+shape for shape in ('rect','circle','ellipse','polygon')}
    node_shapes=[element for element in children if element.tag in shape_tags and element.get('stroke-width')=='2']
    if len(node_shapes)!=len(nodes):
        raise ValueError('FigureSpec renderer changed node output structure')
    for node,shape in zip(spec['nodes'],node_shapes):
        shape.set('id','node-'+node['id'])
        if node['shape']=='circle':
            shape.attrib.update(fill='#FFFFFF',stroke=INK,**{'stroke-width':'3'})
        else:
            shape.attrib.update(stroke='none',filter='url(#module-shadow)')
            if node['shape']=='rounded':
                shape.set('rx','14')
        lines=[line for line in node['label'].split('\n') if line]
        offset=children.index(shape)+1
        start_y=node['y']-(len(lines)-1)*27/2+8
        for i in range(len(lines)):
            label=children[offset+i]
            if label.tag!=ns+'text':
                raise ValueError('FigureSpec renderer changed node label structure')
            label.set('data-node',node['id'])
            label.set('y',f'{start_y+i*27:.1f}')
            label.set('font-family',SERIF if i==0 else SANS)
            label.set('font-size','27' if i==0 else '20')
            label.set('font-style','italic' if i==0 else 'normal')
            label.set('font-weight','bold' if i==0 else 'normal')
            if node['shape']=='circle':
                label.set('font-size','32');label.set('font-style','normal')
    # FigureSpec groups get the same soft panel finish, without hard outlines.
    for element in root:
        if element.tag==ns+'rect' and element.get('stroke-width')=='1':
            element.attrib.update(rx='24',stroke='none',filter='url(#panel-shadow)')
    free_texts=[el for el in root if el.tag==ns+'text'][-len(spec['labels']):]
    for item,label in zip(spec['labels'],free_texts):
        if item['text']==spec['title'] or re.match(r'^\([a-z]\)',item['text']):
            label.attrib.update({'font-family':SERIF,'font-style':'italic','font-weight':'bold',
                                 'font-size':'36' if item['text']==spec['title'] else '27'})
        elif item['y']!=91:
            label.set('font-size','20')
    paths=[element for element in root if element.tag==ns+'path']
    if len(paths)!=len(spec['edges']):
        raise ValueError('FigureSpec renderer changed edge output structure')
    for edge,path in zip(spec['edges'],paths):
        if edge.get('via'):
            a,b=nodes[edge['from']],nodes[edge['to']]
            first,last=edge['via'][0],edge['via'][-1]
            begin=renderer.clip_to_shape(a['x'],a['y'],*first,a['width'],a['height'],a['shape'])
            end=renderer.clip_to_shape(b['x'],b['y'],*last,b['width'],b['height'],b['shape'])
            points=[begin,*edge['via'],end]
            path.set('d','M '+' L '.join(f'{x:.1f},{y:.1f}' for x,y in points))
        if not edge.get('arrow',True):
            path.attrib.pop('marker-end',None)
        path.set('stroke-linejoin','round')
        path.set('stroke-linecap','round')
    root.set('role','img');root.set('aria-label',spec['title'])
    title=ET.Element(ns+'title');title.text=spec['title'];root.insert(0,title)
    ET.register_namespace('',ns[1:-1])
    return ET.tostring(root,encoding='unicode',xml_declaration=False)+'\n'


def preview(svg, spec, edge, preview_root):
    """Own headless process + isolated profile; no personal browser session."""
    width,height=spec['canvas']['width'],spec['canvas']['height']
    document=preview_root/(svg.stem+'.html')
    body=svg.read_text(encoding='utf-8')
    # Browser font metrics, not just character-count estimates, check all text.
    inspector="""<pre id="layout-result" style="display:none"></pre><script>
    const svg=document.querySelector('svg');
    const texts=[...svg.querySelectorAll('text')].map(e=>{
      const b=e.getBBox(); return {text:e.textContent,x:b.x,y:b.y,w:b.width,h:b.height};
    }).filter(t=>t.text.trim());
    const overlap=[];
    for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){
      const a=texts[i],b=texts[j];
      if(Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x)>1 &&
         Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y)>1)overlap.push([a.text,b.text]);
    }
    const nodeOverflow=[...svg.querySelectorAll('text[data-node]')].filter(e=>{
      const a=e.getBBox(),b=document.getElementById('node-'+e.dataset.node).getBBox();
      return a.x<b.x+4||a.y<b.y+2||a.x+a.width>b.x+b.width-4||a.y+a.height>b.y+b.height-2;
    }).map(e=>e.textContent);
    const v=svg.viewBox.baseVal;
    const outside=texts.filter(t=>t.x<0||t.y<0||t.x+t.w>v.width||t.y+t.h>v.height);
    document.getElementById('layout-result').textContent=JSON.stringify({text_count:texts.length,overlap,outside,nodeOverflow});
    </script>"""
    document.write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(spec['title'])+
        '</title><style>@page {size:'+str(width)+'px '+str(height)+'px;margin:0}'+
        'html,body{margin:0;padding:0;background:white}svg{display:block}</style>'+body+inspector,encoding='utf-8')
    command=[str(edge),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
             '--disable-extensions','--hide-scrollbars','--allow-file-access-from-files',
             '--user-data-dir='+str(preview_root/'edge-profile'),
             '--window-size='+str(width)+','+str(height),'--force-device-scale-factor=1',
             '--screenshot='+str(svg.with_suffix('.png')),'--print-to-pdf='+str(svg.with_suffix('.pdf')),
             '--no-pdf-header-footer','--dump-dom',document.as_uri()]
    result=subprocess.run(command,capture_output=True,timeout=90)
    if result.returncode or not svg.with_suffix('.png').exists() or not svg.with_suffix('.pdf').exists():
        raise RuntimeError('Headless figure export failed: '+result.stderr.decode('utf-8',errors='replace')[-1500:])
    match=re.search(r'<pre id="layout-result"[^>]*>(.*?)</pre>',result.stdout.decode('utf-8',errors='replace'),re.S)
    if not match:
        raise RuntimeError('Browser did not return layout inspection')
    report=json.loads(html.unescape(match.group(1)))
    (preview_root/(svg.stem+'.layout.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if report['overlap'] or report['outside'] or report['nodeOverflow']:
        raise RuntimeError('Text collision/out-of-bounds: '+json.dumps(report,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renderer',type=Path,required=True,help='Path to ARIS figure-spec/scripts/figure_renderer.py')
    parser.add_argument('--style-kernel',type=Path,help='Optional figure-style/kernel.py (applies typography defaults)')
    parser.add_argument('--edge',type=Path,help='Optional Edge executable, for PNG/PDF export')
    parser.add_argument('--build-specs',action='store_true',help='Regenerate editable specs from this audited layout')
    parser.add_argument('--vision-val',type=int)
    parser.add_argument('--vision-test',type=int)
    args=parser.parse_args()
    renderer=load_module(args.renderer,'readme_figurespec')
    if args.style_kernel:
        load_module(args.style_kernel,'readme_figure_style').apply_figure_style(frame='none',sizes=(22,20,18))
    if args.build_specs and (args.vision_val is None or args.vision_test is None or
                            args.vision_val+args.vision_test != 2654826-2544979):
        parser.error('Provide verified val/test record counts; their sum must equal 109847')
    layouts=[(ROOT,'evomind_dense_architecture',dense),
             (ROOT,'evomind_moe_architecture',moe),
             (ROOT,'evomind_text_data_pipeline',text_data),
             (ROOT/'vision','evomind_v_architecture',vision_arch),
             (ROOT/'vision','evomind_v_data_pipeline',lambda:vision_data(dict(val=args.vision_val,test=args.vision_test)))]
    results=[]
    for repo,name,build in layouts:
        directory=repo/'figures';specpath=directory/'specs'/(name+'.json')
        specpath.parent.mkdir(parents=True,exist_ok=True)
        if args.build_specs:
            specpath.write_text(json.dumps(build(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        spec=json.loads(specpath.read_text(encoding='utf-8'))
        svg=directory/(name+'.svg')
        svg.write_text(extended_render(spec,renderer),encoding='utf-8')
        if args.edge:
            preview_root=ROOT/'artifacts/readme_diagram_preview'
            preview_root.mkdir(parents=True,exist_ok=True)
            preview(svg,spec,args.edge,preview_root)
        results.append(dict(name=name,svg=str(svg),spec=str(specpath),
                            sha256=hashlib.sha256(svg.read_bytes()).hexdigest(),schema_issues=[]))
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
