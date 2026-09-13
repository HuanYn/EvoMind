"""CPU-only contracts for README architecture figures, not model-quality tests."""
import importlib.util
import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
module_spec = importlib.util.spec_from_file_location('readme_diagrams', ROOT/'scripts/render_readme_diagrams.py')
diagrams = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(diagrams)


class TestReadmeDiagrams(unittest.TestCase):
    def layouts(self):
        return [diagrams.dense(), diagrams.moe(), diagrams.text_data(),
                diagrams.vision_arch(), diagrams.vision_data({'val':54362,'test':55485})]

    def test_nodes_unique_and_edges_resolve(self):
        for spec in self.layouts():
            ids = [node['id'] for node in spec['nodes']]
            self.assertEqual(len(ids), len(set(ids)))
            for edge in spec['edges']:
                self.assertIn(edge['from'], ids)
                self.assertIn(edge['to'], ids)

    def test_text_lineage_is_a_matched_dpo_fork(self):
        spec = diagrams.text_data()
        edges = {(e['from'],e['to']):e for e in spec['edges']}
        self.assertIn(('dpo','grpo'), edges)
        self.assertIn(('dpo','cispo'), edges)
        self.assertNotIn(('grpo','cispo'), edges)
        self.assertNotIn(('cispo','grpo'), edges)
        self.assertEqual(edges['cispo','agent']['style'], 'dashed')
        self.assertEqual(edges['cispo','vlm']['style'], 'solid')

    def test_moe_features_are_not_router_probabilities(self):
        edges = {(e['from'],e['to']) for e in diagrams.moe()['edges']}
        self.assertIn(('tokens','router'), edges)
        self.assertIn(('tokens','e1'), edges)
        self.assertIn(('top','e1'), edges)

    def test_two_residuals_do_not_share_a_skip_bus(self):
        for spec,first,second in [(diagrams.dense(),('x','a1'),('a1','a2')),
                                  (diagrams.moe(),('x','add1'),('add1','add2'))]:
            edges = {(e['from'],e['to']):e for e in spec['edges']}
            self.assertNotEqual(edges[first]['via'][0][0], edges[second]['via'][-1][0])

    def test_vision_accounting_and_steps(self):
        self.assertEqual(2654826+249685,2904511)
        self.assertEqual(2544979+54362+55485,2654826)
        self.assertEqual(2*((2544979+3)//4),1272490)
        labels = {n['id']:n['label'] for n in diagrams.vision_arch()['nodes']}
        self.assertIn('FROZEN',labels['middle'])
        self.assertIn('Gradients pass through',labels['middle'])
        self.assertIn('training T',labels['mixed'])

    def test_main_specs_match_audited_layout(self):
        for name,build in [('evomind_dense_architecture',diagrams.dense),
                           ('evomind_moe_architecture',diagrams.moe),
                           ('evomind_text_data_pipeline',diagrams.text_data)]:
            spec = json.loads((ROOT/'figures/specs'/f'{name}.json').read_text(encoding='utf-8'))
            self.assertEqual(spec,build())

    def test_main_exports_exist_and_svg_is_editable(self):
        for name in ['evomind_dense_architecture','evomind_moe_architecture','evomind_text_data_pipeline']:
            for ext in ['svg','png','pdf']:
                self.assertGreater((ROOT/'figures'/f'{name}.{ext}').stat().st_size,100)
            svg = ET.parse(ROOT/'figures'/f'{name}.svg').getroot()
            ns = {'s':'http://www.w3.org/2000/svg'}
            self.assertGreater(len(svg.findall('s:text',ns)),10)
            self.assertEqual(svg.get('role'),'img')

    def test_document_local_links_exist(self):
        documents = [ROOT/'README.md', ROOT/'figures/README.md', ROOT/'docs/README_DIAGRAM_SOURCES.md']
        if (ROOT/'vision/figures/README.md').exists():
            documents += [ROOT/'vision/README.md', ROOT/'vision/figures/README.md',
                          ROOT/'vision/docs/VISUAL_DATA_SOURCES.md']
        for document in documents:
            text = document.read_text(encoding='utf-8')
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
                if re.match(r'^[a-z]+://',target) or target.startswith('#'):
                    continue
                path = unquote(target.split('#',1)[0]).strip('<>')
                self.assertTrue((document.parent/path).exists(), f'{document}: {path}')


if __name__ == '__main__':
    unittest.main()
