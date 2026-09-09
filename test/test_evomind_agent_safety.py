"""Offline, stdlib-only tool security and multi-turn token-boundary fixtures."""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainer.evomind_tools import (ToolInputError, safe_arithmetic, parse_tool_calls,
                                  parse_arguments, validate_tool_arguments)
from trainer.evomind_agent_tokens import (trim_generated_turn, observation_suffix,
                                        tool_observation_tokens, pack_agent_sample)


def load_agent_tool_functions():
    """Exercise the production definitions without importing torch/transformers."""
    tree = ast.parse((ROOT / 'trainer/train_agent.py').read_text(encoding='utf-8'))
    names = {'WEATHER_DATA', 'TIME_DATA', 'EXCHANGE_DATA', 'TRANSLATE_DATA', 'UNIT_DATA', 'MOCK_RESULTS', 'CHECK_ARGS'}
    body = [node for node in tree.body if (isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id in names for target in node.targets))
            or (isinstance(node, ast.FunctionDef) and node.name == 'execute_tool')]
    scope = dict(safe_arithmetic=safe_arithmetic, validate_tool_arguments=validate_tool_arguments, ToolInputError=ToolInputError)
    exec(compile(ast.Module(body=body, type_ignores=[]), '<production-agent-tool-definitions>', 'exec'), scope)
    return scope


def load_eval():
    spec = importlib.util.spec_from_file_location('evomind_test_eval_toolcall', ROOT / 'scripts/eval_toolcall.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArithmeticTests(unittest.TestCase):
    def test_legal_arithmetic_and_normalization(self):
        for expression, expected in {'256*37': 9472, '2^10': 1024, '（6×7）÷2': 21,
                                     '3²+2³': 17, 'sqrt(144)': 12.0, '-3**2': -9,
                                     'abs(-4)+floor(2.9)+ceil(2.1)': 9, '17//5+17%5': 5,
                                     '2**-2': .25, '0.5+1.5': 2.0}.items():
            with self.subTest(expression=expression):
                self.assertEqual(safe_arithmetic(expression), expected)

    def test_arbitrary_python_and_non_numbers_rejected(self):
        for expression in ["__import__('os').system('whoami')", '(1).__class__', 'math.sqrt(9)',
                           '(lambda: 1)()', '[1,2][0]', '[x for x in [1]]', 'open("x","w")',
                           'True+1', '"x"*100', '1 if 1 else 0', 'sqrt(x=4)', 'pow(2,4)',
                           '2<<4', '1<2', '1j', 'pi', '1;2']:
            with self.subTest(expression=expression), self.assertRaises(ToolInputError):
                safe_arithmetic(expression)

    def test_resource_and_domain_limits(self):
        for expression in ['1'*513, '9'*100, '2**100000000', '9**9**9', '1e309',
                           '1e100*1e100', 'sqrt(-1)', '1/0', '(-1)**0.5',
                           '+'*40+'1', '+'.join(['1']*100), '('*300+'1'+')'*300]:
            with self.subTest(expression=expression), self.assertRaises(ToolInputError):
                safe_arithmetic(expression)

    def test_argument_shape_type_and_size(self):
        self.assertEqual(validate_tool_arguments('calculate_math', '{"expression":"1+2"}'), {'expression':'1+2'})
        for name, args in [('calculate_math', []), ('calculate_math', {'expression':1}),
                           ('calculate_math', {}), ('unit_converter', {'value':True,'from_unit':'km','to_unit':'miles'}),
                           ('get_current_weather', {'location':[]}), ('random_number', {'min':10,'max':2}),
                           ('get_current_time', {'unexpected':'x'}), ('calculate_math', {'expression':'x'*8193})]:
            with self.subTest(name=name,args=str(args)[:100]), self.assertRaises(ToolInputError):
                validate_tool_arguments(name,args)
        for raw in ['null', '[]', '{"value":NaN}', '{"value":Infinity}', '{"value":'+ '['*1000 + '0'+']'*1000+'}']:
            with self.subTest(raw=raw[:30]), self.assertRaises(ToolInputError): parse_arguments(raw)

    def test_parser_filters_malformed_and_bounds_call_count(self):
        text = '<tool_call>[]</tool_call><tool_call>{broken}</tool_call><tool_call>{"name":"calculate_math","arguments":{"expression":"3*4"}}</tool_call>'
        self.assertEqual(len(parse_tool_calls(text)), 1)
        with self.assertRaises(ToolInputError): parse_tool_calls(text*9)
        with self.assertRaises(ToolInputError): parse_tool_calls('x'*65537)

    def test_agent_mock_values_unchanged_and_windows_safe(self):
        execute = load_agent_tool_functions()['execute_tool']
        self.assertEqual(execute('calculate_math', {'expression':'256*37'}), {'result':'9472'})
        self.assertEqual(execute('get_exchange_rate', {'from_currency':'USD','to_currency':'CNY'})['rate'], 7.21)
        self.assertEqual(execute('get_current_weather', {'location':'北京'})['temperature'], '28°C')
        self.assertEqual(execute('get_current_time', {})['datetime'], '2025-03-07 14:30:00')
        self.assertEqual(execute('unit_converter', {'value':30,'from_unit':'celsius','to_unit':'fahrenheit'})['result'], 54.0)
        self.assertEqual(execute('translate_text', {'text':'你好世界','target_language':'english'}), {'translated_text':'Hello World'})
        self.assertIsNone(execute('calculate_math', {'expression':'__import__("os")'}))
        self.assertIsNone(execute('calculate_math', []))
        self.assertIsNone(execute([], {}))


class TokenBoundaryTests(unittest.TestCase):
    def test_eos_is_retained_once_and_padding_ignored(self):
        self.assertEqual(trim_generated_turn([7,2,0,0], [-.1,-.2,9,9], [1,1,1,1], eos_token_id=2,pad_token_id=0), ([7,2],[-.1,-.2]))
        self.assertEqual(trim_generated_turn([7,2,2], [-.1,-.2,9], [1,1,0], eos_token_id=2,pad_token_id=2), ([7,2],[-.1,-.2]))
        self.assertEqual(trim_generated_turn([7,0,9], [-.1,9,9], [1,1,1], eos_token_id=2,pad_token_id=0), ([7],[-.1]))
        with self.assertRaises(ValueError): trim_generated_turn([7],[float('nan')],[1],eos_token_id=2,pad_token_id=0)
        with self.assertRaises(ValueError): trim_generated_turn([7],[],[1],eos_token_id=2,pad_token_id=0)

    def test_multiturn_eos_and_tool_observation_masks(self):
        prompt, response = [100,101], [7,2,80,81,8,2]
        mask, logps = [1,1,0,0,1,1], [-.1,-.2,0,0,-.3,-.4]
        ids, kept_mask, plen, old = pack_agent_sample(prompt,response,mask,logps,20)
        self.assertEqual([i for i,m in zip(ids,kept_mask) if m], [7,2,8,2])
        self.assertEqual(plen,2)
        self.assertEqual(old,[0,-.1,-.2,0,0,-.3,-.4])
        ids, kept_mask, _, old = pack_agent_sample(prompt,response,mask,logps,4)
        self.assertEqual(ids,[80,81,8,2]); self.assertEqual(kept_mask,[0,0,1,1]); self.assertEqual(old,[0,-.3,-.4])
        with self.assertRaises(ValueError): observation_suffix([1,4,3], [1], [2])

    def test_template_suffix_does_not_rewrite_real_assistant(self):
        class Tokenizer:
            eos_token_id=2
            def apply_chat_template(self,messages,**kwargs):
                anchor=messages[1]['content']
                return 'normalized_think_header'+anchor+'<eos>TOOL'+('PROMPT' if kwargs['add_generation_prompt'] else '')
            def __call__(self,text,**kwargs):
                self.seen=text
                return {'input_ids':[2,80,81]}
        tokenizer=Tokenizer()
        result=tool_observation_tokens(tokenizer,[{'role':'tool','content':'answer'}], tools=[],
            add_generation_prompt=True,open_thinking=True,already_ended=True)
        self.assertEqual(result,[80,81]); self.assertEqual(tokenizer.seen,'<eos>TOOLPROMPT')
        self.assertEqual(tool_observation_tokens(tokenizer,[],tools=[],add_generation_prompt=False,
            open_thinking=False,already_ended=False),[2,80,81])


class LocalEvalTests(unittest.TestCase):
    def test_missing_weight_fails_before_any_model_is_constructed(self):
        module=load_eval()
        args=SimpleNamespace(model_format='native',hidden_size=768,num_hidden_layers=8,
            load_from=str(ROOT/'model'),save_dir=str(ROOT/'artifacts/definitely_missing_evomind_test_weight'),
            weight='missing',use_moe=0,lora=None)
        with self.assertRaises(FileNotFoundError): module.init_model(args)
        args.load_from=str(ROOT/'artifacts/a_model_named_but_untrusted_tokenizer')
        with self.assertRaises(ValueError): module.model_artifacts(args)

    def test_local_import_does_not_import_openai_or_torch(self):
        before=set(sys.modules)
        load_eval()
        self.assertFalse({'torch','transformers','openai'} & (set(sys.modules)-before))

    def test_eval_mock_values_and_malformed_arguments(self):
        module=load_eval()
        self.assertEqual(module.execute_tool('get_exchange_rate', {'from_currency':'USD','to_currency':'CNY'})['rate'],7.15)
        self.assertEqual(module.execute_tool('calculate_math', {'expression':'sqrt(144)'})['result'],'12.0')
        self.assertIn('error',module.execute_tool('calculate_math', '[]'))
        self.assertIn('error',module.execute_tool('calculate_math', {'expression':'1e999'}))

    def test_max_turns_terminates_and_marks_unfinished(self):
        module=load_eval(); args=SimpleNamespace(backend='local',max_turns=2)
        output='<tool_call>{"name":"calculate_math","arguments":{"expression":"1+2"}}</tool_call>'
        with patch.object(module,'generate',return_value=output) as generated, contextlib.redirect_stdout(io.StringIO()):
            result=module.run_case('test',module.get_tools(['calculate_math']),args)
        self.assertEqual(generated.call_count,2); self.assertTrue(result['unfinished']); self.assertEqual(result['status'],'max_turns')
        with patch.object(module,'generate',return_value='The answer is 3.'), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(module.run_case('test',[],args)['unfinished'])


if __name__ == '__main__':
    unittest.main()
