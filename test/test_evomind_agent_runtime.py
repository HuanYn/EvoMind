"""Tiny CPU-only real Agent training-loop/checkpoint integration fixtures."""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import copy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import datasets  # Required Windows import order.
import torch
torch.set_num_threads(1)
from trainer import train_agent as agent
from trainer.evomind_agent_runtime import AgentAdamW, checkpoint_paths, load_checkpoint, save_checkpoint
from trainer.evomind_rl_runtime import seed_rng, restore_rng, checked_update
from trainer.evomind_agent_tokens import tool_observation_tokens


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=torch.nn.Embedding(32,8)
        self.dropout=torch.nn.Dropout(.2)
        self.head=torch.nn.Linear(8,32)
    def forward(self,input_ids,attention_mask=None,**kwargs):
        return SimpleNamespace(logits=self.head(self.dropout(self.embedding(input_ids))),aux_loss=torch.tensor(0.))


class TinyTokenizer:
    pad_token_id=0
    eos_token_id=2
    def apply_chat_template(self,*args,**kwargs): return 'prompt'


class Engine:
    def update_policy(self,model): self.model=model


def fake_rollout(*args, **kwargs):
    tokens=torch.randint(4,24,(2,)).tolist()
    return (['bad','valid final answer'], ['c1','c2'], [[1,3],[1,3]],
            [[tokens[0],2],[tokens[1],2]], [[1,1],[1,1]], [[-2.,-2.],[-2.,-2.]],
            [['bad'],['valid final answer']], [False,False])


class AgentRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.cuda_guard=patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CPU fixture must never initialize CUDA'))
        self.cuda_guard.start()
        self.addCleanup(self.cuda_guard.stop)
        temp_root=ROOT/'artifacts/test_tmp'
        temp_root.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='agent_runtime_',dir=temp_root)
        self.root=Path(self.temporary.name)
        self.batch={'messages':[[{'role':'user','content':'test'}]], 'tools':[[]], 'gt':[[]]}
    def tearDown(self): self.temporary.cleanup()
    def initialize(self,name,max_steps=0):
        seed_rng(123,'cpu')
        agent.args=SimpleNamespace(epochs=1,accumulation_steps=2,save_interval=1,log_interval=99,
            grad_clip=1.,max_total_len=20,num_generations=2,beta=.1,epsilon=.2,epsilon_high=5.,
            loss_type='grpo',max_gen_len=2,max_turns=3,thinking_ratio=0.,device='cpu',debug_mode=False,max_steps=max_steps,
            save_dir=str(self.root/name/'out'),resume_dir=str(self.root/name/'state'),save_weight='agent')
        agent.lm_config=SimpleNamespace(hidden_size=8,use_moe=False)
        agent.model=TinyPolicy()
        ref=copy.deepcopy(agent.model).eval().requires_grad_(False)
        agent.optimizer=AgentAdamW(agent.model.parameters(),lr=1e-3)
        agent.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(agent.optimizer,T_max=3)
        agent.tokenizer=TinyTokenizer()
        agent.autocast_ctx=nullcontext()
        agent.training_contract={'fixture':'tiny_actual_Agent_loop'}
        agent.invocation_updates=agent.optimizer_updates=0
        return ref
    def run_loop(self,ref,start=0):
        with patch.object(agent,'rollout_batch',side_effect=fake_rollout), patch.object(agent,'Logger'):
            return agent.rl_train_epoch(0,[copy.deepcopy(self.batch) for _ in range(5-start)],5,
                                        Engine(),ref,start_step=start)

    def test_probe_resume_matches_full_and_tail_checkpoint_is_post_update(self):
        ref=self.initialize('full')
        self.assertFalse(self.run_loop(ref))
        expected={k:v.clone() for k,v in agent.model.state_dict().items()}
        full_paths=checkpoint_paths(agent.args,agent.lm_config)
        full=load_checkpoint(full_paths[1],agent.training_contract)
        self.assertEqual((full['step'],full['optimizer_updates'],full['status']),(5,3,'complete'))
        receipt=json.loads(full_paths[0].with_suffix('.runtime.json').read_text())
        self.assertEqual((receipt['status'],receipt['stop_reason'],receipt['probe_only']),('completed','epochs_complete',False))
        self.assertEqual(agent.scheduler.last_epoch,3)
        self.assertTrue(agent.model.training)
        for k,v in full['model'].items(): self.assertTrue(torch.equal(v,expected[k]))

        ref=self.initialize('resumed',max_steps=1)
        self.assertTrue(self.run_loop(ref))
        self.assertEqual(agent.invocation_updates,1)
        _, path, status_path=checkpoint_paths(agent.args,agent.lm_config)
        saved=load_checkpoint(path,agent.training_contract)
        self.assertEqual((saved['step'],saved['optimizer_updates'],saved['status']),(2,1,'probe_complete'))
        self.assertEqual(json.loads(status_path.read_text())['pending_microsteps'],0)
        receipt=json.loads(checkpoint_paths(agent.args,agent.lm_config)[0].with_suffix('.runtime.json').read_text())
        self.assertEqual((receipt['status'],receipt['stop_reason'],receipt['optimizer_updates_this_invocation']),('probe_complete','max_steps',1))
        with self.assertRaises(ValueError): load_checkpoint(path,{'fixture':'full_cannot_promote_probe'})

        ref=self.initialize('resumed',max_steps=2)
        agent.model.load_state_dict(saved['model'])
        agent.optimizer.load_state_dict(saved['optimizer'])
        agent.scheduler.load_state_dict(saved['scheduler'])
        agent.optimizer_updates=saved['optimizer_updates']
        restore_rng(saved['rng'],'cpu')
        self.assertTrue(self.run_loop(ref,start=saved['step']))
        self.assertEqual(agent.invocation_updates,2)
        for k,v in agent.model.state_dict().items(): self.assertTrue(torch.equal(v,expected[k]), k)
        self.assertFalse(torch.cuda.is_initialized())

    def test_nonfinite_update_and_pending_gradient_checkpoint_are_rejected(self):
        self.initialize('fail')
        next(agent.model.parameters()).grad=torch.full_like(next(agent.model.parameters()),float('nan'))
        before={k:v.clone() for k,v in agent.model.state_dict().items()}
        with self.assertRaises(FloatingPointError):
            checked_update([agent.model],[agent.optimizer],[agent.scheduler],1.)
        for k,v in agent.model.state_dict().items(): self.assertTrue(torch.equal(v,before[k]))
        with self.assertRaises(ValueError):
            save_checkpoint(agent.model,agent.optimizer,agent.scheduler,agent.args,agent.lm_config,
                agent.training_contract,epoch=0,step=1,optimizer_updates=0,invocation_updates=0,status='running')
        self.assertEqual(agent.scheduler.last_epoch,0)

    def test_real_local_tokenizer_template_observation_suffix(self):
        import jinja2
        import tokenizers
        config=json.loads((ROOT/'model/tokenizer_config.json').read_text(encoding='utf-8'))
        template=jinja2.Environment().from_string(config['chat_template'])
        backend=tokenizers.Tokenizer.from_file(str(ROOT/'model/tokenizer.json'))
        class LocalTokenizer:
            eos_token_id=2
            def apply_chat_template(self,messages,**kwargs): return template.render(messages=messages,**kwargs)
            def __call__(self,text,**kwargs): return {'input_ids':backend.encode(text,add_special_tokens=False).ids}
        tokenizer=LocalTokenizer()
        for thinking in (False,True):
            tool_messages=[{'role':'tool','content':'{"result":"42"}'}]
            suffix=tool_observation_tokens(tokenizer,tool_messages,tools=[],add_generation_prompt=True,
                open_thinking=thinking,already_ended=True)
            decoded=backend.decode(suffix,skip_special_tokens=False)
            self.assertTrue(decoded.startswith('\n<|im_start|>user\n<tool_response>'))
            self.assertIn('"result":"42"',decoded)
            self.assertTrue(decoded.endswith('<think>\n') if thinking else decoded.endswith('<think>\n\n</think>\n\n'))

    def test_real_tokenizer_multiturn_rollout_keeps_sampled_prefix_and_both_eos(self):
        import jinja2
        import tokenizers
        config=json.loads((ROOT/'model/tokenizer_config.json').read_text(encoding='utf-8'))
        template=jinja2.Environment().from_string(config['chat_template'])
        backend=tokenizers.Tokenizer.from_file(str(ROOT/'model/tokenizer.json'))
        class Encoded(dict):
            def to(self,device): return self
        class LocalTokenizer:
            eos_token_id=2
            pad_token_id=0
            def apply_chat_template(self,messages,**kwargs): return template.render(messages=messages,**kwargs)
            def __call__(self,text,**kwargs):
                ids=backend.encode(text,add_special_tokens=False).ids
                if kwargs.get('return_tensors')=='pt':
                    return Encoded(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long))
                return {'input_ids':ids}
            def decode(self,ids,**kwargs): return backend.decode(ids,skip_special_tokens=kwargs.get('skip_special_tokens',True))
        tokenizer=LocalTokenizer()
        first='\n\nreasoning\n\n</think>\n<tool_call>{"name":"calculate_math","arguments":{"expression":"6*7"}}</tool_call>'
        generated=[backend.encode(first,add_special_tokens=False).ids+[2], backend.encode('The answer is 42.',add_special_tokens=False).ids+[2]]
        class FakeEngine:
            def __init__(self): self.prompts=[]
            def rollout(self,prompt_ids,**kwargs):
                ids=generated[len(self.prompts)]+[0,17]
                self.prompts.append(prompt_ids[0].tolist())
                return SimpleNamespace(completion_ids=torch.tensor([ids]),per_token_logps=torch.full((1,len(ids)),-.1),completion_mask=torch.ones(1,len(ids)),completions=['unused'])
        engine=FakeEngine()
        result=agent.rollout_single(engine,tokenizer,[{'role':'user','content':'Calculate 6*7'}],
                                   [agent.TOOLS[0]],max_turns=2,thinking_ratio=1.,device='cpu')
        _,_,prompt,response,mask,old,_,unfinished=result
        self.assertFalse(unfinished)
        self.assertEqual(engine.prompts[1][:len(prompt)+len(generated[0])],prompt+generated[0])
        self.assertEqual([token for token,valid in zip(response,mask) if valid],generated[0]+generated[1])
        self.assertEqual(sum(token==2 for token,valid in zip(response,mask) if valid),2)
        self.assertTrue(any(valid==0 for valid in mask))
        self.assertTrue(all(logp==0 for logp,valid in zip(old,mask) if not valid))


if __name__=='__main__': unittest.main()
