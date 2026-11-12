import asyncio
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Support a standalone clone whose root contains the AstrBot plugin files.
if not (ROOT / "astrbot_plugin_voice_lab").is_dir():
    package = types.ModuleType("astrbot_plugin_voice_lab")
    package.__path__ = [str(ROOT)]
    sys.modules["astrbot_plugin_voice_lab"] = package
from astrbot_plugin_voice_lab.voice_state import VoiceStore


class StateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = VoiceStore(self.tmp.name)

    def test_persistence_confirmation_and_undo(self):
        state = self.store.load("a")
        state["candidate"] = self.store.candidate(state, -5, 0)
        self.store.save("a", state)
        restarted = VoiceStore(self.tmp.name)
        self.assertEqual(restarted.load("a")["current"]["rate"], 0)
        accepted = restarted.apply("a", "accept", "喜欢，保留")
        self.assertEqual(accepted["current"]["rate"], -5)
        self.assertEqual(restarted.apply("a", "undo")["current"]["rate"], 0)

    def test_isolation_and_path_safety(self):
        self.store.apply("a", "reject", "user a")
        self.assertEqual(self.store.load("b")["history"], [])
        self.assertEqual(self.store.path("../../escape").parent, Path(self.tmp.name))

    def test_adjustment_limits_and_one_parameter(self):
        state = self.store.load("a")
        for rate, pitch in ((6, 0), (1, 1), (0, 0.5), (True, 0), (float('nan'), 0)):
            with self.assertRaises(ValueError):
                self.store.candidate(state, rate, pitch)
        state["current"]["rate"] = -25
        with self.assertRaises(ValueError):
            self.store.candidate(state, -1, 0)

    def test_history_bounded_and_reject_preserves_current(self):
        for i in range(40):
            state = self.store.apply("a", "reject", str(i))
        self.assertEqual(len(state["history"]), 30)
        self.assertEqual(state["current"]["rate"], 0)
        self.assertEqual(state["history"][0]["feedback"], "10")


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        def decorate(*args, **kwargs):
            return lambda fn: fn
        class Star:
            def __init__(self, context): pass
        class Component:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        modules = {
            'edge_tts': types.SimpleNamespace(),
            'astrbot': types.ModuleType('astrbot'),
            'astrbot.api': types.SimpleNamespace(AstrBotConfig=dict,
                logger=types.SimpleNamespace(warning=lambda *a: None)),
            'astrbot.api.event': types.SimpleNamespace(AstrMessageEvent=object,
                filter=types.SimpleNamespace(llm_tool=decorate, command=decorate,
                    on_llm_request=decorate, on_decorating_result=decorate)),
            'astrbot.api.message_components': types.SimpleNamespace(Plain=type('Plain', (Component,), {}),
                Record=type('Record', (Component,), {})),
            'astrbot.api.provider': types.SimpleNamespace(ProviderRequest=object),
            'astrbot.api.star': types.SimpleNamespace(Context=object, Star=Star, register=decorate,
                StarTools=types.SimpleNamespace(get_data_dir=lambda name: self.tmp.name)),
        }
        self.old = {k: sys.modules.get(k) for k in modules}
        sys.modules.update(modules)
        sys.modules.pop('astrbot_plugin_voice_lab.main', None)
        self.module = importlib.import_module('astrbot_plugin_voice_lab.main')
        self.plugin = self.module.VoiceLab(None, {})
        self.extras = {}
        self.event = types.SimpleNamespace(get_group_id=lambda: '', get_sender_id=lambda: 'user1',
            unified_msg_origin='qq:private:1', chain_result=lambda x: x, send=AsyncMock(),
            set_extra=self.extras.__setitem__, get_extra=lambda key: self.extras.get(key))

    async def asyncTearDown(self):
        for key, value in self.old.items():
            if value is None: sys.modules.pop(key, None)
            else: sys.modules[key] = value
        sys.modules.pop('astrbot_plugin_voice_lab.main', None)

    async def test_failed_preview_never_changes_profile(self):
        self.plugin.synthesize = AsyncMock(side_effect=RuntimeError('network unavailable'))
        result = await self.plugin.voice_preview(self.event, -5, 0, 'test', 'too fast')
        self.assertIn('试听失败', result)
        state = self.plugin.store.load(self.plugin.key(self.event))
        self.assertIsNone(state['candidate'])
        self.assertEqual(state['current']['rate'], 0)
        self.event.send.assert_not_called()

    async def test_preview_send_failure_never_saves_candidate(self):
        self.plugin.synthesize = AsyncMock(return_value='/test.mp3')
        self.event.send.side_effect = RuntimeError('QQ upload failed')
        await self.plugin.voice_preview(self.event, -5, 0, 'test', 'too fast')
        self.assertIsNone(self.plugin.store.load(self.plugin.key(self.event))['candidate'])

    async def test_preview_accept_and_use_saved_voice(self):
        self.plugin.synthesize = AsyncMock(return_value='/test.mp3')
        await self.plugin.voice_preview(self.event, -5, 0, 'test', 'too fast')
        self.assertEqual(self.plugin.store.load(self.plugin.key(self.event))['current']['rate'], 0)
        self.assertEqual(json.loads(await self.plugin.voice_control(self.event, 'accept', 'keep'))['current']['rate'], -5)
        self.extras.clear()
        req = types.SimpleNamespace(system_prompt='persona')
        await self.plugin.voice_instructions(self.event, req)
        await self.plugin.choose_output(self.event, 'voice')
        result = types.SimpleNamespace(chain=[self.module.Plain(text='hello')], is_llm_result=lambda: True)
        self.event.get_result = lambda: result
        await self.plugin.append_voice(self.event)
        self.assertEqual(self.plugin.synthesize.call_args.args[1]['rate'], -5)
        self.assertEqual(len(result.chain), 1)
        self.assertIsInstance(result.chain[0], self.module.Record)

    async def test_group_rejected_and_existing_audio_not_duplicated(self):
        self.event.get_group_id = lambda: 'group'
        self.assertIn('仅支持私聊', await self.plugin.voice_control(self.event, 'status', ''))
        self.event.get_group_id = lambda: ''
        self.extras['voice_lab_ai'] = True
        self.extras['voice_lab_output'] = 'voice'
        self.event.get_result = lambda: types.SimpleNamespace(chain=[self.module.Record(file='/a.mp3')], is_llm_result=lambda: True)
        self.plugin.synthesize = AsyncMock()
        await self.plugin.append_voice(self.event)
        self.plugin.synthesize.assert_not_called()

    async def test_model_error_is_never_synthesized(self):
        self.extras['voice_lab_ai'] = True
        self.extras['voice_lab_output'] = 'voice'
        result = types.SimpleNamespace(chain=[self.module.Plain(text='LLM 响应错误: 422 input_audio')],
                                       is_llm_result=lambda: False)
        self.event.get_result = lambda: result
        self.plugin.synthesize = AsyncMock()
        await self.plugin.append_voice(self.event)
        self.plugin.synthesize.assert_not_called()
        self.assertEqual(len(result.chain), 1)

    async def test_text_choice_and_no_choice_do_not_synthesize(self):
        self.extras['voice_lab_ai'] = True
        result = types.SimpleNamespace(chain=[self.module.Plain(text='hello')], is_llm_result=lambda: True)
        self.event.get_result = lambda: result
        self.plugin.synthesize = AsyncMock()
        await self.plugin.append_voice(self.event)
        await self.plugin.choose_output(self.event, 'text')
        await self.plugin.append_voice(self.event)
        self.plugin.synthesize.assert_not_called()
        self.assertIsInstance(result.chain[0], self.module.Plain)

    async def test_voice_choice_removes_all_text_copies(self):
        self.extras['voice_lab_ai'] = True
        image = object()
        result = types.SimpleNamespace(chain=[self.module.Plain(text='hello'), image,
            self.module.Plain(text='world')], is_llm_result=lambda: True)
        self.event.get_result = lambda: result
        self.plugin.synthesize = AsyncMock(return_value='/voice.mp3')
        await self.plugin.choose_output(self.event, 'voice')
        await self.plugin.append_voice(self.event)
        self.assertEqual(len(result.chain), 2)
        self.assertIsInstance(result.chain[0], self.module.Record)
        self.assertIs(result.chain[1], image)
        self.assertFalse(any(isinstance(c, self.module.Plain) for c in result.chain))

    async def test_failed_voice_keeps_text_and_does_not_claim_sent(self):
        self.extras['voice_lab_ai'] = True
        result = types.SimpleNamespace(chain=[self.module.Plain(text='hello')], is_llm_result=lambda: True)
        self.event.get_result = lambda: result
        self.plugin.synthesize = AsyncMock(side_effect=RuntimeError('offline'))
        await self.plugin.choose_output(self.event, 'voice')
        await self.plugin.append_voice(self.event)
        self.assertEqual(result.chain[0].text, 'hello')
        self.assertFalse(any(isinstance(c, self.module.Record) for c in result.chain))

    async def test_disabled_and_invalid_choice_do_not_enable_voice(self):
        self.plugin.config['auto_voice'] = False
        self.assertIn('关闭', await self.plugin.choose_output(self.event, 'voice'))
        self.assertIsNone(self.extras.get('voice_lab_output'))
        self.assertIn('失败', await self.plugin.choose_output(self.event, 'both'))


if __name__ == '__main__':
    unittest.main()
