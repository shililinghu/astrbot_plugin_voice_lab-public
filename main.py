import asyncio
import json
import time
import uuid
from pathlib import Path

import edge_tts
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain, Record
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools, register

from .voice_state import VoiceStore


RULES = """
你可以使用声音养成工具，为自己的声音逐步调音。
基础音色固定为 zh-CN-XiaoxiaoNeural。先用 voice_lab_control 的 status 查询已保存的设置与反馈。
用户要求优化声音、说太快或太尖等时，调用 voice_lab_preview，每次只微调一个参数。
预览会合成真实语音供用户试听。不要声称自己听过、能自评听感或在训练新的声线。
候选音色不会自动替换正式设置；用户明确表示喜欢并要求保留后，才用 voice_lab_control 的 accept。
用户不喜欢则 reject，要求恢复则 undo。不要反复调整已经得到认可的参数。
每次普通聊天回复，自行选择文字或语音，只使用一种。用户明确指定时优先遵从。
选择语音时先调用 voice_lab_output，mode 填 voice，然后正常生成简短口语回复；插件会只发送语音条，不发文字副本。
选择文字时调用 voice_lab_output，mode 填 text，或直接文字回复。不必每条都用语音，不要解释你正在选择形式。
语音适合简短问候、安慰和表达情绪；代码、清单、长解释适合文字。按聊天语境与人设选择。
不要另行发送“语音转文字”、字幕或重复同一段内容。无需声称只能输出文字。
工具失败时坦诚说明，不能宣称已发送、修改或保存成功。
"""


@register("astrbot_plugin_voice_lab", "Local", "声音试听、微调、保存与回退", "0.1.2")
class VoiceLab(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        root = Path(StarTools.get_data_dir("astrbot_plugin_voice_lab"))
        self.store = VoiceStore(root / "profiles", config.get("base_voice", "zh-CN-XiaoxiaoNeural"))
        self.audio = root / "audio"
        self.audio.mkdir(parents=True, exist_ok=True)
        self.lock = asyncio.Lock()
        self.synthesis_lock = asyncio.Lock()

    def key(self, event):
        if event.get_group_id():
            raise ValueError("声音养成第一版仅支持私聊，请私聊机器人操作")
        return str(event.unified_msg_origin) + ":" + event.get_sender_id()

    @staticmethod
    def describe(state):
        return json.dumps(state, ensure_ascii=False)

    async def synthesize(self, text, profile):
        text = text.strip()
        limit = min(600, max(20, int(self.config.get("max_chars", 300))))
        if not text or len(text) > limit:
            raise ValueError(f"语音文字应为 1 到 {limit} 字，请缩短后重试")
        async with self.synthesis_lock:
            # Retain recent files for platform uploads; bound disk use across restarts.
            files = sorted(self.audio.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
            now = time.time()
            for i, old in enumerate(files):
                if now - old.stat().st_mtime > 86400 or (len(files) - i > 100 and now - old.stat().st_mtime > 3600):
                    old.unlink(missing_ok=True)
            target = self.audio / (uuid.uuid4().hex + ".mp3")
            try:
                stream = edge_tts.Communicate(text, profile["voice"],
                    rate=f'{profile["rate"]:+d}%', pitch=f'{profile["pitch"]:+d}Hz')
                await asyncio.wait_for(stream.save(str(target)), timeout=60)
                if not target.exists() or target.stat().st_size == 0:
                    raise RuntimeError("语音服务返回了空文件")
                return str(target.resolve())
            except Exception:
                target.unlink(missing_ok=True)
                raise

    @filter.on_llm_request()
    async def voice_instructions(self, event: AstrMessageEvent, req: ProviderRequest):
        if event.get_group_id():
            return
        event.set_extra("voice_lab_ai", True)
        req.system_prompt = (req.system_prompt or "") + RULES

    @filter.llm_tool(name="voice_lab_preview")
    async def voice_preview(self, event: AstrMessageEvent, rate_delta: int,
                            pitch_delta: int, text: str, feedback: str) -> str:
        """根据用户反馈小幅调整声音并发送候选试听，成功后等待用户确认才替换正式声音。

        Args:
            rate_delta(number): 相对正式语速的变化，整数 -5 到 5，单位百分比；0 表示不变。
            pitch_delta(number): 相对正式音调的变化，整数 -5 到 5，单位 Hz；与语速只能改一项。
            text(string): 试听短句，建议与上次相同，最多 300 字。
            feedback(string): 用户实际反馈或首次选择理由，不得虚构试听反馈。
        """
        try:
            key = self.key(event)
            async with self.lock:
                state = self.store.load(key)
                candidate = self.store.candidate(state, rate_delta, pitch_delta)
                path = await self.synthesize(text, candidate)
                await event.send(event.chain_result([Record(file=path)]))
                state["candidate"] = candidate
                self.store.remember(state, "preview", feedback)
                self.store.save(key, state)
            event.set_extra("voice_lab_preview", True)
            return "已提交候选试听语音给 QQ；请用户确认能听到及是否喜欢。正式设置尚未改变。" + self.describe(candidate)
        except Exception as exc:
            logger.warning("声音养成试听失败: %s", exc)
            return f"试听失败，未保存新候选：{exc}"

    @filter.llm_tool(name="voice_lab_control")
    async def voice_control(self, event: AstrMessageEvent, action: str, feedback: str) -> str:
        """查询声音与试听反馈，或根据用户明确要求保存候选、拒绝候选、回退。

        Args:
            action(string): status 查询；accept 用户认可并要求保留；reject 拒绝候选；undo 恢复上一版。
            feedback(string): 用户本轮真实反馈，status 时可填空字符串。
        """
        try:
            key = self.key(event)
            async with self.lock:
                state = self.store.load(key) if action == "status" else self.store.apply(key, action, feedback)
            return self.describe(state)
        except Exception as exc:
            return f"操作失败：{exc}"

    @filter.llm_tool(name="voice_lab_output")
    async def choose_output(self, event: AstrMessageEvent, mode: str) -> str:
        """选择本轮最终回复的形式。语音只发语音条，文字只发文字，不生成文字副本。

        Args:
            mode(string): voice 表示语音回复，text 表示文字回复。按语境自行选择，用户指定时遵从。
        """
        if event.get_group_id():
            return "当前仅支持私聊，群聊请使用文字。"
        if mode not in ("voice", "text"):
            return "选择失败：mode 只能为 voice 或 text。"
        if mode == "voice" and not self.config.get("auto_voice", True):
            return "插件语音开关已关闭，本轮请使用文字。"
        event.set_extra("voice_lab_output", mode)
        return "本轮已选择" + ("语音。请正常生成简短口语回复，随后由插件合成；此时尚未发送，不要附加字幕。" if mode == "voice" else "文字。直接生成文字回复即可。")

    @filter.command("声音状态")
    async def status(self, event: AstrMessageEvent):
        yield event.plain_result(await self.voice_control(event, "status", ""))

    @filter.command("声音确认")
    async def accept(self, event: AstrMessageEvent):
        yield event.plain_result(await self.voice_control(event, "accept", "用户通过指令确认保留"))

    @filter.command("声音回退")
    async def undo(self, event: AstrMessageEvent):
        yield event.plain_result(await self.voice_control(event, "undo", "用户通过指令要求回退"))

    @filter.command("声音试听")
    async def sample(self, event: AstrMessageEvent):
        result = await self.voice_preview(event, 0, 0,
            "晚上好，今天过得怎么样？我们慢慢聊，我在这里听你说。", "用户请求试听当前声音")
        yield event.plain_result(result)

    @filter.on_decorating_result()
    async def append_voice(self, event: AstrMessageEvent):
        if not self.config.get("auto_voice", True) or not event.get_extra("voice_lab_ai"):
            return
        if event.get_group_id() or event.get_extra("voice_lab_preview"):
            return
        if event.get_extra("voice_lab_output") != "voice":
            return
        result = event.get_result()
        # Agent errors and plugin command results are not conversational replies.
        # An LLM request starting does not imply that it succeeded.
        if result is None or not result.is_llm_result():
            return
        if any(isinstance(c, Record) for c in result.chain):
            return
        text = "\n".join(c.text for c in result.chain if isinstance(c, Plain)).strip()
        if not text:
            return
        try:
            profile = self.store.load(self.key(event))["current"]
            path = await self.synthesize(text, profile)
            # Synthesize first; on failure keep the original text intact.
            # Replace all spoken text with one record, preserving unrelated media.
            chain = []
            inserted = False
            for component in result.chain:
                if isinstance(component, Plain):
                    if not inserted:
                        chain.append(Record(file=path))
                        inserted = True
                else:
                    chain.append(component)
            result.chain = chain
        except Exception as exc:
            logger.warning("声音养成自动语音失败: %s", exc)
            result.chain.append(Plain(text="\n[本次语音合成失败，文字已保留；请检查声音养成日志。]"))
