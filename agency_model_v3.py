# -*- coding: utf-8 -*-
"""
AI Agency Model — v3
- Multi-agent workflow
- LLMProvider interface (plug-and-play)
- End-to-end demo run
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum, auto
import argparse, json, random, textwrap

# ========= LLM LAYER =========

class LLMProvider:
    """Интерфейс за LLM. Смени с реална имплементация (OpenAI, Azure, Anthropic и т.н.)."""
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

class EchoLLM(LLMProvider):
    """Демо провайдър — имитира отговор на модел, в реалност връща 'синтезирани' идеи."""
    def generate(self, prompt: str, **kwargs) -> str:
        # Много прост шаблон; ти можеш да вкараш истински LLM тук.
        return textwrap.dedent(f"""
        [LLM SYNTHESIS]
        - Анализ на бриф: кратки срокове; фокус върху бързо нанасяне и покривност.
        - Инсайт: хората искат ремонтът да отнеме 1 уикенд.
        - Идеи (hooks): "Една ръка, голям ефект", "Свърши го за следобед", "Чисто нанасяне, чист уикенд".
        - Хаштагове: #DIY #WeekendMakeover #ЛекоТотал
        """).strip()

# ========= DATA MODELS =========

@dataclass
class Brief:
    client: str
    brand: str
    product: str
    objectives: List[str]
    kpis: List[str]
    target_audience: str
    budget_eur: float
    timing_from: str
    timing_to: str
    mandatories: List[str] = field(default_factory=list)
    tone_of_voice: Optional[str] = None
    channels_pref: List[str] = field(default_factory=list)
    notes: Optional[str] = None

@dataclass
class Debrief:
    clarified_points: List[str]
    risks: List[str]
    missing_info: List[str]
    approved_tone_of_voice: str
    updated_constraints: List[str] = field(default_factory=list)
    llm_notes: Optional[str] = None

@dataclass
class Strategy:
    brand_promise: str
    key_message: str
    reason_to_believe: str
    audience_insight: str
    positioning: str
    channel_roles: Dict[str, str]
    content_pillars: List[str]

@dataclass
class CreativeConcept:
    territory: str
    big_idea: str
    visual_directions: List[str]
    copy_hooks: List[str]
    hashtags: List[str]

@dataclass
class CopyAsset:
    channel: str
    headline: str
    body: str
    cta: str

@dataclass
class MediaLine:
    channel: str
    objective: str
    budget_eur: float
    kpi: str

@dataclass
class MediaPlan:
    total_budget_eur: float
    allocation: List[MediaLine]
    timeline_weeks: List[str]

@dataclass
class Deliverable:
    debrief: Debrief
    strategy: Strategy
    creative: CreativeConcept
    copies: List[CopyAsset]
    media_plan: MediaPlan
    qa_notes: List[str]

# ========= TOOLS =========

class BrandVoiceTool:
    @staticmethod
    def polish(text: str, tone: str) -> str:
        tone_map = {
            "професионален": ["точно", "уверено", "ясно"],
            "приятелски": ["достъпно", "топло", "усмихнато"],
            "инспириращ": ["вдъхновяващо", "смело", "визионерско"],
        }
        tags = tone_map.get((tone or "").lower(), [tone or ""])
        return f"{text.strip()} ({', '.join(t for t in tags if t)})".strip()

class ResearchTool:
    @staticmethod
    def competitor_snapshot(brand: str, product: str) -> Dict[str, Any]:
        return {
            "brand": brand,
            "product": product,
            "competitors": [
                {"name": "CompA", "usp": "по-бързо изсъхване", "price_index": 1.1},
                {"name": "CompB", "usp": "по-висока покривност", "price_index": 0.9},
            ],
        }

class PostTemplateTool:
    @staticmethod
    def format_post(channel: str, hook: str, body: str, cta: str, hashtags: List[str]) -> str:
        if channel.lower() in {"facebook", "instagram"}:
            return textwrap.dedent(f"""
            {hook}

            {body}

            {cta}
            {' '.join('#'+h for h in hashtags)}
            """).strip()
        return f"{hook} — {body} | {cta} | {' '.join('#'+h for h in hashtags)}"

# ========= AGENTS =========

class Agent:
    name: str = "Agent"
    def log(self, msg: str) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{self.name}] {msg}")

class AccountManagerAgent(Agent):
    name = "AccountManager"
    def __init__(self, llm: Optional[LLMProvider] = None) -> None:
        self.llm = llm or EchoLLM()

    def debrief(self, brief: Brief) -> Debrief:
        self.log("Дебриф + LLM синтез...")
        clarified = [
            "KPIs приоритизирани: Reach > CTR > Conversions",
            f"Бюджет валидиран: {brief.budget_eur:.0f} EUR",
        ]
        risks = ["Кратък тайминг", "Ограничени продуктови визуали"]
        missing = ["Актуален бранд кит (лого, палитра)", "Списък с неразрешени твърдения"]
        tone = brief.tone_of_voice or "професионален"

        llm_notes = self.llm.generate(
            prompt=f"Направи кратък синтез на бриф за {brief.brand} {brief.product} "
                   f"към аудитория: {brief.target_audience}. KPIs: {', '.join(brief.kpis)}."
        )
        return Debrief(clarified_points=clarified, risks=risks, missing_info=missing,
                       approved_tone_of_voice=tone, llm_notes=llm_notes)

class StrategistAgent(Agent):
    name = "Strategist"
    def craft_strategy(self, brief: Brief, research: Dict[str, Any]) -> Strategy:
        self.log("Изграждане на стратегия...")
        channels = brief.channels_pref or ["Facebook", "Instagram", "YouTube", "Google Search"]
        channel_roles = {ch: ("Awareness" if ch in ("Facebook","Instagram","YouTube") else "Consideration")
                         for ch in channels}
        pillars = ["Ползи/покривност", "Лесно нанасяне", "Демонстрации/UGC"]
        return Strategy(
            brand_promise=f"{brief.brand}: безкомпромисно покритие за по-малко време.",
            key_message=f"Новият {brief.product} пести време с отлична покривност.",
            reason_to_believe="Формула 2-в-1: грунд + боя; бързо изсъхване.",
            audience_insight="Ремонтът да се случи за 1 уикенд, чисто и бързо.",
            positioning="Умно решение за безстрес боядисване.",
            channel_roles=channel_roles,
            content_pillars=pillars,
        )

class CreativeDirectorAgent(Agent):
    name = "CreativeDirector"
    def concept(self, strategy: Strategy, debrief: Debrief) -> CreativeConcept:
        self.log("Креативна територия + hooks от LLM бележките...")
        hooks = [
            "Една ръка, голям ефект",
            "Свърши го за следобед",
            "Чисто нанасяне, чист уикенд",
        ]
        # Допълнение от llm_notes, ако има:
        if debrief.llm_notes:
            for line in debrief.llm_notes.splitlines():
                line = line.strip("-• ").strip()
                if line and len(line) < 80 and "Идеи" not in line and "LLM" not in line:
                    hooks.append(line)
        hooks = list(dict.fromkeys(hooks))  # уникални
        return CreativeConcept(
            territory="Weekend Makeover",
            big_idea="Спестяваш време, печелиш уикенда.",
            visual_directions=["Преди/След", "Валяк в движение", "Цветни палитри"],
            copy_hooks=hooks,
            hashtags=["DIY","WeekendMakeover","ЛекоТотал","Боядисване"],
        )

class CopywriterAgent(Agent):
    name = "Copywriter"
    def generate_copies(self, concept: CreativeConcept, strategy: Strategy, tone: str) -> List[CopyAsset]:
        self.log("Генерирам копита по канали...")
        channels = ["Facebook","Instagram","YouTube Shorts","Google Search"]
        copies: List[CopyAsset] = []
        for ch in channels:
            hook = random.choice(concept.copy_hooks)
            body = f"{strategy.key_message} {strategy.reason_to_believe} Подходящо за бързи трансформации."
            if ch == "Google Search":
                headline = hook
                body_txt = strategy.key_message
            else:
                headline = hook
                body_txt = PostTemplateTool.format_post(ch, hook, body, "Виж повече", concept.hashtags)
            copies.append(CopyAsset(
                channel=ch,
                headline=BrandVoiceTool.polish(headline, tone),
                body=BrandVoiceTool.polish(body_txt, tone),
                cta="Купи сега"
            ))
        return copies

class MediaPlannerAgent(Agent):
    name = "MediaPlanner"
    def plan(self, brief: Brief) -> MediaPlan:
        self.log("Медиаплан...")
        channels = brief.channels_pref or ["Facebook","Instagram","YouTube","Google Search"]
        allocation: List[MediaLine] = []
        remaining = brief.budget_eur
        for i, ch in enumerate(channels):
            portion = round(brief.budget_eur * (0.4 if i == 0 else 0.2), 2)
            remaining -= portion
            allocation.append(MediaLine(
                channel=ch,
                objective="Awareness" if ch in ("Facebook","Instagram","YouTube") else "Intent",
                budget_eur=portion,
                kpi="CPM" if ch != "Google Search" else "CPC"
            ))
        if remaining > 0:
            allocation[0].budget_eur += remaining
        return MediaPlan(
            total_budget_eur=brief.budget_eur,
            allocation=allocation,
            timeline_weeks=["W1","W2","W3","W4"]
        )

class QAAgent(Agent):
    name = "QA"
    def review(self, brief: Brief, strategy: Strategy, creative: CreativeConcept, copies: List[CopyAsset]) -> List[str]:
        self.log("QA проверка...")
        notes = []
        for m in brief.mandatories:
            present_any = any(m.lower() in (c.body + c.headline).lower() for c in copies)
            if not present_any:
                notes.append(f"Липсва mandatory '{m}' в копитата.")
        if "Conversions" in brief.kpis and all(c.channel != "Google Search" for c in copies):
            notes.append("Добавете performance канал за конверсии (Search/Shopping).")
        if not notes:
            notes.append("QA OK.")
        return notes

# ========= WORKFLOW =========

class Step(Enum):
    INTAKE = auto()
    DEBRIEF = auto()
    STRATEGY = auto()
    CREATIVE = auto()
    COPY = auto()
    MEDIA = auto()
    QA = auto()
    PACKAGE = auto()

@dataclass
class ProjectState:
    brief: Optional[Brief] = None
    debrief: Optional[Debrief] = None
    strategy: Optional[Strategy] = None
    creative: Optional[CreativeConcept] = None
    copies: List[CopyAsset] = field(default_factory=list)
    media_plan: Optional[MediaPlan] = None
    qa_notes: List[str] = field(default_factory=list)

class Orchestrator(Agent):
    name = "Orchestrator"
    def __init__(self, llm: Optional[LLMProvider] = None) -> None:
        self.account = AccountManagerAgent(llm=llm)
        self.strategy_agent = StrategistAgent()
        self.creative = CreativeDirectorAgent()
        self.copy = CopywriterAgent()
        self.media = MediaPlannerAgent()
        self.qa = QAAgent()

    def run(self, brief: Brief) -> Deliverable:
        self.log("Старт → INTAKE")
        state = ProjectState(brief=brief)
        state.debrief = self.account.debrief(brief)
        research = ResearchTool.competitor_snapshot(brief.brand, brief.product)
        state.strategy = self.strategy_agent.craft_strategy(brief, research)
        state.creative = self.creative.concept(state.strategy, state.debrief)
        state.copies = self.copy.generate_copies(state.creative, state.strategy, state.debrief.approved_tone_of_voice)
        state.media_plan = self.media.plan(brief)
        state.qa_notes = self.qa.review(brief, state.strategy, state.creative, state.copies)
        return Deliverable(
            debrief=state.debrief,
            strategy=state.strategy,
            creative=state.creative,
            copies=state.copies,
            media_plan=state.media_plan,
            qa_notes=state.qa_notes,
        )

# ========= DEMO =========

def demo_input() -> Brief:
    return Brief(
        client="Protecta",
        brand="LEKO",
        product="Total Paint",
        objectives=["Awareness","Drive to Site"],
        kpis=["Reach","CTR","Conversions"],
        target_audience="Домaкинства 25-45, активни DIY потребители",
        budget_eur=12000.0,
        timing_from="2025-11-01",
        timing_to="2025-12-15",
        mandatories=["2-в-1","бързо изсъхване"],
        tone_of_voice="професионален",
        channels_pref=["Facebook","Instagram","YouTube","Google Search"],
        notes="Лансиране Q4; Q1 ремаркетинг.",
    )

def serialize_deliverable(deliv: Deliverable) -> str:
    def default(o):
        if hasattr(o, "__dict__"):
            return o.__dict__
        return asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)
    return json.dumps(deliv, default=default, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run end-to-end demo")
    args = parser.parse_args()

    if args.demo:
        orch = Orchestrator(llm=EchoLLM())
        brief = demo_input()
        result = orch.run(brief)
        print("\n===== DELIVERABLE (JSON) =====")
        print(serialize_deliverable(result))

        print("\n===== SUMMARY =====")
        print("- Brand promise:", result.strategy.brand_promise)
        print("- Big idea:", result.creative.big_idea)
        print("- QA:", "; ".join(result.qa_notes))
    else:
        print("Usage: python agency_model_v3.py --demo")

if __name__ == "__main__":
    main()
