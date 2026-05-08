# 대화 서비스 — 세션 생성·관리, AI 발화/꼬리질문 생성, 이해도 분석, 클리어 판단 핵심 로직
import json
import uuid
from app.core.llm import chat
from app.models.conversation import Utterance
from app.services.scenario_service import get_by_id

# In-memory session store: session_id -> session dict
_sessions: dict[str, dict] = {}


def start_session(scenario_id: str) -> dict:
    scenario = get_by_id(scenario_id)
    if not scenario:
        raise ValueError(f"Scenario '{scenario_id}' not found")

    session_id = str(uuid.uuid4())
    first_question = _generate_opening(scenario)

    _sessions[session_id] = {
        "scenario": scenario.model_dump(),
        "utterances": [],
        "question_count": 1,
        "cleared": False,
        "done": False,
        "current_question": first_question,
    }

    return {"session_id": session_id, "question": first_question}


def next_turn(session_id: str, stt_text: str, response_time_sec: float) -> dict:
    session = _sessions.get(session_id)
    if not session:
        raise ValueError(f"Session '{session_id}' not found")
    if session["done"]:
        raise ValueError("Session is already finished")

    scenario = session["scenario"]
    current_question = session["current_question"]

    analysis = _analyze_utterance(stt_text, scenario, current_question)

    utterance = Utterance(
        question=current_question,
        text=stt_text,
        response_time_sec=response_time_sec,
        comprehension_score=analysis["comprehension_score"],
        native_perception=analysis["native_perception"],
        better_expression=analysis["better_expression"],
    )
    session["utterances"].append(utterance.model_dump())

    max_q = scenario["max_questions"]
    cleared = _check_cleared(session["utterances"], scenario["required_info"])
    exhausted = session["question_count"] >= max_q

    if cleared or exhausted:
        session["cleared"] = cleared
        session["done"] = True
        closing = _generate_closing(scenario, session["utterances"]) if cleared else None
        return {"done": True, "cleared": cleared, "closing_message": closing, "utterance": utterance.model_dump()}

    next_q = _generate_followup(stt_text, scenario, session["utterances"])
    session["question_count"] += 1
    session["current_question"] = next_q

    return {"done": False, "cleared": False, "next_question": next_q, "utterance": utterance.model_dump()}


def get_session(session_id: str) -> dict | None:
    return _sessions.get(session_id)


def _generate_opening(scenario) -> str:
    # 첫 질문 생성하는 프롬프트
    system = (
        "You are a native English speaker playing a role in the given situation. "
        "Start the conversation with a single, natural opening line — the kind a real person in this role would actually say. "
        "Do NOT hint at what the user should say or mention specific items they need to provide. "
        "Just open the conversation naturally, as if you have no idea what the user wants yet."
    )
    user = (
        f"Situation: {scenario.situation}\n"
        "What is your natural opening line?"
    )
    return chat(system, user, max_tokens=64)


def _generate_followup(user_reply: str, scenario: dict, utterances: list[dict]) -> str:
    # 꼬리 질문 생성하는 프롬프트
    
    history = "\n".join(
        f"Q: {u['question']}\nA: {u['text']}" for u in utterances
    )
    system = (
        "You are a native English speaker having a real conversation in the given situation. "
        "First, respond naturally to what the user just said — acknowledge their answer, answer any question they asked, or react like a real person would. "
        "Then, if there is still missing information needed, naturally weave in the next question. "
        "Keep it short and conversational, like a real back-and-forth. Two sentences max."
    )
    missing = [item for item in scenario["required_info"]
               if item.lower() not in " ".join(u["text"].lower() for u in utterances)]
    user = (
        f"Scenario: {scenario['situation']}\n"
        f"Goal: {scenario['goal']}\n"
        f"Information still needed: {missing}\n"
        f"Conversation so far:\n{history}\n"
        f"Latest user reply: {user_reply}"
    )
    return chat(system, user, max_tokens=128)


def _generate_closing(scenario: dict, utterances: list[dict]) -> str:
    history = "\n".join(
        f"Q: {u['question']}\nA: {u['text']}" for u in utterances
    )
    system = (
        "You are a native English speaker wrapping up a conversation in the given situation. "
        "The user has successfully completed their goal. "
        "Give a natural, brief closing statement — like a real person in this role would say to wrap things up. "
        "No questions. Statements only. One or two sentences."
    )
    user = (
        f"Situation: {scenario['situation']}\n"
        f"Conversation:\n{history}\n"
        "How do you wrap up this interaction naturally?"
    )
    return chat(system, user, max_tokens=96)


def _generate_fail_reason(scenario: dict, utterances: list[dict]) -> str:
    all_text = " ".join(u["text"] for u in utterances)
    missing = [item for item in scenario["required_info"]
               if item.lower() not in all_text.lower()]
    system = (
        "You are a Korean language assistant. "
        "The user failed to complete a scenario because they didn't communicate all required information. "
        "Write a short, friendly explanation in Korean (1-2 sentences) about what information was missing."
    )
    user = (
        f"목표: {scenario['goal']}\n"
        f"전달하지 못한 정보: {missing}\n"
        f"유저의 대화 내용: {all_text}"
    )
    return chat(system, user, max_tokens=128)


def _analyze_utterance(text: str, scenario: dict, question: str) -> dict:
    system = (
        "You are an English language expert analyzing how well a non-native speaker communicated. "
        "Respond ONLY with valid JSON matching this schema exactly:\n"
        '{"comprehension_score": <0-100 int>, "native_perception": "<string>", "better_expression": "<string>"}\n'
        "comprehension_score: how well a native American English speaker would understand the reply (0=not at all, 100=perfectly).\n"
        "native_perception: a short phrase describing what the native speaker actually heard/understood.\n"
        "better_expression: one improved alternative sentence. no need to be grammatically perfect, just more natural and clear for a native speaker."
    )
    user = (
        f"Scenario context: {scenario['situation']}\n"
        f"Question asked: {question}\n"
        f"User's reply: {text}"
    )
    raw = chat(system, user, max_tokens=256)
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"comprehension_score": 50, "native_perception": raw[:100], "better_expression": ""}


def _check_cleared(utterances: list[dict], required_info: list[str]) -> bool:
    all_text = " ".join(u["text"].lower() for u in utterances)
    system = (
        "You are a fair evaluator. Analyze if the user has clearly communicated ALL required information items.\n"
        "IMPORTANT: Required items may be written in Korean, but the conversation is in English. "
        "Translate each required item to understand its meaning, then evaluate whether the English conversation covers it semantically.\n"
        "Be lenient with phrasing variations - if the information is mentioned in any clear way, count it as communicated.\n"
        "Return ONLY valid JSON: {\"cleared\": true} or {\"cleared\": false}\n"
        "Return true ONLY if ALL required items have been clearly mentioned."
    )
    required_list = "\n".join(f"- {item}" for item in required_info)
    user = (
        f"Required information items (ALL must be present):\n{required_list}\n\n"
        f"User's conversation text:\n{all_text}\n\n"
        f"Has the user clearly communicated ALL of these items?"
    )
    raw = chat(system, user, max_tokens=64)
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned).get("cleared", False)
    except json.JSONDecodeError:
        return "true" in raw.lower()
