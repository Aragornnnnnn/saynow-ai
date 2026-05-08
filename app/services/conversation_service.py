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
        return {"done": True, "cleared": cleared, "utterance": utterance.model_dump()}

    next_q = _generate_followup(stt_text, scenario, session["utterances"])
    session["question_count"] += 1
    session["current_question"] = next_q

    return {"done": False, "cleared": False, "next_question": next_q, "utterance": utterance.model_dump()}


def get_session(session_id: str) -> dict | None:
    return _sessions.get(session_id)


def _generate_opening(scenario) -> str:
    # 첫 질문 생성하는 프롬프트
    
    system = (
        "You are a native English speaker playing a role in the scenario. "
        "Your goal is to help the user complete their objective through natural conversation. "
        "You must ask a natural opening QUESTION that is directly relevant to the user's goal. "
        "No need to ask small talk questions, but you can say simple greetings - ask something that helps them achieve their objective. "
        "Keep it concise and natural. One sentence only."
    )
    user = (
        f"Scenario: {scenario.situation}\n"
        f"User's goal: {scenario.goal}\n"
        f"Required information to collect: {scenario.required_info}\n"
        f"What opening question would you ask to begin guiding the user toward their goal?"
    )
    return chat(system, user, max_tokens=128)


def _generate_followup(user_reply: str, scenario: dict, utterances: list[dict]) -> str:
    # 꼬리 질문 생성하는 프롬프트
    
    history = "\n".join(
        f"Q: {u['question']}\nA: {u['text']}" for u in utterances
    )
    system = (
        "You are a native English speaker in the scenario. "
        "Based on the conversation history, ask a short follow-up question "
        "to help guide the user toward the goal. One sentence only."
    )
    user = (
        f"Scenario: {scenario['situation']}\n"
        f"Goal: {scenario['goal']}\n"
        f"Conversation so far:\n{history}\n"
        f"Latest user reply: {user_reply}"
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
        return json.loads(raw)
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
    raw = chat(system, user, max_tokens=32)
    try:
        return json.loads(raw).get("cleared", False)
    except json.JSONDecodeError:
        return False
