# 턴 평가 서비스 — STT → 발화 분석 → 슬롯 추출 → 시나리오 상태 판단 → 응답 생성
import json
from app.core.llm import chat
from app.models.turn_evaluation import FilledSlot, TtsContent, TurnEvaluationResponse
from app.services.stt_service import transcribe_with_confidence
from app.services.tts_service import synthesize
from app.services.scenario_service import get_by_id


def evaluate_turn(
    audio_bytes: bytes,
    filename: str,
    scenario_id: str,
    scenario_situation: str,
    scenario_goal: str,
    required_keys: list[str],
    max_follow_up_count: int,
    current_question: str,
    filled_slots: list[FilledSlot],
    conversation_history: list[dict],
) -> TurnEvaluationResponse:
    # 1. STT
    stt_result = transcribe_with_confidence(audio_bytes, filename)
    transcript = stt_result["text"].strip()
    stt_confidence = stt_result["confidence"]

    # 2. 시나리오에서 required_info(슬롯 키 목록) 로드
    scenario = get_by_id(scenario_id)
    if not scenario:
        raise ValueError(f"Scenario '{scenario_id}' not found")

    # 3. 이번 턴에서 새로 채워진 슬롯 추출
    existing_keys = {s.slotKey for s in filled_slots}
    new_slots = _extract_slots(transcript, required_keys, filled_slots)

    # 4. 누적 슬롯 (기존 + 신규)
    all_filled_keys = existing_keys | {s.slotKey for s in new_slots}

    # 5. scenarioStatus 결정
    all_covered = all(key in all_filled_keys for key in required_keys)
    follow_up_count = _count_follow_ups(conversation_history)

    scenario_status = "SUCCESS" if all_covered else "IN_PROGRESS"

    # 6. 다음 질문 or 결과 메시지 생성
    if scenario_status == "SUCCESS":
        closing_text = _generate_closing(scenario_situation, conversation_history, transcript)
        tts_audio = synthesize(closing_text)
        return TurnEvaluationResponse(
            transcript=transcript,
            sttConfidence=stt_confidence,
            scenarioStatus="SUCCESS",
            filledSlots=new_slots,
            resultMessage=TtsContent(messageText=closing_text, ttsAudio=tts_audio),
        )
    elif follow_up_count >= max_follow_up_count:
        failure_text = _generate_failure(scenario_situation, conversation_history, transcript)
        tts_audio = synthesize(failure_text)
        return TurnEvaluationResponse(
            transcript=transcript,
            sttConfidence=stt_confidence,
            scenarioStatus="FAILURE",
            filledSlots=new_slots,
            resultMessage=TtsContent(messageText=failure_text, ttsAudio=tts_audio),
        )
    else:
        missing_keys = [k for k in required_keys if k not in all_filled_keys]
        next_q_text = _generate_followup(
            transcript, scenario_situation, missing_keys, conversation_history
        )
        tts_audio = synthesize(next_q_text)
        return TurnEvaluationResponse(
            transcript=transcript,
            sttConfidence=stt_confidence,
            scenarioStatus="IN_PROGRESS",
            filledSlots=new_slots,
            nextQuestion=TtsContent(questionText=next_q_text, ttsAudio=tts_audio),
        )


def _count_follow_ups(history: list[dict]) -> int:
    assistant_turns = sum(1 for turn in history if turn.get("role") == "assistant")
    # 시작 질문 1회는 제외하고, 이후 꼬리질문만 계산한다.
    return max(0, assistant_turns - 1)


def _extract_slots(
    transcript: str,
    required_keys: list[str],
    existing_slots: list[FilledSlot],
) -> list[FilledSlot]:
    """이번 턴 발화에서 새로 채워진 슬롯만 추출"""
    existing_summary = ", ".join(f"{s.slotKey}={s.slotValue}" for s in existing_slots) or "none"
    keys_list = "\n".join(f"- {k}" for k in required_keys)
    system = (
        "You are an information extractor. Given a user's spoken reply, extract any values "
        "that match the required slot keys. Return ONLY valid JSON: "
        '{"slots": [{"slotKey": "...", "slotValue": "..."}]}\n'
        "Only include slots newly mentioned in this reply. "
        "Do not repeat slots already filled. If nothing new is mentioned, return {\"slots\": []}."
    )
    user = (
        f"Required slot keys:\n{keys_list}\n\n"
        f"Already filled slots: {existing_summary}\n\n"
        f"User's reply: {transcript}"
    )
    raw = chat(system, user, max_tokens=256)
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        return [FilledSlot(**s) for s in data.get("slots", [])]
    except (json.JSONDecodeError, Exception):
        return []


def _generate_followup(
    user_reply: str,
    situation: str,
    missing_keys: list[str],
    history: list[dict],
) -> str:
    history_str = "\n".join(
        f"{'AI' if h['role'] == 'assistant' else 'User'}: {h['content']}" for h in history
    )
    missing_str = ", ".join(missing_keys)
    system = (
        "You are a native English speaker having a real conversation in the given situation. "
        "First, respond naturally to what the user just said. "
        "Then naturally weave in a question to gather the still-missing information. "
        "Two sentences max. No lists."
    )
    user = (
        f"Situation: {situation}\n"
        f"Conversation so far:\n{history_str}\n"
        f"User just said: {user_reply}\n"
        f"Still need: {missing_str}"
    )
    return chat(system, user, max_tokens=128)


def _generate_closing(
    situation: str,
    history: list[dict],
    last_reply: str,
) -> str:
    history_str = "\n".join(
        f"{'AI' if h['role'] == 'assistant' else 'User'}: {h['content']}" for h in history
    )
    system = (
        "You are a native English speaker wrapping up a conversation. "
        "The user has successfully provided all needed information. "
        "Give a brief, natural closing statement. One or two sentences. No questions."
    )
    user = (
        f"Situation: {situation}\n"
        f"Conversation:\n{history_str}\n"
        f"User's last reply: {last_reply}"
    )
    return chat(system, user, max_tokens=96)


def _generate_failure(
    situation: str,
    history: list[dict],
    last_reply: str,
) -> str:
    history_str = "\n".join(
        f"{'AI' if h['role'] == 'assistant' else 'User'}: {h['content']}" for h in history
    )
    system = (
        "You are a native English speaker ending a conversation because the scenario was not completed in time. "
        "Give a brief, natural failure message. One or two sentences. No questions."
    )
    user = (
        f"Situation: {situation}\n"
        f"Conversation:\n{history_str}\n"
        f"User's last reply: {last_reply}"
    )
    return chat(system, user, max_tokens=96)
