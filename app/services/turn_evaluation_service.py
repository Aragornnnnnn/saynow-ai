# 턴 평가 서비스 — STT → 발화 분석 → 슬롯 추출 → 시나리오 상태 판단 → 응답 생성
import json
from app.core.llm import chat
from app.models.turn_evaluation import FilledSlot, TtsContent, TurnEvaluationResponse
from app.services.stt_service import transcribe_with_confidence
from app.services.tts_service import synthesize


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

    # 2. 이번 턴에서 새로 채워진 슬롯 추출
    existing_keys = {s.slotKey for s in filled_slots}
    new_slots = _extract_slots(transcript, required_keys, filled_slots, conversation_history)

    # 3. 누적 슬롯 (기존 + 신규)
    all_filled_keys = existing_keys | {s.slotKey for s in new_slots}

    # 4. scenarioStatus 결정
    all_covered = all(key in all_filled_keys for key in required_keys)
    follow_up_count = _count_follow_ups(conversation_history)

    scenario_status = "SUCCESS" if all_covered else "IN_PROGRESS"

    # 5. 다음 질문 or 결과 메시지 생성
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
    history: list[dict],
) -> list[FilledSlot]:
    existing_summary = ", ".join(f"{s.slotKey}={s.slotValue}" for s in existing_slots) or "none"
    keys_list = "\n".join(f"- {k}" for k in required_keys)
    history_str = "\n".join(
        f"{'AI' if h['role'] == 'assistant' else 'User'}: {h['content']}" for h in history
    )
    system = (
        "You are an information extractor. Given the full conversation history and the user's latest reply, "
        "extract any values that match the required slot keys. Return ONLY valid JSON: "
        '{"slots": [{"slotKey": "...", "slotValue": "..."}]}\n'
        "Check the entire conversation — a slot may have been mentioned in an earlier turn, not just the latest reply. "
        "Do not repeat slots already listed in 'Already filled slots'. "
        "If nothing new is found, return {\"slots\": []}."
    )
    user = (
        f"Required slot keys:\n{keys_list}\n\n"
        f"Already filled slots: {existing_summary}\n\n"
        f"Conversation history:\n{history_str}\n\n"
        f"User's latest reply: {transcript}"
    )
    raw = chat(system, user, max_tokens=256)
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        return [FilledSlot(**s) for s in data.get("slots", [])]
    except (json.JSONDecodeError, Exception):
        return []


# 꼬리 질문 생성하는 함수
def _generate_followup(
    user_reply: str, # 현재 턴에서 유저가 말한 내용 (STT 변환된 텍스트)
    situation: str, # 시나리오 상황 설명. 예: "You are ordering coffee at a cafe"
    missing_keys: list[str], #  아직 채워지지 않은 슬롯 키 목록. 예: ["hot_or_iced", "size"]. LLM이 이 정보를 기준으로 뭘 물어볼지 결정
    history: list[dict], # 현재 턴 이전까지의 전체 대화 기록. [{"role": "assistant", "content": "..."}, {"role": "user", "content": "..."}] 형태
) -> str:
    history_str = "\n".join(
        f"{'AI' if h['role'] == 'assistant' else 'User'}: {h['content']}" for h in history
    )
    missing_str = ", ".join(missing_keys)
    system = (
        "You are a native English speaker having a real conversation in the given situation. "
        "Before generating a response, carefully review the full conversation history. "
        "Do NOT ask about anything the user has already mentioned in any previous turn. "
        "First, respond naturally to what the user just said. "
        "Then ask only about information that is genuinely absent from the entire conversation. "
        "Two sentences max. No lists."
    )
    user = (
        f"Situation: {situation}\n"
        f"Conversation so far:\n{history_str}\n"
        f"User just said: {user_reply}\n"
        f"Still need (verify against history before asking): {missing_str}"
    )
    return chat(system, user, max_tokens=128)


# 슬롯 모두 충족 시 (= 시나리오 클리어 시) 자연스러운 마무리 멘트 생성 (SUCCESS)
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


# 질문 횟수 초과 시 (= 시나리오 실패 시) 종료 발화 생성 (FAILURE)
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
