# 대화 라우터 — POST /conversation/start (세션 시작), POST /conversation/next (다음 턴 진행)
from fastapi import APIRouter # == @RestController

# Autowired
from app.models.conversation import StartRequest, StartResponse, NextRequest, NextResponse
from app.services import conversation_service
from app.services.tts_service import synthesize

router = APIRouter(prefix="/conversation", tags=["conversation"])


@router.post("/start", response_model=StartResponse, summary="대화 세션 시작", description="시나리오 ID를 받아 새로운 대화 세션을 시작하고 첫 질문과 음성 데이터를 반환합니다.")
def start(body: StartRequest): # == @RequestBody
    """
    시나리오를 선택하여 대화 세션을 시작합니다.

    - scenario_id: 선택한 시나리오 ID (예: cafe_1, airport_2)
    - 반환: session_id, 첫 번째 AI 질문, 질문의 음성(MP3 base64)
    """
    try:
        result = conversation_service.start_session(body.scenario_id)
        result["audio_base64"] = synthesize(result["question"])
        return StartResponse(success=True, data=result) # == ResponseEntity.ok(StartResponse)

    except ValueError as e:   # == try {} catch (ValueError e)
        return StartResponse(success=False, error=str(e))


@router.post("/next", response_model=NextResponse, summary="꼬리질문 진행", description="세션 ID, 유저 발화 텍스트, 응답 시간 정보를 받아 다음 질문과 음성 데이터를 반환합니다. 시나리오 클리어 여부도 함께 반환됩니다.")
def next_turn(body: NextRequest):
    """
    사용자의 음성 답변(STT 결과)을 받아서 이해도를 분석하고 다음 질문을 생성합니다.

    - session_id: 현재 세션 ID
    - stt_text: 사용자의 답변 (STT로 변환된 영어 텍스트)
    - response_time_sec: 마이크 버튼부터 답변까지 걸린 시간(초)
    - 반환: 이해도 점수, 다음 질문(또는 세션 종료)
    """
    try:
        result = conversation_service.next_turn(
            body.session_id, body.stt_text, body.response_time_sec
        )
        if not result["done"] and "next_question" in result:
            result["audio_base64"] = synthesize(result["next_question"])
        elif result["done"] and result.get("closing_message"):
            result["audio_base64"] = synthesize(result["closing_message"])
        return NextResponse(success=True, data=result)
    except ValueError as e:
        return NextResponse(success=False, error=str(e))
