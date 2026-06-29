# LAN-28 Turn Feedback Del Markup Plan

## 목표

`NEEDS_IMPROVEMENT`의 `correctionExpression` 하나에만 렌더링용 취소선 마크업을 담는다. 새 필드는 추가하지 않는다.

## 계약

- `GOOD`은 기존처럼 `correctionExpression=null`을 유지한다.
- `NEEDS_IMPROVEMENT`의 `correctionExpression`은 개선 표현을 담되, 사용자의 원문 중 삭제되거나 교체되는 틀린 부분만 `<del>`, `</del>`로 감싼다.
- 추가만 필요한 교정은 취소선으로 감쌀 원문이 없을 수 있으므로 `<del>` 없이 개선 표현만 허용한다.
- `correctionReason`은 한국어 설명만 담고, 개선 표현 전체를 반복하지 않는다.
- 허용 HTML 태그는 `<del>`과 `</del>`뿐이다. 다른 태그는 제거한다.

## 작업 순서

1. 회귀 테스트를 먼저 추가하고 RED를 확인한다.
2. `turn-feedback` 프롬프트와 예시를 새 계약으로 수정한다.
3. `correctionExpression` 후처리에 `<del>` sanitizing과 plain 개선 표현 보정 로직을 추가한다.
4. README 계약 설명을 새 렌더링 방식으로 갱신한다.
5. focused 테스트, 전체 unittest, compileall, diff check를 실행한다.
6. 배포는 하지 않고 로컬 커밋까지만 정리한다.
