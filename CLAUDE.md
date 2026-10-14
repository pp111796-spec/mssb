# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MSSB(MacArthur Story Stem Battery) 검사 영상을 분석해 사람 채점자를 위한 AI 채점 초안(이야기별 + 세션 종합 패턴)을 만드는 human-in-the-loop PoC 도구. 상세 설계·근거는 `플랜.md`(계획, gitignored), 알려진 한계는 `문제점.md`(gitignored), 그 뒤의 실제 의사결정 기록은 `결정사항.md`(gitignore 대상은 아니지만 아직 커밋 전) 참고. 사용자/설치 안내는 `README.md`가 최신 기준이며, 이 CLAUDE.md는 코드를 다룰 때 필요한 사실만 간결하게 둔다.

## Commands

```powershell
# 가상환경은 반드시 OneDrive/한글 경로 밖에 (openSMILE 네이티브 라이브러리가 ASCII 경로만 처리 — 문제점.md 참고)
python -m venv C:\mssb_venv
C:\mssb_venv\Scripts\Activate.ps1
cd "C:\Users\user\OneDrive - 건양대학교\바탕 화면\AI_digital"
pip install -e .
pip install -r requirements.txt

pytest                      # 전체 테스트
pytest tests/test_report.py # 단일 파일
pytest tests/test_report.py::test_report_handles_no_recurring_patterns  # 단일 테스트

# 파이프라인 실행 (4단계=사람 확인이 segmentation과 coding 사이에 끼어들므로 단계별 실행)
python scripts\run_pipeline.py --session <id> --video <mp4> --child-age-months <n> --child-gender <male|female> --subject-type <adult|child> --stage segmentation
streamlit run src\review_app\app.py     # 구간분리 확인·수정, 이후 게시까지 — 로컬 검토 앱
python scripts\run_pipeline.py ... --stage coding
python scripts\run_pipeline.py ... --stage synthesis

streamlit run src\public_app\app.py     # 공개 결과 뷰어(비밀번호 게이트) 로컬 실행 — .streamlit/secrets.toml 필요
```

## Architecture

**데이터 흐름(파이프라인, `src/mssb_coder/pipeline.py`)**: `run_segmentation_stage` → (사람 확인, Streamlit) → `run_coding_stage` → `run_synthesis_stage`. 세 단계 모두 `scripts/run_pipeline.py --stage`로 개별 실행 가능 — 2번째 단계 전에 사람이 `stem_timestamps.json`을 Streamlit에서 고칠 수 있게 일부러 끊어놓은 것.

1. **입력/구간분리** — `video_prep.py`(ffmpeg) → `transcribe.py`(faster-whisper, 전체 세션 통전사) → `segmentation.py`(Claude 호출, 전사문만 전송)가 스템별 `start_s/end_s/handoff_timestamp_s/additional_examiner_segments_s`를 반환 → `storage.save_stem_timestamps`. `handoff_timestamp_s`가 화자(검사자/아동) 태깅의 유일한 기준 — 별도 문자열 매칭 없음.
2. **코딩 단계(`run_coding_stage`)** — 확정된 구간마다: `video_prep`으로 프레임/오디오 추출 → `facial_expression.py`(Py-Feat), `pose_analysis.py`(MediaPipe Tasks API) + `posture_metrics.py`, `object_tracking.py`(YOLO+ByteTrack, 가중치 없으면 빈 결과), `voice_prosody.py`(openSMILE eGeMAPS, warm_up 구간 기준선 대비 z-score) → `feature_summary.py`가 4개 모달리티를 텍스트로 합침 → `verbatim.py`가 발화+행동신호 통합 축어록 생성(`storage.save_stem_verbatim`) → `claude_client.call_stem_coding`(Claude 호출, 전사문+CV 텍스트 요약만 전송) → `schema.StemCodingResult` pydantic 검증 → `storage.save_stem_coding`.
3. **종합 단계** — `session_synthesis.call_session_synthesis`가 전체 스템의 **AI 원본**(`*_ai_codes.json`, 사람 검토 전) 결과를 입력으로 세션 전체 패턴을 종합 → `schema.SessionSynthesis` 검증 → `storage.save_session_synthesis`.

**핵심 설계 불변식**:
- CV 분석(영상/이미지)은 항상 로컬에서만 실행된다. Claude API로는 전사문·CV 수치 요약 등 **텍스트만** 전송된다(`claude_client.py`/`segmentation.py`/`session_synthesis.py` 전부 이 원칙).
- 세션 원본 데이터(`data/`, 영상, JSON)는 `MSSB_DATA_DIR` 환경변수가 가리키는 OneDrive 밖 경로에 저장된다(`storage.get_data_dir` — 없으면 에러, 조용히 기본값으로 안 넘어감). 이 리포 자체는 OneDrive 동기화 경로 안에 있어서 원본 데이터를 여기 두면 안 됨.
- `storage.REPO_RESULTS_DIR`(`<repo>/results/`)는 `MSSB_DATA_DIR`과 별개 — 공개 사이트(`src/public_app/app.py`)가 같은 git 저장소에서 읽는 "게시용" 결과만 들어간다. 영상은 절대 여기 들어가지 않음.
- `interpretation_layer`는 스키마 수준에서 `1_single_story_observation`/`2_cross_story_pattern`으로 고정 — 진단·위험분류(층위3/4)에 해당하는 필드는 스키마에 아예 두지 않는다(`schema.py`).
- `subject_type`(adult/child)은 CV 도구·임계값(`config/cv_thresholds_{adult,child}.yaml`)만 전환하는 스위치이고, Claude 프롬프트 계층에는 적용되지 않는다(성인 대역 영상은 배선 검증용일 뿐, 채점 "내용"은 신뢰하지 않음).
- `labels.py`가 영문 id(스템/범주/항목/enum)를 한글 표시 라벨로 바꾸는 유일한 지점 — `report.py`/`review_app`/`public_app`이 전부 여기를 통해서만 한글화하고, `scrub_english_ids`가 Claude 자유 텍스트에 섞여나온 영문 id도 사후 치환한다.
- `report.build_session_report_markdown()`은 **학부모가 받는 실제 문서**라, 범주/항목 체크리스트나 타임스탬프 근거 인용을 의도적으로 배제하고 "이야기별 해석"(스템당 한 단락, 5범주 rationale을 이어붙임) + "종합 해석"(패턴 서술만) 두 섹션으로만 구성한다. 같은 체크리스트·근거 상세는 `review_app`의 스템 상세 드릴다운(`render_stem_detail`)에만 남아 있다 — 두 화면이 서로 다른 독자(학부모 vs 검사자)를 겨냥한 서로 다른 뷰라는 게 설계 의도다.
- `bundle.py`(순수 직렬화, 디스크 비의존) ↔ `storage.build_session_export_bundle`/`publish_session_bundle_to_repo`(디스크 I/O) — 로컬 검토 앱과 공개 앱이 같은 번들 포맷을 공유.

**알려진 비효율 (호스팅·비용 논의 때 근거로 쓰임, `결정사항.md` 참고)**:
- `run_coding_stage`가 `transcribe_session`을 다시 호출한다(`run_segmentation_stage`에서 이미 한 번 함). 전사 결과를 저장하지 않아서 whisper large-v3가 세션당 두 번 돈다.
- Claude 호출 3곳(`segmentation`/`claude_client`/`session_synthesis`)은 `response.usage`를 저장하지 않는다. 그래서 세션당 API 비용은 Anthropic 콘솔에서만 확인할 수 있다.
- 모델은 `claude_client.MODEL_VERSION = "claude-opus-5"`이고 `tool_choice={"type": "tool", ...}`(강제 도구 호출)을 쓴다. Opus 5.5 이상은 강제 tool_choice를 400으로 거부하므로, 모델을 올리려면 호출 방식도 바꿔야 한다.

**앱 2개, 역할이 분리됨**:
- `src/review_app/app.py` — 로컬에서만 실행. 세션 종합 초안이 1차 화면, 근거 클릭 시 스템 상세로 드릴다운. 검토·수정 저장(`*_reviewed.json`) + "🚀 사업단 공유 사이트에 게시" 버튼(`results/`에 git add/commit/push).
- `src/public_app/app.py` — Streamlit Community Cloud에 배포되는 공개 뷰어(`APP_PASSWORD` 비밀번호 게이트). `results/*.json`만 읽으며 영상·`MSSB_DATA_DIR`에는 전혀 접근하지 않음. 자체 `requirements.txt`(가벼움, CV 라이브러리 없음)를 둬서 Streamlit Cloud가 무거운 루트 `requirements.txt` 대신 이걸 설치하게 함.

**테스트**: `pytest`, `tests/`에 `src/mssb_coder/` 각 모듈당 거의 1:1로 대응하는 테스트 파일. 외부 설치(ffmpeg/CV 라이브러리) 없이 도는 것(`schema`/`storage`/`coding_systems`/`posture_metrics`/`feature_summary`/`labels`/`bundle`/`verbatim`)과, 실제 실행엔 해당 CV 패키지가 필요한 것(`facial_expression`/`voice_prosody`/`object_tracking`/`transcribe`)이 섞여 있다.
