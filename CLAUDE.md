# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MSSB(MacArthur Story Stem Battery) 검사 영상을 분석해 사람 채점자를 위한 AI 채점 초안(이야기별 + 세션 종합 패턴)을 만드는 human-in-the-loop PoC 도구. 상세 설계·근거는 `플랜.md`(계획, gitignored), 알려진 한계는 `문제점.md`(gitignored), 그 뒤의 실제 의사결정 기록은 `결정사항.md`(커밋됨) 참고. 보고서 문장·프롬프트 문체를 다룰 때는 `.claude/skills/report-tone/SKILL.md`(보고서 말투 원칙)를 따른다. 사용자/설치 안내는 `README.md`가 최신 기준이며, 이 CLAUDE.md는 코드를 다룰 때 필요한 사실만 간결하게 둔다.

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

# 파이프라인 실행 — 단계마다 따로 (한 번에 몰아 돌리는 옵션은 일부러 없음)
python scripts\run_pipeline.py --session <id> --video <mp4> --child-age-months <n> --child-gender <male|female> --subject-type <adult|child> --stage segmentation
C:\mssb_venv\Scripts\streamlit.exe run src\review_app\app.py   # 「1. 구간 확인」에서 구간 확정 (확정해야 cv·coding 실행 가능)
python scripts\run_pipeline.py --session <id> --stage cv        # (선택) 표정·자세·인형·음성 원본 수치만 저장 (Claude 호출 없음)
python scripts\calibrate_cv.py --session <id> --subject-type child   # 저장된 수치로 임계값 보정 리포트 (Claude·영상 불필요)
python scripts\run_pipeline.py --session <id> --stage coding    # 같은 구간으로 이미 채점한 스템은 건너뜀, --force로 전체 재채점
python scripts\run_pipeline.py --session <id> --stage synthesis
# segmentation 때 저장한 session.json(연령·성별·영상 경로)을 이후 단계가 재사용하므로 --session만 주면 된다

streamlit run src\public_app\app.py     # 공개 결과 뷰어(비밀번호 게이트) 로컬 실행 — .streamlit/secrets.toml 필요
```

## Architecture

**데이터 흐름(파이프라인, `src/mssb_coder/pipeline.py`)**: `run_segmentation_stage` → (사람 확인: 검토 앱 「1. 구간 확인」에서 확정) → (`run_cv_stage`) → `run_coding_stage` → `run_synthesis_stage`. 모두 `scripts/run_pipeline.py --stage`로 따로 실행한다.

**단계 사이의 안전장치**:
- **구간 확정**: 「1. 구간 확인」(`review_app/segmentation_page.py`)의 '구간 확정'이 `segmentation_review.json`(구간 지문 sha256)을 만든다. cv·coding은 미확정이거나 확정 뒤 `stem_timestamps.json`이 바뀌었으면 `SegmentationNotConfirmedError`로 거부한다(`storage.segmentation_review_status`).
- **구간 변경 시 재계산**: 프레임·오디오(`frames/<stem>/source_window.json`), CV 수치(`StemCvTimeline.source_window`), 채점 결과(`StemCodingResult.source_window`)가 만든 당시 구간(`schema.StemWindow`)을 기록한다. 구간이 바뀐 스템만 다시 추출·계산·채점하고, 기록이 없는 예전 파일은 경고와 함께 재사용한다.
- **실패 격리·이어하기**: `run_coding_stage`는 스템마다 따로 처리해 하나가 실패해도 나머지를 저장하고 끝에 `StageFailedError`(실패 목록)를 던진다. 재실행하면 같은 구간으로 이미 채점한 스템은 건너뛴다(`force=True`면 전부). 종합은 현재 채점 대상 스템이 모두 현재 구간으로 채점됐을 때만 돈다(`load_codings_for_synthesis`).
- **시간 기준**: 저장되는 CV 수치·이벤트 계산은 상대 시각(프레임=스템 시작, 음성=전환 시각 기준)이고, `cv_events.to_session_time`이 한 곳에서 세션 절대 시각으로 바꾼다. Claude에 보내는 전사문·비언어 요약·축어록·근거 시각은 모두 세션 절대 시각(`timestamp_basis="session_absolute"`).

1. **입력/구간분리** — `video_prep.py`(ffmpeg) → `transcribe.py`(faster-whisper, 전체 세션 통전사, `transcript.json`으로 저장해 이후 재사용) → `segmentation.py`(Claude 호출, 전사문만 전송)가 스템별 `start_s/end_s/handoff_timestamp_s/additional_examiner_segments_s`를 반환 → `pipeline.normalize_segments`(이야기 목록에 없는 이름 거부, 녹화 앞뒤 여백 `pre_session`/`post_session`은 채점 제외로 허용, `scored`는 yaml 기준) → `stem_timestamps_ai_draft.json`(원안)과 `stem_timestamps.json` 저장, 아동 정보·영상 경로는 `session.json`. 다시 돌리면 기존 구간은 `.bak.json`으로 백업. `handoff_timestamp_s`가 화자(검사자/아동) 태깅의 유일한 기준 — 별도 문자열 매칭 없음.
2. **CV 단계(`run_cv_stage`, Claude 호출 없음)** — 스템(+기준선용 워밍업)마다 `video_prep`으로 프레임/오디오 추출(이미 디스크에 있으면 재사용) → `facial_expression.py`(Py-Feat, AU는 0~1 **확률**), `pose_analysis.py`(MediaPipe Tasks API, 2명 검출 후 코 visibility 최고인 사람=아동) , `object_tracking.py`(YOLO 프레임별 탐지, 정규화 좌표 bbox, 탐지 실패 프레임도 `bbox=None`으로 기록, 가중치 없으면 빈 결과), `voice_prosody.py`(openSMILE eGeMAPS) → 임계값 적용 전 **원본 수치**를 `cv/<stem>_cv.json`(`schema.StemCvTimeline`, `cv_timeline.py`로 변환)에 저장. `calibration.py`/`scripts/calibrate_cv.py`가 이 파일만 읽어 데이터 품질·분포·육안 확인 후보를 리포트한다.
3. **코딩 단계(`run_coding_stage`)** — 저장된 CV 타임라인을 재사용(없으면 계산)하고 `cv_events.derive_events`로 이벤트를 만든다. 판정 기준은 `config/cv_thresholds_*.yaml`(근거 등급 [A]/[B]/[C]는 yaml 주석 참고)이며 **자세·음성은 그 아이 워밍업 구간 대비 상대 z-score**(`cv_events.compute_baselines`), 표정은 AU 확률 ≥0.5가 연속 프레임 유지 + 아동 평가에서 신뢰할 만한 AU만 사용. yaml 핵심 값이 null이거나 기준선·결측 조건이 안 맞으면 그 신호는 이벤트 없이 `quality_notes`로 사유만 남긴다(빈 이벤트=측정 불가일 수 있음) → `feature_summary.py`가 4개 모달리티를 텍스트로 합침 → `verbatim.py`가 발화+행동신호 통합 축어록 생성(`storage.save_stem_verbatim`) → `claude_client.call_stem_coding`(Claude 호출, 전사문+CV 텍스트 요약만 전송) → `schema.StemCodingResult` pydantic 검증 → `storage.save_stem_coding` → 축어록 저장(채점 성공 뒤).

**Claude 호출(`claude_client.call_structured`, 세 호출 공통)**: `claude-opus-5-5`, structured outputs(`output_config.format` json_schema, `coding_systems.to_structured_output_schema`가 `additionalProperties:false`를 붙이고 미지원 제약 키워드를 뺌 — `description`은 유지), `effort="high"`(Opus 5.5 기본값은 medium), thinking은 생략(항상 켜짐), `max_tokens=64000` 스트리밍, 시스템 프롬프트 `cache_control`, 안전 분류기 오탐 대비 server-side fallback(`fallbacks="default"`). 실제 응답 모델을 `model_version`에, 토큰 사용량을 세션 폴더 `api_usage.jsonl`에, 스키마 검증에 실패한 원본을 `failed/`에 남긴다. 강제 `tool_choice`는 Opus 5.5에서 400이라 쓰지 않는다. 실제 API로는 아직 한 번도 호출해 보지 않았다(2026-10 기준 — 첫 실제 세션에서 확인 필요).
4. **종합 단계** — `session_synthesis.call_session_synthesis`가 전체 스템의 **AI 원본**(`*_ai_codes.json`, 사람 검토 전) 결과를 입력으로 세션 전체 패턴을 종합 → `schema.SessionSynthesis` 검증 → `storage.save_session_synthesis`. 같은 호출에서 패턴 외에 `behavioral_observation`(검사 중 행동관찰)/`strengths`/`support_areas`/`recommendations`(제언, AI 초안·사람 확인 전제)도 함께 생성한다 — 입력은 스템별 rationale뿐이라 행동관찰의 비언어 단서는 rationale에 이미 적힌 것만 재료다. 이 4필드는 기본값이 있어 예전 JSON도 로드되지만, 채우려면 synthesis 단계 재실행(=API 호출)이 필요하다.

**핵심 설계 불변식**:
- CV 분석(영상/이미지)은 항상 로컬에서만 실행된다. Claude API로는 전사문·CV 수치 요약 등 **텍스트만** 전송된다(`claude_client.py`/`segmentation.py`/`session_synthesis.py` 전부 이 원칙).
- 세션 원본 데이터(`data/`, 영상, JSON)는 `MSSB_DATA_DIR` 환경변수가 가리키는 OneDrive 밖 경로에 저장된다(`storage.get_data_dir` — 없으면 에러, 조용히 기본값으로 안 넘어감). 이 리포 자체는 OneDrive 동기화 경로 안에 있어서 원본 데이터를 여기 두면 안 됨.
- `storage.REPO_RESULTS_DIR`(`<repo>/results/`)는 `MSSB_DATA_DIR`과 별개 — 공개 사이트(`src/public_app/app.py`)가 같은 git 저장소에서 읽는 "게시용" 결과만 들어간다. 영상은 절대 여기 들어가지 않음.
- `interpretation_layer`는 스키마 수준에서 `1_single_story_observation`/`2_cross_story_pattern`으로 고정 — 진단·위험분류(층위3/4)에 해당하는 필드는 스키마에 아예 두지 않는다(`schema.py`). 단 `recommendations`(제언)는 사용자 결정으로 AI가 쓰되, 프롬프트에서 진단명·장애명·위험수준·약물/치료 권고를 금지하고 검토 앱에서 사람이 확인하는 것을 안전장치로 삼는다.
- `subject_type`(adult/child)은 CV 도구·임계값(`config/cv_thresholds_{adult,child}.yaml`)만 전환하는 스위치이고, Claude 프롬프트 계층에는 적용되지 않는다(성인 대역 영상은 배선 검증용일 뿐, 채점 "내용"은 신뢰하지 않음).
- `labels.py`가 영문 id(스템/범주/항목/enum)를 한글 표시 라벨로 바꾸는 유일한 지점 — `report.py`/`review_app`/`public_app`이 전부 여기를 통해서만 한글화하고, `scrub_english_ids`가 Claude 자유 텍스트에 섞여나온 영문 id도 사후 치환한다.
- `report.build_session_report_markdown()`은 **학부모가 받는 실제 문서**라, 범주/항목 체크리스트나 타임스탬프 근거 인용을 의도적으로 배제하고 서술형 섹션으로만 구성한다. 순서: 인적 사항 → 검사 개요 → 의뢰 사유 및 배경 → 검사 중 행동관찰 → 이야기별 해석(스템당 한 단락, 5범주 rationale을 이어붙임) → 종합 해석(패턴 서술만) → 강점과 지원이 필요한 영역 → 제언 → 해석상 유의점. 의뢰 사유·배경, 검사일, 검사자는 다른 검사가 포함되는 전체 보고서에서 가져올 예정이라 `report.PLACEHOLDER` 문구로 비워 둔다. 점수는 쓰지 않는다. 신규 섹션은 값이 비어 있으면(예전 세션) 생략된다. 해석상 유의점은 `SessionSynthesis.reviewed_at`(검토본 저장 시 기록)이 있을 때만 "검사자가 검토해 확정했다", 없으면 "아직 검토하지 않았다"로 적는다. PDF는 `report_pdf.py`가 같은 마크다운을 fpdf2+나눔고딕(`assets/fonts`)으로 옮긴다. 같은 체크리스트·근거 상세는 `review_app`의 스템 상세(`stem_page.render_stem_detail`)에만 남아 있다 — 두 화면이 서로 다른 독자(학부모 vs 검사자)를 겨냥한 서로 다른 뷰라는 게 설계 의도다.
- 보고서·PDF·내보내기 번들은 `storage.load_effective_stem_codings`(현재 구간의 채점 대상 스템만, 스템마다 `*_reviewed.json` 우선)를 쓰고, 세션 종합의 입력은 AI 원본(`*_ai_codes.json`)이다. 그래서 스템 검토 수정은 종합 서술에 자동 반영되지 않는다(화면에 안내).
- `bundle.py`(순수 직렬화, 디스크 비의존, 객체가 아닌 JSON·더 새 `bundle_format_version` 거부) ↔ `storage.build_session_export_bundle`/`publish_session_bundle_to_repo`(디스크 I/O) — 로컬 검토 앱과 공개 앱이 같은 번들 포맷을 공유.

**알려진 한계 (코드·산출물을 직접 확인한 것)**:
- 학부모 보고서의 "이야기별 해석"은 5범주 rationale을 그대로 이어붙인 것이라 문체는 rationale을 쓰게 하는 프롬프트에 달려 있다. 스템 채점·종합 프롬프트는 `claude_client.REPORT_TONE_GUIDE`(하다체·능동형·코드/변인명 금지·항목 개수 표현 금지, `.claude/skills/report-tone/SKILL.md`와 같은 원칙)를 공유하지만, 그 전에 생성된 세션의 저장 문장은 메모체("~됨", "부재로 코딩했다")가 남아 있다. `labels.scrub_english_ids`는 AU 코드 묶음·"긴장 계열"·"자기수정"만 규칙으로 치환할 뿐 어미나 항목 개수 표현은 못 고친다 — 고치려면 문장 재작성(Claude API 또는 사람)이 필요하다(`adult_demo_01`은 2026-10-15에 재작성 완료, 원본은 세션 폴더 `pre_tone_bak/`).
- Py-Feat 머리 방향(pitch/yaw) 단위는 도(degree)로 가정했고 실제 영상으로 검증하지 못했다(`facial_expression.py`). 표정 시각은 Py-Feat 결과의 입력 경로(frame_NNNNNN.png)로 프레임 번호를 계산하고 한 프레임에 얼굴이 여럿이면 가장 큰 얼굴을 쓴다(`pick_one_face_per_frame`). `--stage cv`는 프레임·오디오가 없고 영상 경로도 없는 스템은 경고만 남기고 건너뛴다.
- `adult_demo_01`(성인 대역 시연, 유일한 세션)은 이번 안정화 전에 만든 세션이라 `session.json`·`transcript.json`·구간 확정 기록이 없고, 채점 결과에 `source_window`/`timestamp_basis`가 없다. 그래서 cv·coding을 다시 돌리려면 먼저 「1. 구간 확인」에서 확정해야 하고, 근거별 영상 재생 버튼은 재채점 전까지 꺼져 있다.

**알려진 비효율**: CV 모델(Py-Feat·MediaPipe·YOLO)을 스템마다 다시 불러온다. 비용 기록은 `api_usage.jsonl`로 세션별 확인 가능하다(호스팅·비용 논의 근거는 `결정사항.md`).

**앱 2개, 역할이 분리됨**:
- `src/review_app/` — 로컬에서만 실행. `app.py`는 세션 선택·진행 상태(`review_ops.session_progress`)·화면 전환만 맡고, 「1. 구간 확인」(`segmentation_page.py`: 구간 표 수정·검증·확정, 전사문 미리보기, 검사자 재개입 구간), 「2. 결과 검토」(`synthesis_page.py`: 종합 초안 검토·저장, 하단에서 스템 선택), 스템 상세(`stem_page.py`: 부록 B·항목·판단 이유·근거 제외 수정 → `*_reviewed.json`, AI 원본 되돌리기는 `.bak`로 이동), 구간 미리보기 영상(`video_panel.py`: 구간만 자른 360p 클립을 `preview/`에 캐시), 공통 도우미(`common.py`: 캐시된 설정, `plain`=영문 id 치환, 세션별 위젯 key, 파일 오류 안내)로 나뉜다. 판단 로직(구간 검증, 스템 수정 적용, 재생 가능 여부)은 테스트를 위해 `mssb_coder/review_ops.py`에 둔다. 위젯 key는 반드시 `widget_key(session_id, ...)`로 만든다(세션을 바꿨을 때 입력이 섞이지 않게). 「2. 결과 검토」(`synthesis_page.py`)는 스크롤을 줄이려고 3단(행동관찰·단발성 | 반복 패턴 | 강점·지원·제언) 레이아웃이고, 촘촘한 CSS는 `app.py`의 `COMPACT_CSS`, 입력칸 높이는 `fit_height`가 글자 수로 어림잡는다. 화면의 .md·.json 내보내기와 "사업단 공유 사이트에 게시" 버튼은 제거됨(`storage.publish_session_bundle_to_repo`/`git_publish`는 남아 있으나 앱에서 호출하지 않음 — 게시는 현재 수동). 8501 포트에 시스템 파이썬으로 띄운 옛 streamlit 프로세스가 남아 있으면 코드 수정이 반영되지 않으니, 안 바뀌면 `Get-NetTCPConnection -LocalPort 8501`로 프로세스를 확인해 종료 후 `C:\mssb_venv\Scripts\streamlit.exe`로 다시 띄운다.
- `src/public_app/app.py` — Streamlit Community Cloud에 배포되는 공개 뷰어(`APP_PASSWORD` 비밀번호 게이트). `results/*.json`만 읽으며 영상·`MSSB_DATA_DIR`에는 전혀 접근하지 않음. 자체 `requirements.txt`(가벼움, CV 라이브러리 없음)를 둬서 Streamlit Cloud가 무거운 루트 `requirements.txt` 대신 이걸 설치하게 함.

**테스트**: `pytest`, `tests/`에 `src/mssb_coder/` 각 모듈당 거의 1:1로 대응하는 테스트 파일. 전부 외부 설치(ffmpeg/CV 라이브러리/Claude API) 없이 돈다 — `requirements-test.txt`(pydantic·pyyaml·dotenv·streamlit·fpdf2·pytest)만으로 전체가 통과하며, GitHub Actions(`.github/workflows/test.yml`, windows-latest)가 main 푸시·PR마다 돌린다. `test_pipeline.py`는 CV·전사·Claude를 monkeypatch로 바꿔 단계 흐름(확정 확인·재계산·실패 격리·이어하기)을, `test_claude_client.py`는 가짜 클라이언트로 요청 형식·거절·잘림을, `test_apps.py`는 Streamlit `AppTest`로 두 앱을 화면 없이 띄워 본다. 공용 가짜 세션은 `tests/factories.py`. 실제 모델 추론 함수(`run_facial_expression`/`run_voice_prosody`/`run_pose_analysis`/`run_object_tracking`의 추론부, `transcribe_session`)는 해당 패키지가 필요해 단위 테스트가 없다. 검토 앱을 고쳤으면 사용자에게 넘기기 전에 `AppTest`로 실제로 띄워 예외가 없는지 확인한다.
