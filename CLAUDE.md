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
streamlit run src\review_app\app.py     # 구간분리 확인·수정, 결과 검토 — 로컬 검토 앱 (반드시 C:\mssb_venv의 streamlit으로 — 아래 참고)
python scripts\run_pipeline.py ... --stage cv        # 표정·자세·인형·음성 원본 수치만 저장 (Claude 호출 없음)
python scripts\calibrate_cv.py --session <id> --subject-type child   # 저장된 수치로 임계값 보정 리포트 (Claude·영상 불필요)
python scripts\run_pipeline.py ... --stage coding
python scripts\run_pipeline.py ... --stage synthesis

streamlit run src\public_app\app.py     # 공개 결과 뷰어(비밀번호 게이트) 로컬 실행 — .streamlit/secrets.toml 필요
```

## Architecture

**데이터 흐름(파이프라인, `src/mssb_coder/pipeline.py`)**: `run_segmentation_stage` → (사람 확인, Streamlit) → `run_coding_stage` → `run_synthesis_stage`. 세 단계 모두 `scripts/run_pipeline.py --stage`로 개별 실행 가능 — 2번째 단계 전에 사람이 `stem_timestamps.json`을 Streamlit에서 고칠 수 있게 일부러 끊어놓은 것.

1. **입력/구간분리** — `video_prep.py`(ffmpeg) → `transcribe.py`(faster-whisper, 전체 세션 통전사) → `segmentation.py`(Claude 호출, 전사문만 전송)가 스템별 `start_s/end_s/handoff_timestamp_s/additional_examiner_segments_s`를 반환 → `storage.save_stem_timestamps`. `handoff_timestamp_s`가 화자(검사자/아동) 태깅의 유일한 기준 — 별도 문자열 매칭 없음.
2. **CV 단계(`run_cv_stage`, Claude 호출 없음)** — 스템(+기준선용 워밍업)마다 `video_prep`으로 프레임/오디오 추출(이미 디스크에 있으면 재사용) → `facial_expression.py`(Py-Feat, AU는 0~1 **확률**), `pose_analysis.py`(MediaPipe Tasks API, 2명 검출 후 코 visibility 최고인 사람=아동) , `object_tracking.py`(YOLO 프레임별 탐지, 정규화 좌표 bbox, 탐지 실패 프레임도 `bbox=None`으로 기록, 가중치 없으면 빈 결과), `voice_prosody.py`(openSMILE eGeMAPS) → 임계값 적용 전 **원본 수치**를 `cv/<stem>_cv.json`(`schema.StemCvTimeline`, `cv_timeline.py`로 변환)에 저장. `calibration.py`/`scripts/calibrate_cv.py`가 이 파일만 읽어 데이터 품질·분포·육안 확인 후보를 리포트한다.
3. **코딩 단계(`run_coding_stage`)** — 저장된 CV 타임라인을 재사용(없으면 계산)하고 `cv_events.derive_events`로 이벤트를 만든다. 판정 기준은 `config/cv_thresholds_*.yaml`(근거 등급 [A]/[B]/[C]는 yaml 주석 참고)이며 **자세·음성은 그 아이 워밍업 구간 대비 상대 z-score**(`cv_events.compute_baselines`), 표정은 AU 확률 ≥0.5가 연속 프레임 유지 + 아동 평가에서 신뢰할 만한 AU만 사용. yaml 핵심 값이 null이거나 기준선·결측 조건이 안 맞으면 그 신호는 이벤트 없이 `quality_notes`로 사유만 남긴다(빈 이벤트=측정 불가일 수 있음) → `feature_summary.py`가 4개 모달리티를 텍스트로 합침 → `verbatim.py`가 발화+행동신호 통합 축어록 생성(`storage.save_stem_verbatim`) → `claude_client.call_stem_coding`(Claude 호출, 전사문+CV 텍스트 요약만 전송) → `schema.StemCodingResult` pydantic 검증 → `storage.save_stem_coding`.
4. **종합 단계** — `session_synthesis.call_session_synthesis`가 전체 스템의 **AI 원본**(`*_ai_codes.json`, 사람 검토 전) 결과를 입력으로 세션 전체 패턴을 종합 → `schema.SessionSynthesis` 검증 → `storage.save_session_synthesis`. 같은 호출에서 패턴 외에 `behavioral_observation`(검사 중 행동관찰)/`strengths`/`support_areas`/`recommendations`(제언, AI 초안·사람 확인 전제)도 함께 생성한다 — 입력은 스템별 rationale뿐이라 행동관찰의 비언어 단서는 rationale에 이미 적힌 것만 재료다. 이 4필드는 기본값이 있어 예전 JSON도 로드되지만, 채우려면 synthesis 단계 재실행(=API 호출)이 필요하다.

**핵심 설계 불변식**:
- CV 분석(영상/이미지)은 항상 로컬에서만 실행된다. Claude API로는 전사문·CV 수치 요약 등 **텍스트만** 전송된다(`claude_client.py`/`segmentation.py`/`session_synthesis.py` 전부 이 원칙).
- 세션 원본 데이터(`data/`, 영상, JSON)는 `MSSB_DATA_DIR` 환경변수가 가리키는 OneDrive 밖 경로에 저장된다(`storage.get_data_dir` — 없으면 에러, 조용히 기본값으로 안 넘어감). 이 리포 자체는 OneDrive 동기화 경로 안에 있어서 원본 데이터를 여기 두면 안 됨.
- `storage.REPO_RESULTS_DIR`(`<repo>/results/`)는 `MSSB_DATA_DIR`과 별개 — 공개 사이트(`src/public_app/app.py`)가 같은 git 저장소에서 읽는 "게시용" 결과만 들어간다. 영상은 절대 여기 들어가지 않음.
- `interpretation_layer`는 스키마 수준에서 `1_single_story_observation`/`2_cross_story_pattern`으로 고정 — 진단·위험분류(층위3/4)에 해당하는 필드는 스키마에 아예 두지 않는다(`schema.py`). 단 `recommendations`(제언)는 사용자 결정으로 AI가 쓰되, 프롬프트에서 진단명·장애명·위험수준·약물/치료 권고를 금지하고 검토 앱에서 사람이 확인하는 것을 안전장치로 삼는다.
- `subject_type`(adult/child)은 CV 도구·임계값(`config/cv_thresholds_{adult,child}.yaml`)만 전환하는 스위치이고, Claude 프롬프트 계층에는 적용되지 않는다(성인 대역 영상은 배선 검증용일 뿐, 채점 "내용"은 신뢰하지 않음).
- `labels.py`가 영문 id(스템/범주/항목/enum)를 한글 표시 라벨로 바꾸는 유일한 지점 — `report.py`/`review_app`/`public_app`이 전부 여기를 통해서만 한글화하고, `scrub_english_ids`가 Claude 자유 텍스트에 섞여나온 영문 id도 사후 치환한다.
- `report.build_session_report_markdown()`은 **학부모가 받는 실제 문서**라, 범주/항목 체크리스트나 타임스탬프 근거 인용을 의도적으로 배제하고 서술형 섹션으로만 구성한다. 순서: 인적 사항 → 검사 개요 → 의뢰 사유 및 배경 → 검사 중 행동관찰 → 이야기별 해석(스템당 한 단락, 5범주 rationale을 이어붙임) → 종합 해석(패턴 서술만) → 강점과 지원이 필요한 영역 → 제언 → 해석상 유의점. 의뢰 사유·배경, 검사일, 검사자는 다른 검사가 포함되는 전체 보고서에서 가져올 예정이라 `report.PLACEHOLDER` 문구로 비워 둔다. 점수는 쓰지 않는다. 신규 섹션은 값이 비어 있으면(예전 세션) 생략된다. 같은 체크리스트·근거 상세는 `review_app`의 스템 상세 드릴다운(`render_stem_detail`)에만 남아 있다 — 두 화면이 서로 다른 독자(학부모 vs 검사자)를 겨냥한 서로 다른 뷰라는 게 설계 의도다.
- `bundle.py`(순수 직렬화, 디스크 비의존) ↔ `storage.build_session_export_bundle`/`publish_session_bundle_to_repo`(디스크 I/O) — 로컬 검토 앱과 공개 앱이 같은 번들 포맷을 공유.

**알려진 한계 (코드·산출물을 직접 확인한 것)**:
- 학부모 보고서의 "이야기별 해석"은 5범주 rationale을 그대로 이어붙인 것이라 "부재로 코딩했다", "존재로 채점됨", "AU04" 같은 채점자용 표현이 섞여 나온다. 학부모용 문체로 쓰려면 규칙 기반 걸러내기나 Claude 재작성이 필요하다.
- `report.py`의 종합 해석 소제목 템플릿("…영역은 전체 N개 이야기 중 M개에서 일관되게 관찰되었다")은 M이 작은 범주(회피/위축의 '반복' 하나, 공격성 3/10)도 "일관되게"로 읽히게 한다. 해석상 유의점의 "검사자가 검토해 확정한 것"도 실제 검토 여부와 무관하게 고정 문구다.
- Py-Feat 머리 방향(pitch/yaw) 단위는 도(degree)로 가정했고 실제 영상으로 검증하지 못했다(`facial_expression.py`). `--stage cv`는 프레임·오디오가 없고 `--video`도 없는 스템은 경고만 남기고 건너뛴다(예: 채점 제외 스템이라 프레임을 뽑지 않았던 옛 세션의 warm_up).

**알려진 비효율 (호스팅·비용 논의 때 근거로 쓰임, `결정사항.md` 참고)**:
- `run_coding_stage`가 `transcribe_session`을 다시 호출한다(`run_segmentation_stage`에서 이미 한 번 함). 전사 결과를 저장하지 않아서 whisper large-v3가 세션당 두 번 돈다.
- Claude 호출 3곳(`segmentation`/`claude_client`/`session_synthesis`)은 `response.usage`를 저장하지 않는다. 그래서 세션당 API 비용은 Anthropic 콘솔에서만 확인할 수 있다.
- 모델은 `claude_client.MODEL_VERSION = "claude-opus-5"`이고 `tool_choice={"type": "tool", ...}`(강제 도구 호출)을 쓴다. Opus 5.5 이상은 강제 tool_choice를 400으로 거부하므로, 모델을 올리려면 호출 방식도 바꿔야 한다.

**앱 2개, 역할이 분리됨**:
- `src/review_app/app.py` — 로컬에서만 실행. 세션 종합 초안이 1차 화면, 근거 클릭 시 스템 상세로 드릴다운. 검토·수정 저장(`*_reviewed.json`) + 보고서 PDF 다운로드. 「2. 결과 검토」(`synthesis_page.py`)는 스크롤을 줄이려고 3단(행동관찰·단발성 | 반복 패턴 | 강점·지원·제언) 레이아웃이고, 촘촘한 CSS는 `app.py`의 `COMPACT_CSS`, 입력칸 높이는 `fit_height`가 글자 수로 어림잡는다. 화면의 .md·.json 내보내기와 "사업단 공유 사이트에 게시" 버튼은 제거됨(`storage.publish_session_bundle_to_repo`/`git_publish`는 남아 있으나 앱에서 호출하지 않음 — 게시는 현재 수동). 8501 포트에 시스템 파이썬으로 띄운 옛 streamlit 프로세스가 남아 있으면 코드 수정이 반영되지 않으니, 안 바뀌면 `Get-NetTCPConnection -LocalPort 8501`로 프로세스를 확인해 종료 후 `C:\mssb_venv\Scripts\streamlit.exe`로 다시 띄운다.
- `src/public_app/app.py` — Streamlit Community Cloud에 배포되는 공개 뷰어(`APP_PASSWORD` 비밀번호 게이트). `results/*.json`만 읽으며 영상·`MSSB_DATA_DIR`에는 전혀 접근하지 않음. 자체 `requirements.txt`(가벼움, CV 라이브러리 없음)를 둬서 Streamlit Cloud가 무거운 루트 `requirements.txt` 대신 이걸 설치하게 함.

**테스트**: `pytest`, `tests/`에 `src/mssb_coder/` 각 모듈당 거의 1:1로 대응하는 테스트 파일. 전부 외부 설치(ffmpeg/CV 라이브러리) 없이 도는 순수 함수 테스트다(`schema`/`storage`/`coding_systems`/`posture_metrics`/`feature_summary`/`labels`/`bundle`/`verbatim`/`cv_events`/`calibration`, 그리고 `facial_expression`/`voice_prosody`/`object_tracking`의 임계값·이벤트 로직). 실제 모델 추론 함수(`run_facial_expression`/`run_voice_prosody`/`run_pose_analysis`/`run_object_tracking`의 추론부, `transcribe_session`)는 해당 패키지가 필요해 단위 테스트가 없다.
