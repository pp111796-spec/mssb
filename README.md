# MSSB AI 코딩 보조 도구 — PoC

MSSB(MacArthur Story Stem Battery) 검사 영상을 분석해, 사람 채점자를 위한 AI 채점 초안(이야기별 + 세션 종합 패턴)을 만드는 human-in-the-loop 도구. 상세 설계는 `플랜.md`(계획), `문제점.md`(알려진 한계) 참고.

현재 상태: 영상 파일과 `ANTHROPIC_API_KEY`만 있으면 전체 파이프라인이 실제로 돌아간다 (표정 분석 제외 — 아래 "지금 상태" 참고). 인형/소품 추적은 마일스톤 0(라벨링·파인튜닝)이 없으면 자동으로 건너뛴다.

## ⚠️ 가상환경은 반드시 `C:\mssb_venv`처럼 OneDrive/한글 경로 밖에 만들 것

이 프로젝트 폴더(`...OneDrive - 건양대학교\바탕 화면\AI_digital`)는 경로에 한글이 섞여 있다. `pip install -e .`로 이 폴더 **안에** `.venv`를 만들면, openSMILE의 네이티브 라이브러리가 설정 파일 경로를 ASCII로만 처리하려다가 `UnicodeEncodeError`로 죽는다 — 실측으로 확인한 문제다 (`문제점.md` 참고). 그래서 venv는 항상 아래처럼 **한글이 없는 경로**에 만든다. 프로젝트 코드 자체는 그대로 이 폴더(OneDrive)에 둬도 된다 — 문제는 venv/설치된 패키지 경로에만 있다.

## 실제 영상 실행 전 반드시 확인할 것 (매뉴얼 §13 근거)

도구가 자동으로 강제하지 않는다 — 사람이 직접 확인해야 하는 항목:

- [ ] 보호자 서면 동의서에 **영상 녹화·보관·활용 범위 + "AI/컴퓨터비전 분석 및 그 요약의 제3자 클라우드(Anthropic, 미국) 전송·국외 이전"** 항목이 명시되어 있는지 (§13.1·§13.2)
- [ ] 소속 기관 IRB/연구윤리 승인 필요 여부, 필요시 승인 여부
- [ ] 영상·식별정보 분리 보관, 암호화 저장, 아동 실명을 파일명·경로에 쓰지 않는 원칙(§8.1·§13.2) 준수
- [ ] 아동학대 신고의무자 해당 여부 사전 확인 및 기관 보고 경로 확보(§13.3) — 이 도구는 위험 신호를 감지해도 진단·신고 판단을 자동으로 내리지 않으며, 사람이 확인해야 함
- [ ] Anthropic API 계정의 데이터 보존 정책이 이 용도에 맞는지

## 설정

```powershell
winget install Gyan.FFmpeg
winget install Python.Python.3.12

# venv는 반드시 OneDrive/한글 경로 밖에 (위 경고 참고)
python -m venv C:\mssb_venv
C:\mssb_venv\Scripts\Activate.ps1

cd "C:\Users\user\OneDrive - 건양대학교\바탕 화면\AI_digital"
pip install -e .
pip install -r requirements.txt
copy .env.example .env
# .env를 열어 ANTHROPIC_API_KEY, MSSB_DATA_DIR(OneDrive 밖 경로!)를 채울 것
```

OpenFace는 pip으로 안 깔리는 Windows 바이너리라 대신 **Py-Feat을 1차 구현으로 채택**했다 (`requirements.txt` 안내 주석 참고). 인형 라벨링 도구(Roboflow/CVAT 등)는 마일스톤 0에서 별도 설치.

## 실행

```powershell
python scripts\run_pipeline.py --session sample_01 --video <mp4 경로> --child-age-months 54 --child-gender female --subject-type adult --stage segmentation
# 여기서 Streamlit으로 구간분리 결과를 확인·수정 (4단계, 사람 확인)
streamlit run src\review_app\app.py
python scripts\run_pipeline.py --session sample_01 --video <mp4 경로> --child-age-months 54 --child-gender female --subject-type adult --stage coding
python scripts\run_pipeline.py --session sample_01 --child-age-months 54 --child-gender female --subject-type adult --stage synthesis
```

## 테스트

```powershell
pytest
```

## 공개 사이트 (결과 뷰어) — Streamlit Community Cloud

**중요**: 영상 분석은 항상 로컬에서만 이루어진다. 공개 사이트(`src/public_app/app.py`)는 영상·`MSSB_DATA_DIR`에 전혀 접근하지 않고, 로컬에서 만든 "내보내기 번들"(JSON, 영상 미포함)을 업로드해야만 결과를 보여준다. 업로드한 내용은 저장되지 않고 브라우저 세션이 끝나면 사라진다 — 이건 아동 평가 결과를 제3자 무료 클라우드에 계속 쌓아두지 않기 위한 의도적 설계다.

**로컬에서 결과 내보내기**: `streamlit run src\review_app\app.py`로 세션을 열고 **"🌐 공개 사이트 업로드용 내보내기 (.json)"** 버튼으로 번들 파일을 받는다.

**저장소는 Private로 둔다** — `reference/`(MSSB 실제 대본·매뉴얼·배포자료), `문제점.md`, `플랜.md`는 검사 타당성·저작권 문제로 `.gitignore`에 아예 추가해 저장소에 올리지 않는다(로컬에만 존재). Private여도 Streamlit Community Cloud에서 문제없이 배포된다.

**배포 절차** (한 번만 하면 됨):
1. GitHub에 **Private** 저장소를 만든다 (이미 만들어져 있다면 생략).
2. 이 리포를 그 저장소로 push한다 (`git remote add origin ...`, `git push -u origin master`) — `git push`는 GitHub 로그인 창이 뜨므로 본인이 직접 로그인해야 한다.
3. [share.streamlit.io](https://share.streamlit.io) 에서 GitHub 계정으로 로그인 → 처음이면 Streamlit이 해당 private 저장소에 접근할 수 있도록 GitHub 권한을 요청하는 화면이 뜬다(승인 필요) → "New app" → 이 리포 선택.
4. **Main file path**를 `src/public_app/app.py`로 지정 (레포 루트의 `requirements.txt`가 아니라 `src/public_app/requirements.txt`가 자동으로 쓰인다 — 무거운 CV 라이브러리 없이 가볍게 배포됨).
5. 배포된 앱 화면 > 오른쪽 아래 메뉴 > **Settings > Secrets**에 아래를 붙여넣는다:
   ```
   APP_PASSWORD = "원하는 비밀번호"
   ```
6. 이 비밀번호를 아는 사람만 접근 가능. 배포된 URL(`https://<앱이름>.streamlit.app`)이 곧 "홈페이지" 주소가 된다.

로컬에서 먼저 테스트하려면 `.streamlit/secrets.toml.example`을 `.streamlit/secrets.toml`로 복사해 비밀번호를 채운 뒤 `streamlit run src\public_app\app.py`.

## 지금 상태 (실제로 되는 것 vs 스텁)

**실제 구현되고 동작 확인됨** (`C:\mssb_venv`에 `requirements.txt` 설치 후):
- `video_prep.py` — ffmpeg 오디오/프레임 추출 (실제 ffmpeg 확인됨)
- `transcribe.py` — faster-whisper 전사 + `tag_speakers` 화자 분기
- `pose_analysis.py` — MediaPipe Pose 실제 연동
- `voice_prosody.py` — openSMILE eGeMAPS 실제 연동, **합성 톤(220Hz)으로 피치 추출값(≈219.7Hz) 검증 완료**
- `object_tracking.py` — YOLO+ByteTrack 실제 연동. `models/doll_yolo/`에 가중치가 없으면(마일스톤 0 전) 가짜 탐지 대신 빈 결과를 조용히 반환하도록 설계 (테스트됨)
- `facial_expression.py` — Py-Feat 실제 연동 코드 작성됨. **단, 실제 얼굴이 있는 이미지로 끝까지 검증하지는 못함** — 합성 테스트 이미지로는 얼굴 탐지 자체가 안 되기 때문. 마일스톤 A에서 실제 영상으로 첫 검증 필요
- `segmentation.call_segmentation`, `claude_client.call_stem_coding`, `session_synthesis.call_session_synthesis` — Anthropic API 실제 호출 코드 작성됨. `ANTHROPIC_API_KEY` 필요 (API 응답 형식은 실제 호출 전까지 미검증)
- `schema.py`/`storage.py`/`coding_systems.py`/`posture_metrics.py`/`feature_summary.py` — pydantic 검증까지 포함해 완전히 테스트됨 (`pytest`, 외부 설치 불필요)
- `src/review_app/app.py` — Streamlit 기동 확인됨

**여전히 미확정(TODO)**:
- `config/cv_thresholds_adult.yaml` / `cv_thresholds_child.yaml`의 임계값은 전부 `null` — 마일스톤 A에서 실측 채울 것 (`문제점.md` 3번)
- `models/doll_yolo/`에 파인튜닝된 가중치 없음 — 마일스톤 0(인형 라벨링) 필요
- Py-Feat 실제 검증 — 마일스톤 A에서 실제 영상 필요
