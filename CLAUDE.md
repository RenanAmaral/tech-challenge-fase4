# CLAUDE.md — contexto do projeto para o Claude Code

Este arquivo é lido automaticamente pelo Claude Code. Ele resume o que o projeto é, as decisões já tomadas e como trabalhar nele.

## O trabalho

Tech Challenge da **Fase 4 da pós-graduação em IA para Devs (FIAP)**. O enunciado pede uma solução de **monitoramento multimodal de pacientes** num hospital:

- **Vídeo:** detectar desvios em sessões de fisioterapia (pose estimation) e eventos críticos em vídeos de cirurgia (YOLOv8); gerar relatório automático.
- **Áudio:** transcrever consultas e detectar sinais de fadiga, disartria e falta de ar; identificar termos críticos e sentimento.
- **Anomalias:** sinais vitais, evolução de prescrições e padrões de movimentação na internação.
- **Alertas** automáticos para a equipe médica.

**Entregáveis:** repositório Git (código + relatório técnico com fluxo multimodal, modelos e resultados) e **vídeo de até 15 min** no YouTube/Vimeo.

O usuário é aluno da pós; o trabalho é em grupo. Responder em **português do Brasil**.

## Decisões já tomadas (não reverter sem pedir)

- **Azure não é obrigatória** (confirmado com o professor). Tudo é **open source, gratuito e roda em CPU**. Equivalências: Azure Speech → faster-whisper; Text Analytics → léxico clínico pt-BR + negação estilo NegEx; sentimento → pysentimiento ou léxico; OpenPose → MediaPipe (aceita JSON do OpenPose); alertas → Telegram/Discord.
- O professor avalia **como a solução é construída** (arquitetura, desacoplamento, justificativas), mais do que a tecnologia.
- **Fusão por noisy-OR ponderado**, não média: um achado grave sozinho não pode ser "diluído" por modalidades normais. Eventos sentinela (queda, gaze retida) não são atenuados.
- **Isolation Forest não decide sozinho** (22,6 falsos episódios/dia em sinais vitais; F1 0,63 em prescrições). Fica como comparação e anotação.
- Desvios de execução na fisioterapia têm escore 0,50–0,75 (atenção, vão para o relatório); só a queda é crítica (0,95).
- **Dados 100% sintéticos e rotulados** na demo (LGPD). Os números validam a implementação, não o desempenho clínico — o relatório diz isso explicitamente; manter essa honestidade.

## Arquitetura

```
pipelines por modalidade → Event (contrato comum) → barramento → fusão (janela 15 min) → alertas → equipe
```

- `src/hospital_monitor/events.py` — contrato `Event` (patient_id, modality, kind, anomaly_score 0–1, timestamp, description, evidence). **Todo módulo novo deve emitir `Event`**; é o que mantém os pipelines desacoplados.
- `bus.py` — `InMemoryBus` (demo/testes) e `RedisStreamBus`.
- `config.py` + `config/settings.yaml` — limiares (NEWS2), pesos da fusão, cooldown de alertas.
- `video/` — `pose.py` (MediaPipe, YOLOv8-pose, JSON), `kinematics.py` (ângulos, DTW), `exercise_analyzer.py`, `surgical.py` (regras sobre detecções YOLOv8), `render.py`, `report.py`.
- `audio/` — `transcribe.py` (Whisper ou `.txt` ao lado do áudio), `acoustic.py` (Praat: jitter, shimmer, HNR, pausas, inspirações), `voice_analyzer.py` (z-score vs. perfil saudável + Random Forest), `clinical_nlp.py`, `pipeline.py`.
- `anomaly/` — `vitals.py` (streaming: limiares + baseline robusto + multivariado χ²), `prescriptions.py` (regras farmacológicas), `movement.py`, `metrics.py`.
- `fusion/engine.py` — noisy-OR + bônus de concordância (0,15 por modalidade extra).
- `alerts/manager.py` — deduplicação (30 min), atualização quando entra modalidade nova, escalonamento, reconhecimento. `notifiers.py` — console, arquivo, Telegram, Discord.
- `orchestrator.py` — liga barramento → fusão → alertas; `publish_many` agrupa eventos simultâneos.
- `services.py` — monta pipelines e gera artefatos em `models/` se faltarem.
- `evaluation.py` — métricas de todos os detectores (seeds fixos).
- `api/app.py` — FastAPI. `dashboard/app.py` — Streamlit.
- `scripts/run_demo.py` — demo de um dia (28/10/2026) com 4 pacientes; `report_builder.py` monta `outputs/relatorio_demo.html`.

## Comandos (Windows / PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python scripts/run_demo.py                 # ~2 min; saídas em outputs/
python scripts/run_demo.py --realtime 1 --skip-eval   # para gravar o vídeo
pytest -q                                  # 33 testes
streamlit run dashboard/app.py             # painel em http://localhost:8501
uvicorn hospital_monitor.api.app:app --reload   # API em http://localhost:8000/docs
pip install -r requirements-ml.txt         # opcional: Whisper, YOLOv8, MediaPipe, pysentimiento
```

Python 3.10–3.12 recomendado (MediaPipe ainda não tem wheel para versões mais novas). Se o `Activate.ps1` for bloqueado: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Whisper no Windows (problema conhecido)

- **Resolvido na máquina do usuário** (05/11/2026): VC++ Redistributable instalado, ctranslate2 4.8.5, `av<16` (o faster-whisper 1.2.x quebra com PyAV 16+; já fixado no `requirements-ml.txt`).
- Áudio **sintético** (demo) não tem fala real: o Whisper devolve sílabas sem sentido. Por isso `run_demo.py` usa sempre `sidecar`, e a API usa sidecar quando o cliente envia `transcript`. Teste o Whisper só com gravação de voz real.
- `HM_TRANSCRIBER` só vale quando o chamador pede `auto`; escolha explícita é respeitada.
- Sintoma original: `WhisperModel(...)` dá *segfault* (violação de acesso) com ctranslate2 4.8.x, em qualquer modelo ou `compute_type`. Com o torch instalado (vem do pysentimiento), aparece `WinError 126` ao carregar `shm.dll`.
- Causa mais provável: falta o **Microsoft Visual C++ Redistributable 2015–2022 (x64)**. Correção: `winget install Microsoft.VCRedist.2015+.x64` (pedir ao usuário; é instalação no sistema) e reiniciar o terminal.
- Se persistir com o torch carregado: `$env:KMP_DUPLICATE_LIB_OK="TRUE"` (conflito de OpenMP entre torch e ctranslate2) ou testar o Whisper num venv sem o pysentimiento.
- O modo `auto` testa o Whisper num subprocesso (`audio/transcribe.py: whisper_works`) e cai para o `.txt` se ele quebrar. Para forçar: `$env:HM_TRANSCRIBER="sidecar"`.
- O Whisper é opcional para a entrega: o pipeline de voz e NLP funciona com a transcrição do `.txt` (UTF-8; o leitor aceita BOM e cp1252).
- `scripts/analyze_audio.py` imprime um resumo (use `--full` para o JSON) e grava o JSON completo em `outputs/real/audio_<paciente>.json`. Ele força UTF-8 no console; em outros scripts, use `$env:PYTHONUTF8="1"`.

## Voz real: use o baseline do paciente

- O perfil "saudável" padrão (`models/voice_reference.json`) é **sintético**: contra ele, voz humana real dá falso positivo (teste de 06/11/2026: voz normal pelo WhatsApp → disartria 0,73).
- Para voz real: `python scripts/voice_baseline.py normal1.mp4 normal2.mp4 normal3.mp4 --patient P010` (3–5 gravações de 15–30 s, falando normal, **mesmo canal** da consulta). Grava `models/voice_baseline_P010.json` (fora do Git); `analyze_audio.py --patient P010` passa a usá-lo e desliga o Random Forest.
- `load_audio` aceita MP4/M4A/OPUS do WhatsApp via ffmpeg.
- A condição e o escore de voz vêm do desvio do perfil; o Random Forest só atenua quando confia em "normal".

## Resultados de referência (da demo atual)

24 eventos, 11 alertas (5 críticos, 6 atenção), 1 atualização, 8 duplicados suprimidos. Sinais vitais (detector combinado): precisão 1,00, recall por episódio 1,00, atraso 8,9 min, 0 falsos/dia. Fisioterapia: recall 1,00 em todos os desvios, 4/128 reps normais marcadas, quedas 8/8 e 0/8 falsas. Cirurgia: 5/5 em cada falha, 0 achados em vídeos normais. NLP: 20/20. Se mudar algo que altere esses números, **atualize `docs/relatorio_tecnico.md` e o README**.

## O que ainda falta / próximos passos

- [x] Whisper instalado e rodando na máquina local.
- [x] Voz real testada (06/11/2026): baseline com 3 textos + consultas A–D pelo WhatsApp; B (falta de ar simulada) → dispneia 0,67; A, C, D → voz normal. Resultados na seção 6.3 do relatório técnico. Gravações ficam fora do Git.
- [ ] Rodar as mesmas consultas com `--transcriber whisper` na máquina local e conferir a transcrição.
- [x] Vídeo de referência real (06/11/2026): 6 agachamentos corretos, YOLOv8m-pose 2D (`--backend yolo`, pesos baixados do GitHub), 6 reps segmentadas. Referência em `models/squat_reference_real.json` (fora do Git). Assimetria agora é relativa à referência (efeito do ângulo da câmera em 2D); reps rasas na borda do vídeo são descartadas.
- [ ] Gravar o vídeo de **sessão** (6 agachamentos, 2º e 5º errados) com a **mesma câmera e posição** da referência e rodar `python scripts/analyze_video.py fisio sessao.mp4 --patient P010 --backend yolo`. O esqueleto é desenhado sobre o vídeo original em `outputs/real/fisio_P010.mp4`.
- [ ] Gravar vídeos de exercício (correto e incorreto) e consultas simuladas com consentimento do grupo.
- [ ] YOLOv8 de cirurgia: precisa de pesos com fine-tuning em instrumental (m2cai16-tool, Cholec80 ou dataset rotulado no Roboflow) em `models/surgical_yolov8n.pt`. Sem isso, a demo usa detecções simuladas.
- [ ] Opcional: validar sinais vitais em dados reais com `scripts/download_physionet.py --records 1 2 3 --inject desaturation`.
- [ ] Gravar o vídeo seguindo `docs/roteiro_video.md` e colocar o link no README.

## Convenções

- Código e comentários em português; identificadores em inglês ou português sem acento (`dessaturacao`, `queda`).
- Rótulos para usuários passam por `reporting.kind_pt()`.
- Horários da demo em America/Sao_Paulo; eventos sempre com timezone.
- Antes de commitar: `pytest -q` e `ruff check src scripts tests dashboard --select E,F,W --ignore E501`.
- Nunca versionar `.env`, dados reais de paciente ou pesos grandes (`*.pt`, `*.task` estão no `.gitignore`).
