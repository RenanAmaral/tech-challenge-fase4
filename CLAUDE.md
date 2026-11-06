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
pytest -q                                  # 29 testes
streamlit run dashboard/app.py             # painel em http://localhost:8501
uvicorn hospital_monitor.api.app:app --reload   # API em http://localhost:8000/docs
pip install -r requirements-ml.txt         # opcional: Whisper, YOLOv8, MediaPipe, pysentimiento
```

Python 3.10–3.12 recomendado (MediaPipe ainda não tem wheel para versões mais novas). Se o `Activate.ps1` for bloqueado: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Resultados de referência (da demo atual)

24 eventos, 11 alertas (5 críticos, 6 atenção), 1 atualização, 8 duplicados suprimidos. Sinais vitais (detector combinado): precisão 1,00, recall por episódio 1,00, atraso 8,9 min, 0 falsos/dia. Fisioterapia: recall 1,00 em todos os desvios, 4/128 reps normais marcadas, quedas 8/8 e 0/8 falsas. Cirurgia: 5/5 em cada falha, 0 achados em vídeos normais. NLP: 20/20. Se mudar algo que altere esses números, **atualize `docs/relatorio_tecnico.md` e o README**.

## O que ainda falta / próximos passos

- [ ] Rodar com os modelos reais na máquina local (`requirements-ml.txt`): testar `scripts/analyze_audio.py` com um áudio gravado e `scripts/analyze_video.py fisio` com um vídeo do celular.
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
