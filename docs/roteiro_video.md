# Roteiro do vídeo (até 15 min)

Preparação:

- rode `python scripts/run_demo.py` uma vez antes de gravar;
- deixe abertos o terminal, o navegador com `outputs/relatorio_demo.html` e o painel (`streamlit run dashboard/app.py`);
- se quiser mostrar o Telegram, preencha o `.env` antes.

| Tempo | Bloco | O que mostrar | Fala-guia |
| --- | --- | --- | --- |
| 0:00–1:30 | Problema e arquitetura | Diagrama do README | "Três pipelines independentes publicam eventos num barramento; um motor de fusão calcula o risco por paciente; um serviço de alertas avisa a equipe. Tudo open source e custo zero." |
| 1:30–2:30 | Contrato de evento e código | `events.py`, árvore do repositório | Explicar por que o contrato comum desacopla os módulos. |
| 2:30–5:00 | Vídeo — fisioterapia | `fisioterapia_P001.mp4` (assimetria) e `fisioterapia_P002.mp4` (queda); relatório da sessão | Pose → ângulos → repetições → DTW contra a referência. Mostrar a tabela por repetição e os quadros-chave. |
| 5:00–6:00 | Vídeo — cirurgia | `relatorio_cirurgia_P004.html` | YOLOv8 + regras: bisturi fora do campo, câmera obstruída, 5 gazes no início e 4 no fim. |
| 6:00–8:30 | Áudio | `consulta_P003.wav` tocando; `analise_P003.json` | Whisper → features do Praat (jitter, shimmer, HNR) → disartria; NLP: termos críticos e negação ("não tenho febre"). Comparar com P002 (normal). |
| 8:30–10:30 | Anomalias | Gráficos de sinais vitais e de movimentação; seção 5 do relatório | Limiares, baseline e multivariado em streaming; erro de dose 10× da enoxaparina; agitação noturna. |
| 10:30–13:00 | Fusão e alerta ao vivo | Terminal com `python scripts/run_demo.py --realtime 1 --skip-eval` e depois o painel | Mostrar o P003 indo de atenção a crítico quando voz, relato e vitais concordam, e a atualização do P001 às 14:20. Reconhecer um alerta no painel. |
| 13:00–14:30 | Resultados e decisões | Seção 8 do relatório (métricas) | Por que noisy-OR em vez de média; por que o Isolation Forest não decide sozinho (22,6 falsos episódios/dia); limites dos dados sintéticos. |
| 14:30–15:00 | Encerramento | README | Como rodar com dados reais e próximos passos. |
