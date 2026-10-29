.PHONY: install install-ml demo eval test api dashboard
install:      ; pip install -r requirements.txt && pip install -e .
install-ml:   ; pip install -r requirements-ml.txt
demo:         ; python scripts/run_demo.py
demo-video:   ; python scripts/run_demo.py --realtime 1.0 --skip-eval
eval:         ; python scripts/evaluate.py
test:         ; pytest -q
api:          ; uvicorn hospital_monitor.api.app:app --reload
dashboard:    ; streamlit run dashboard/app.py
