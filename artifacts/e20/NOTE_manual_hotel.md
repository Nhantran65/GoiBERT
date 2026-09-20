Root launched the two Hotel trials manually (detached workers on cuda:0 / cuda:2) at ~18:40 UTC to use free GPUs, while the
watcher's controller runs the Restaurant trials sequentially on cuda:1. When the controller reaches the Hotel trials it will
find the output dirs already present and raise FileExistsError -> the watcher then runs eval_e20.py, possibly before the manual
Hotel trials finish. Re-run `.venv-e0-cuda/bin/python scripts/eval_e20.py` after both Hotel status.json say completed
(logs: artifacts/e20/logs/*__absa_hotel_manual.log).
