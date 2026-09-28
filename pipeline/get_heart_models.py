"""Download the licensed TotalSegmentator heart models with patience: a long read timeout and retries.

    python pipeline/get_heart_models.py            (after the licence key is set)

The model server can be slow from here; TotalSegmentator's own download gives up after 5 minutes without data.
This waits up to an hour per stall and retries each model up to 6 times. Models: 301 heart chambers (needed),
920 aortic sinuses, 509 coronary arteries. Once all are in, run segment_heart.py; it will not download again.
"""
import time

import requests

_post = requests.post


def patient_post(*a, **k):
    k['timeout'] = (60, 3600)                                    # connect, and read (per stall)
    return _post(*a, **k)


requests.post = patient_post

from totalsegmentator.libs import download_pretrained_weights  # noqa: E402

for task_id, name in ((301, 'heart chambers'), (920, 'aortic sinuses'), (509, 'coronary arteries')):
    for attempt in range(1, 7):
        try:
            print(f'{name} (task {task_id}), attempt {attempt} ...', flush=True)
            download_pretrained_weights(task_id)
            print(f'  {name}: done', flush=True)
            break
        except Exception as e:                                   # noqa: BLE001
            print(f'  failed: {e}; retrying in 60 s', flush=True)
            time.sleep(60)
    else:
        print(f'  {name}: gave up after 6 attempts')
