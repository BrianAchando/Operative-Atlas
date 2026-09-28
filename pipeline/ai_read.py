"""Put the atlas questions to NV-Reason-CT and save its answers for the atlas.

    python pipeline/ai_read.py work/ct.nii.gz                      # reads public/data/ai_questions.json
    python pipeline/ai_read.py ct.nii.gz --questions ai_questions.json --out ai_answers.json

Writes ai_answers.json: {"model", "date", "gpu", "answers": {question id: {"answer", "thinking"}}}. Copy it to
public/data/ in the atlas; the "AI read" panels pick it up. Needs a CUDA GPU with about 16-24 GB (Colab L4 or A100,
Kaggle T4 may fit). Resumable: answers already in --out are kept, so a run cut short can be started again.

NV-Reason-CT is a research model (OpenMDW-1.1), not a medical device. Its answers are shown in the atlas as unverified.
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ct', help='the reference CT, NIfTI in Hounsfield units (work/ct.nii.gz)')
    ap.add_argument('--questions', default=str(ROOT / 'public' / 'data' / 'ai_questions.json'))
    ap.add_argument('--out', default=str(ROOT / 'public' / 'data' / 'ai_answers.json'))
    ap.add_argument('--model', default='nvidia/NV-Reason-CT')
    ap.add_argument('--no-thinking', action='store_true', help='faster, shorter answers without the reasoning trace')
    ap.add_argument('--max-new-tokens', type=int, default=2048)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('NV-Reason-CT needs a CUDA GPU (about 16-24 GB). Use Colab (L4/A100) or Kaggle (T4).')

    qs = json.loads(Path(a.questions).read_text())['questions']
    out = Path(a.out)
    done = json.loads(out.read_text()) if out.exists() else {}
    answers = done.get('answers', {})
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    gpu = torch.cuda.get_device_name(0)
    print(f'{gpu}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB, {dtype}')

    model = AutoModelForImageTextToText.from_pretrained(a.model, trust_remote_code=True, dtype=dtype, attn_implementation='sdpa').eval().to('cuda')
    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)

    def ask(prompt):
        msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=not a.no_thinking)
        inp = proc(text=text, images3d=[a.ct], anatomy_region='chest', return_tensors='pt').to(model.device)
        with torch.inference_mode():
            ids = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False, use_cache=True)
        r = proc.batch_decode(ids[:, inp.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # split the reasoning trace from the answer
        m = re.search(r'<think>(.*?)</think>', r, re.S)
        think = m.group(1).strip() if m else ''
        ans = re.sub(r'<think>.*?</think>', '', r, flags=re.S)
        if '</think>' in ans: think, ans = ans.split('</think>', 1)
        return ans.strip(), think.strip()

    cache = {v['prompt']: v for v in answers.values() if 'prompt' in v}
    for i, q in enumerate(qs, 1):
        if q['id'] in answers: continue
        t0 = time.time()
        if q['q'] in cache: ans, think = cache[q['q']]['answer'], cache[q['q']]['thinking']
        else: ans, think = ask(q['q'])
        answers[q['id']] = cache[q['q']] = {'prompt': q['q'], 'answer': ans, 'thinking': think}
        print(f'[{i}/{len(qs)}] {q["id"]}  {time.time() - t0:.0f} s\n  {ans[:160]!r}')
        out.write_text(json.dumps({'model': a.model, 'date': time.strftime('%Y-%m-%d'), 'gpu': gpu, 'answers': answers}, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
