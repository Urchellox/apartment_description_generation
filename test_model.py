"""Проверка сохранённой модели: python test_model.py [папка_модели]
Генерирует описания для тестовых входов и проверяет, что в тексте нет ошибок в фактах."""
import json, os, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "final_model"
N_SAMPLES = 3   # сколько вариантов генерировать на каждый вход
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device).eval()
opts = json.load(open(os.path.join(MODEL_DIR, "ui_options.json"), encoding="utf-8"))
print("Модель загружена:", MODEL_DIR, "| устройство:", device, "| параметров: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))


def build_input(c):
    return "; ".join([f"комнат: {c['rooms']}", f"площадь: {c['area']:.1f} м²".replace(".", ","),
                      f"этаж: {c['floor']}", f"состояние: {c['condition']}",
                      f"санузел: {c['bathroom']}", f"балкон: {c['balcony']}"])


@torch.no_grad()
def generate(c, n=N_SAMPLES):
    prompt = opts["prompt_prefix"] + build_input(c) + opts["prompt_suffix"]
    enc = tok([prompt] * n, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    out = model.generate(**enc, max_new_tokens=320, do_sample=True, temperature=0.7, top_p=0.9,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return [t.strip() for t in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]


def check(c, text):
    """Список найденных проблем (пустой = всё ок)."""
    t, errs = text.lower(), []
    if len(t.split()) < 15: errs.append("слишком короткий текст")
    if not re.search(r"[.!?]$", t): errs.append("текст обрывается")
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*м²", t)
    if m and abs(float(m.group(1).replace(",", ".")) - c["area"]) > 0.05: errs.append(f"площадь: в тексте {m.group(1)}")
    m = re.search(r"(\d+)[\s-]*комнатн", t)
    if m and int(m.group(1)) != c["rooms"]: errs.append(f"комнат: в тексте {m.group(1)}")
    m = re.search(r"(\d+)\s*этаж", t)
    if m and int(m.group(1)) != c["floor"]: errs.append(f"этаж: в тексте {m.group(1)}")
    sep, comb = "раздельн" in t, "совмещ" in t
    if sep != comb and ((sep and c["bathroom"] != "Раздельный") or (comb and c["bathroom"] != "Совмещённый")):
        errs.append("санузел не тот")
    no_b = re.search(r"без балкона|нет балкона|балкон отсутствует|балкона нет|отсутствие балкона|отсутствует балкон", t)
    if no_b and c["balcony"] != "Нет": errs.append("написано, что балкона нет, а он есть")
    if not no_b and "балкон" in t and c["balcony"] == "Нет": errs.append("упомянут балкон, которого нет")
    return errs


def pick(key, i):   # значения берём из ui_options, чтобы совпадали с обучающими
    v = opts[key]; return v[i % len(v)]

cases = [dict(rooms=r, area=a, floor=f, condition=pick("condition", i), bathroom=pick("bathroom", i), balcony=pick("balcony", i))
         for i, (r, a, f) in enumerate([(1, 32.5, 2), (2, 58.0, 4), (2, 65.0, 8), (3, 82.3, 5), (3, 74.0, 12), (4, 110.0, 3)])]

total = bad = 0
for c in cases:
    print("\n" + "=" * 70 + "\nВХОД:", build_input(c))
    for k, text in enumerate(generate(c), 1):
        errs = check(c, text); total += 1; bad += bool(errs)
        print(f"\n[{k}] {'OK' if not errs else 'ОШИБКИ: ' + '; '.join(errs)}\n{text}")

print("\n" + "=" * 70 + f"\nИТОГО: {total - bad}/{total} текстов без ошибок фактов ({100 * (total - bad) / total:.0f}%)")
print("Модель работает" if bad / total < 0.2 else "Много ошибок: проверьте формат промпта и что сохранена именно обученная модель")
