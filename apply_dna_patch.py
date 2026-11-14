#!/usr/bin/env python3
"""
Применение и криминалистическая проверка компактного дифф-патча
ДНК-редактирования весов (SGNE-CC, 3.13 МБ) поверх Qwen/Qwen3.5-0.8B (BF16).

Примеры запуска:
  # 1. Скачать Qwen/Qwen3.5-0.8B из HuggingFace, проверить все инварианты и собрать модель:
  python3 apply_dna_patch.py --out ./Qwen3.5-0.8B-SGNE-CC

  # 2. Только проверить SHA-256 и все геометрические метрики статьи в памяти (без записи 1.75 ГБ):
  python3 apply_dna_patch.py --base-model /path/to/Qwen3.5-0.8B --verify-only

  # 3. Сравнить бок о бок геометрию SGNE-CC и чекпоинта MIT-p50c (HF repo_id, папка или .npz):
  python3 apply_dna_patch.py \
      --compare RikiyaT/Qwen3.5-0.8B-weight-space-programming --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import shutil
import sys

sys.dont_write_bytecode = True
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DELTA_PATH = str(SCRIPT_DIR / "sgne_cc_dna_delta.npz")
EXPECTED_DELTA_SHA256 = "4e4d36e032869bc10f56e2a2033a52d3f2774321cf867ef6f74274a7c0d1566d"
DEFAULT_BASE_REPO = "Qwen/Qwen3.5-0.8B"
EXPECTED_BASE_SHA256 = {
    "model.safetensors": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    "config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
}

ALLOWED_COL_SUFFIXES = (
    "mlp.down_proj.weight",
    "linear_attn.out_proj.weight",
    "self_attn.o_proj.weight",
)
ALLOWED_GATE_SUFFIX = "linear_attn.in_proj_a.weight"

NPZ_PREFIX_TO_TENSOR = {
    "mlp_down": "mlp.down_proj.weight",
    "dnet_out": "linear_attn.out_proj.weight",
    "gqa_o": "self_attn.o_proj.weight",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def u16_to_bf16(u16_arr: np.ndarray) -> torch.Tensor:
    u16_le = np.asarray(u16_arr, dtype="<u2")
    f32_le = (u16_le.astype("<u4") << 16).view("<f4")
    f32_native = np.asarray(f32_le, dtype=np.float32)
    return torch.from_numpy(f32_native).to(dtype=torch.bfloat16).clone()


def resolve_base_dir(base_model: str, total_steps: int = 4) -> Path:
    local_path = Path(base_model).expanduser()
    if local_path.is_dir() and (local_path / "model.safetensors").is_file():
        return local_path
    if base_model.startswith((".", "/", "~")):
        raise FileNotFoundError(
            f"Локальная папка базовой модели с model.safetensors не найдена: {local_path.resolve()}"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Пакет huggingface_hub не установлен. Установите: pip install -r requirements.txt"
        ) from exc
    print(f"[1/{total_steps}] Загрузка базовой модели {base_model} из HuggingFace Hub...")
    dl_dir = snapshot_download(repo_id=base_model)
    return Path(dl_dir)


def verify_base_model_hashes(base_dir: Path) -> None:
    for fname, exp_sha in EXPECTED_BASE_SHA256.items():
        fpath = base_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"В папке базовой модели отсутствует обязательный файл {fname}: {base_dir}")
        act_sha = sha256_file(fpath)
        print(f"[SHA-256 base] {fname}: {act_sha}")
        if act_sha != exp_sha:
            raise ValueError(
                f"Несовпадение SHA-256 файла базовой модели {fname}!\n"
                f"  Ожидалось: {exp_sha}\n"
                f"  Получено:  {act_sha}\n"
                f"  (Используйте --skip-hash-check только при осознанной проверке альтернативной ревизии.)"
            )


def resolve_compare_path(comp_spec: str) -> Path:
    local_path = Path(comp_spec).expanduser()
    if local_path.exists():
        return local_path
    if comp_spec.startswith((".", "/", "~")) or comp_spec.endswith((".npz", ".safetensors")):
        raise FileNotFoundError(f"Локальный путь для --compare не найден: {local_path.resolve()}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Пакет huggingface_hub не установлен. Установите: pip install -r requirements.txt"
        ) from exc
    print(f"  [HF] Загрузка чекпоинта для сравнения {comp_spec} из HuggingFace Hub...")
    dl_dir = snapshot_download(repo_id=comp_spec)
    return Path(dl_dir)


def _read_base_shapes(base_sf_path: Path) -> dict[str, tuple[int, ...]]:
    with safe_open(str(base_sf_path), framework="pt", device="cpu") as fb:
        return {k: tuple(fb.get_slice(k).get_shape()) for k in fb.keys()}


def load_patch_maps(
    patch_path: Path, base_sf_path: Path
) -> tuple[dict[str, tuple[np.ndarray, torch.Tensor]], dict[str, torch.Tensor]]:
    """Загружает и строго валидирует изменённые столбцы и строки ворот из .npz или safetensors."""
    col_patch_map: dict[str, tuple[np.ndarray, torch.Tensor]] = {}
    full_patch_map: dict[str, torch.Tensor] = {}
    base_shapes = _read_base_shapes(base_sf_path)

    if patch_path.is_dir() and (patch_path / "model.safetensors").is_file():
        patch_path = patch_path / "model.safetensors"

    if patch_path.suffix == ".npz":
        npz = dict(np.load(patch_path, allow_pickle=False))
        consumed_keys: set[str] = set()

        col_re = re.compile(r"^(mlp_down|dnet_out|gqa_o)_L(\d+)_(changed_cols|cols_bf16_u16)$")
        gate_re = re.compile(r"^da_L(\d+)_full_bf16_u16$")

        for k in sorted(npz.keys()):
            if k in consumed_keys:
                continue
            m_col = col_re.match(k)
            m_gate = gate_re.match(k)
            if m_col:
                prefix, layer_s, _ = m_col.groups()
                layer_i = int(layer_s)
                k_cols = f"{prefix}_L{layer_i}_changed_cols"
                k_vals = f"{prefix}_L{layer_i}_cols_bf16_u16"
                if k_cols not in npz or k_vals not in npz:
                    raise ValueError(f"Непарный ключ в .npz патче: ожидались оба {k_cols} и {k_vals}")
                consumed_keys.update((k_cols, k_vals))

                t_key = f"model.language_model.layers.{layer_i}.{NPZ_PREFIX_TO_TENSOR[prefix]}"
                if t_key not in base_shapes:
                    raise ValueError(f"Целевой тензор {t_key} отсутствует в базовой модели")
                rows_expected, cols_total = base_shapes[t_key]

                raw_cols = npz[k_cols]
                if raw_cols.ndim != 1 or not np.issubdtype(raw_cols.dtype, np.integer):
                    raise ValueError(f"Ключ {k_cols} должен быть одномерным целочисленным массивом индексов")
                cols = raw_cols.astype(np.int64)
                if cols.size == 0:
                    raise ValueError(f"Ключ {k_cols} пуст (0 столбцов)")
                if len(np.unique(cols)) != cols.size:
                    raise ValueError(f"Ключ {k_cols} содержит дублирующиеся индексы столбцов")
                if int(cols.min()) < 0 or int(cols.max()) >= cols_total:
                    raise ValueError(
                        f"Индексы столбцов в {k_cols} выходят за пределы [0, {cols_total - 1}]: "
                        f"[{int(cols.min())}, {int(cols.max())}]"
                    )

                vals_u16 = npz[k_vals]
                if vals_u16.dtype != np.uint16 or vals_u16.shape != (rows_expected, cols.size):
                    raise ValueError(
                        f"Неверная форма или тип {k_vals}: получено {vals_u16.shape} ({vals_u16.dtype}), "
                        f"ожидалось ({rows_expected}, {cols.size}) uint16"
                    )
                vals_bf16 = u16_to_bf16(vals_u16)
                if not bool(torch.isfinite(vals_bf16).all().item()):
                    raise ValueError(f"Тензор {k_vals} содержит NaN или Inf после декодирования BF16")

                col_patch_map[t_key] = (cols, vals_bf16)

            elif m_gate:
                layer_i = int(m_gate.group(1))
                consumed_keys.add(k)
                t_key = f"model.language_model.layers.{layer_i}.{ALLOWED_GATE_SUFFIX}"
                if t_key not in base_shapes:
                    raise ValueError(f"Целевой тензор ворот {t_key} отсутствует в базовой модели")
                exp_shape = base_shapes[t_key]
                vals_u16 = npz[k]
                if vals_u16.dtype != np.uint16 or vals_u16.shape != exp_shape:
                    raise ValueError(
                        f"Неверная форма или тип {k}: получено {vals_u16.shape} ({vals_u16.dtype}), "
                        f"ожидалось {exp_shape} uint16"
                    )
                vals_bf16 = u16_to_bf16(vals_u16)
                if not bool(torch.isfinite(vals_bf16).all().item()):
                    raise ValueError(f"Тензор {k} содержит NaN или Inf после декодирования BF16")
                full_patch_map[t_key] = vals_bf16
            else:
                raise ValueError(f"Неизвестный ключ в .npz патче: {k!r}")

        return col_patch_map, full_patch_map

    if patch_path.suffix == ".safetensors":
        with safe_open(str(base_sf_path), framework="pt", device="cpu") as fb, safe_open(
            str(patch_path), framework="pt", device="cpu"
        ) as fp:
            fb_keys = set(fb.keys())
            fp_keys = set(fp.keys())
            if fb_keys != fp_keys:
                missing = sorted(fb_keys - fp_keys)[:5]
                extra = sorted(fp_keys - fb_keys)[:5]
                raise ValueError(
                    f"Набор тензоров в {patch_path} не совпадает с базовой моделью "
                    f"(missing={missing}, extra={extra})"
                )
            for key in sorted(fb_keys):
                wb = fb.get_tensor(key)
                wp = fp.get_tensor(key)
                if wb.shape != wp.shape:
                    raise ValueError(f"Несовпадение формы тензора {key}: {wp.shape} против базового {wb.shape}")
                wb_bf16 = wb.to(dtype=torch.bfloat16)
                wp_bf16 = wp.to(dtype=torch.bfloat16)
                if torch.equal(wb_bf16, wp_bf16):
                    continue
                if not bool(torch.isfinite(wp_bf16).all().item()):
                    raise ValueError(f"Изменённый тензор {key} в {patch_path} содержит NaN или Inf")

                is_layer_tensor = key.startswith("model.language_model.layers.")
                if is_layer_tensor and key.endswith(ALLOWED_COL_SUFFIXES):
                    diff_cols = (wb_bf16 != wp_bf16).any(dim=0)
                    cols = torch.where(diff_cols)[0].numpy().astype(np.int64)
                    col_patch_map[key] = (cols, wp_bf16[:, diff_cols].clone())
                elif is_layer_tensor and key.endswith(ALLOWED_GATE_SUFFIX):
                    full_patch_map[key] = wp_bf16.clone()
                else:
                    raise ValueError(
                        f"Обнаружена модификация вне поддерживаемого пространства выходных проекций и ворот: {key}"
                    )
        return col_patch_map, full_patch_map

    raise ValueError(f"Неподдерживаемый формат патча: {patch_path}")


def audit_patch_geometry(
    base_sf_path: Path,
    col_patch_map: dict[str, tuple[np.ndarray, torch.Tensor]],
    full_patch_map: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Вычисляет все проверяемые геометрические характеристики патча в двойной точности (FP64)."""
    total_text_params = 0
    total_write_cols = 0
    total_gate_rows = 0

    mlp_matrices = 0
    mlp_cols = 0
    dnet_matrices = 0
    dnet_cols = 0
    gqa_cols = 0

    min_ratio = float("inf")
    max_ratio = 0.0
    max_abs_dev_pct = 0.0
    min_cos = 1.0
    frob_sq = 0.0
    gate_frob_sq = 0.0
    base_sq = 0.0
    angles_deg: list[float] = []

    layer_svd: dict[str, tuple[int, float]] = {}
    gate_details: list[str] = []
    changed_gate_rows = 0
    selected_region_scalars = 0
    bitwise_changed_scalars = 0

    with safe_open(str(base_sf_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            sh = f.get_slice(key).get_shape()
            if key.startswith("model.language_model."):
                n_el = 1
                for s in sh:
                    n_el *= s
                total_text_params += n_el
                if key.endswith(ALLOWED_COL_SUFFIXES):
                    total_write_cols += sh[1]
                elif key.endswith(ALLOWED_GATE_SUFFIX):
                    total_gate_rows += sh[0]

            if key in col_patch_map:
                cols_np, vals_bf16 = col_patch_map[key]
                idx = torch.from_numpy(cols_np).long()
                w_bf16 = f.get_tensor(key).to(dtype=torch.bfloat16)
                w_orig_bf16 = w_bf16[:, idx]
                w_new_bf16 = vals_bf16.to(dtype=torch.bfloat16)

                selected_region_scalars += int(w_orig_bf16.numel())
                bitwise_changed_scalars += int((w_orig_bf16 != w_new_bf16).sum().item())

                w_orig = w_orig_bf16.double()
                w_new = w_new_bf16.double()
                dw = w_new - w_orig

                orig_norms = torch.linalg.vector_norm(w_orig, dim=0)
                new_norms = torch.linalg.vector_norm(w_new, dim=0)
                ratios = new_norms / orig_norms
                devs = (ratios - 1.0).abs() * 100.0
                cosines = (w_orig * w_new).sum(dim=0) / (orig_norms * new_norms)

                min_ratio = min(min_ratio, float(ratios.min().item()))
                max_ratio = max(max_ratio, float(ratios.max().item()))
                max_abs_dev_pct = max(max_abs_dev_pct, float(devs.max().item()))
                min_cos = min(min_cos, float(cosines.min().item()))
                frob_sq += float((dw * dw).sum().item())
                base_sq += float((w_orig * w_orig).sum().item())
                angles_deg.extend(torch.rad2deg(torch.arccos(cosines.clamp(-1.0, 1.0))).tolist())

                s_vals = torch.linalg.svdvals(dw)
                rank1_pct = float((s_vals[0] ** 2 / (s_vals ** 2).sum() * 100.0).item())

                layer_id = key.split(".")[3]
                if key.endswith("mlp.down_proj.weight"):
                    mlp_matrices += 1
                    mlp_cols += int(idx.numel())
                    layer_svd[f"mlp_down_L{layer_id}"] = (int(idx.numel()), rank1_pct)
                elif key.endswith("linear_attn.out_proj.weight"):
                    dnet_matrices += 1
                    dnet_cols += int(idx.numel())
                    layer_svd[f"dnet_out_L{layer_id}"] = (int(idx.numel()), rank1_pct)
                elif key.endswith("self_attn.o_proj.weight"):
                    gqa_cols += int(idx.numel())

            elif key in full_patch_map:
                w_orig_full = f.get_tensor(key).to(dtype=torch.bfloat16)
                w_rep = full_patch_map[key].to(dtype=torch.bfloat16)
                diff_mask = (w_orig_full != w_rep).any(dim=1)
                rows = torch.where(diff_mask)[0].tolist()
                changed_gate_rows += len(rows)
                selected_region_scalars += len(rows) * int(w_orig_full.shape[1])
                bitwise_changed_scalars += int((w_orig_full != w_rep).sum().item())
                layer_id = key.split(".")[3]
                for r in rows:
                    r0 = w_orig_full[r].double()
                    r1 = w_rep[r].double()
                    dr = r1 - r0
                    gate_frob_sq += float((dr * dr).sum().item())
                    n0 = float(torch.linalg.vector_norm(r0).item())
                    n1 = float(torch.linalg.vector_norm(r1).item())
                    rcos = float(((r0 @ r1) / (n0 * n1)).item())
                    gate_details.append(
                        f"L{layer_id} row {r} ({r + 1}-я из {w_orig_full.shape[0]} голов): "
                        f"‖a‖₂ {n0:.6f} -> {n1:.6f}, cos={rcos:.6f}"
                    )

    total_edited_cols = mlp_cols + dnet_cols + gqa_cols
    selected_pct = (selected_region_scalars / total_text_params * 100.0) if total_text_params else 0.0
    bitwise_pct = (bitwise_changed_scalars / total_text_params * 100.0) if total_text_params else 0.0
    frob_sq_per_col = frob_sq / max(total_edited_cols, 1)
    rel_frob = math.sqrt(frob_sq / base_sq) if base_sq > 0 else 0.0
    if angles_deg:
        ang_sorted = sorted(angles_deg)
        median_angle = float(np.median(np.asarray(ang_sorted)))
        max_angle = ang_sorted[-1]
    else:
        median_angle = max_angle = 0.0

    return {
        "total_text_params": total_text_params,
        "total_write_cols": total_write_cols,
        "total_gate_rows": total_gate_rows,
        "total_edited_cols": total_edited_cols,
        "mlp_matrices": mlp_matrices,
        "mlp_cols": mlp_cols,
        "dnet_matrices": dnet_matrices,
        "dnet_cols": dnet_cols,
        "gqa_cols": gqa_cols,
        "changed_gate_rows": changed_gate_rows,
        "gate_details": gate_details,
        "edited_params": selected_region_scalars,
        "selected_region_scalars": selected_region_scalars,
        "selected_pct": selected_pct,
        "bitwise_changed_scalars": bitwise_changed_scalars,
        "bitwise_pct": bitwise_pct,
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
        "max_abs_dev_pct": max_abs_dev_pct,
        "min_cos": min_cos,
        "frob_norm": math.sqrt(frob_sq),
        "total_frob_norm": math.sqrt(frob_sq + gate_frob_sq),
        "frob_sq_per_col": frob_sq_per_col,
        "rel_frob": rel_frob,
        "median_angle": median_angle,
        "max_angle": max_angle,
        "layer_svd": layer_svd,
    }


def assert_official_sgne_invariants(stats: dict[str, Any]) -> None:
    checks = [
        ("total_text_params", stats["total_text_params"], 752_393_024),
        ("selected_region_scalars", stats["selected_region_scalars"], 2_032_640),
        ("bitwise_changed_scalars", stats["bitwise_changed_scalars"], 1_993_382),
        ("total_edited_cols", stats["total_edited_cols"], 1984),
        ("mlp_cols", stats["mlp_cols"], 802),
        ("dnet_cols", stats["dnet_cols"], 1182),
        ("gqa_cols", stats["gqa_cols"], 0),
        ("changed_gate_rows", stats["changed_gate_rows"], 1),
    ]
    for name, got, exp in checks:
        if got != exp:
            raise AssertionError(f"Нарушен инвариант SGNE-CC {name}: получено {got}, ожидалось {exp}")
    if not (stats["max_abs_dev_pct"] < 0.17):
        raise AssertionError(
            f"Нарушен сферический инвариант SGNE-CC: max_abs_dev_pct={stats['max_abs_dev_pct']:.6f}% >= 0.17%"
        )


def print_audit_report(title: str, stats: dict[str, Any]) -> None:
    print(f"\n=== Геометрический паспорт весов: {title} ===")
    print(
        f"  • Затронуто параметров (область): {stats['selected_region_scalars']:,} / {stats['total_text_params']:,} "
        f"({stats['selected_pct']:.3f}%)"
    )
    print(
        f"  • Побитово изменено скаляров BF16: {stats['bitwise_changed_scalars']:,} / {stats['total_text_params']:,} "
        f"({stats['bitwise_pct']:.3f}%)"
    )
    print(
        f"  • Изменено выходных столбцов    : {stats['total_edited_cols']} / {stats['total_write_cols']} "
        f"(MLP: {stats['mlp_cols']} в {stats['mlp_matrices']} сл., "
        f"DeltaNet: {stats['dnet_cols']} в {stats['dnet_matrices']} сл., "
        f"GQA: {stats['gqa_cols']})"
    )
    print(f"  • Изменено строк ворот          : {stats['changed_gate_rows']} / {stats['total_gate_rows']}")
    for gd in stats["gate_details"]:
        print(f"      - {gd}")
    gate_frob_note = (
        f"; полная с воротами ‖ΔW‖_F = {stats['total_frob_norm']:.6f}"
        if stats["changed_gate_rows"] > 0
        else ""
    )
    print(
        f"  • Норма Фробениуса столбцов     : ‖ΔW_cols‖_F = {stats['frob_norm']:.6f} "
        f"(удельная энергия ‖ΔW_cols‖_F²/N_cols = {stats['frob_sq_per_col']:.4f}{gate_frob_note})"
    )
    print(
        f"  • Относ. норма на своём носителе: ‖ΔW_cols‖_F / ‖W_cols‖_F = {stats['rel_frob']:.4f} "
        f"(по N_cols = {stats['total_edited_cols']} затронутым столбцам)"
    )
    print(f"  • Диапазон ‖w'‖₂ / ‖w‖₂ (BF16)  : [{stats['min_ratio']:.6f}, {stats['max_ratio']:.6f}]")
    print(f"  • Макс. отклонение нормы        : {stats['max_abs_dev_pct']:.4f}%")
    print(f"  • Мин. косинус поворота         : {stats['min_cos']:.6f}")
    print(f"  • Угол поворота столбца         : медиана {stats['median_angle']:.1f}°, максимум {stats['max_angle']:.1f}°")
    key_layers = [k for k in ("mlp_down_L15", "mlp_down_L19", "mlp_down_L23") if k in stats["layer_svd"]]
    if key_layers:
        svd_str = ", ".join(
            f"{k}: {stats['layer_svd'][k][1]:.4f}% ({stats['layer_svd'][k][0]} ст.)" for k in key_layers
        )
        print(f"  • Доля 1-й моды SVD (ранг-1)    : {svd_str}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Сборка и криминалистическая проверка патча Qwen3.5-0.8B-SGNE-CC."
    )
    parser.add_argument(
        "--base-model",
        "--base",
        dest="base_model",
        default=DEFAULT_BASE_REPO,
        help="Путь к локальной папке или HF repo_id базовой модели (по умолчанию Qwen/Qwen3.5-0.8B).",
    )
    parser.add_argument(
        "--delta",
        default=DEFAULT_DELTA_PATH,
        help="Путь к файлу дельты sgne_cc_dna_delta.npz (по умолчанию рядом со скриптом).",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help=(
            "Опциональный HF repo_id (например RikiyaT/Qwen3.5-0.8B-weight-space-programming) "
            "или путь к папке с model.safetensors / .npz для бокового сравнения."
        ),
    )
    parser.add_argument(
        "--out",
        default="./Qwen3.5-0.8B-SGNE-CC",
        help="Целевая папка для сохранения собранной модели.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Разрешить перезапись существующего model.safetensors в целевой папке --out.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Проверить SHA-256, нормы столбцов, SVD-ранг и ворота в памяти без записи собранной копии на диск.",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Пропустить сверку эталонных SHA-256 файла дельты и базовой модели.",
    )
    args = parser.parse_args()
    total_steps = 3 if args.verify_only else 4

    delta_path = Path(args.delta).expanduser()
    if not delta_path.is_file():
        print(f"Ошибка: файл дельты не найден: {delta_path}", file=sys.stderr)
        return 1

    actual_sha = sha256_file(delta_path)
    print(f"[SHA-256 delta] {delta_path.name}: {actual_sha}")
    if not args.skip_hash_check and actual_sha != EXPECTED_DELTA_SHA256:
        print(
            f"Ошибка контрольной суммы дельты!\n  Ожидалось: {EXPECTED_DELTA_SHA256}\n  Получено:  {actual_sha}",
            file=sys.stderr,
        )
        return 1

    try:
        base_dir = resolve_base_dir(args.base_model, total_steps=total_steps)
    except Exception as exc:
        print(f"Ошибка загрузки базовой модели: {exc}", file=sys.stderr)
        return 1

    sf_path = base_dir / "model.safetensors"
    if not sf_path.is_file():
        print(f"Ошибка: в {base_dir} не найден файл model.safetensors", file=sys.stderr)
        return 1

    if not args.skip_hash_check:
        try:
            verify_base_model_hashes(base_dir)
        except Exception as exc:
            print(f"Ошибка проверки эталонных хэшей базовой модели: {exc}", file=sys.stderr)
            return 1

    print(f"[2/{total_steps}] Сверка геометрии патча с базовой моделью {sf_path}...")
    try:
        col_patch_map, full_patch_map = load_patch_maps(delta_path, sf_path)
        sgne_stats = audit_patch_geometry(sf_path, col_patch_map, full_patch_map)
        if not args.skip_hash_check and actual_sha == EXPECTED_DELTA_SHA256:
            assert_official_sgne_invariants(sgne_stats)
    except Exception as exc:
        print(f"Ошибка валидации патча: {exc}", file=sys.stderr)
        return 1

    print_audit_report("SGNE-CC (ДНК-редактирование весов)", sgne_stats)

    if args.compare:
        try:
            comp_path = resolve_compare_path(args.compare)
            comp_cols, comp_full = load_patch_maps(comp_path, sf_path)
            comp_stats = audit_patch_geometry(sf_path, comp_cols, comp_full)
        except Exception as exc:
            print(f"Ошибка --compare: {exc}", file=sys.stderr)
            return 1
        print_audit_report(f"Сравнение ({args.compare})", comp_stats)
        print(
            f"\n  => Макс. отклонение нормы: {comp_stats['max_abs_dev_pct']:.2f}% (сравнение) против "
            f"{sgne_stats['max_abs_dev_pct']:.4f}% у SGNE-CC (уровень шага округления BF16)."
        )
        print(
            f"  => Абсолютная норма и удельная энергия столбца: "
            f"‖ΔW‖_F = {comp_stats['total_frob_norm']:.4f} (‖ΔW_cols‖_F²/N_cols = {comp_stats['frob_sq_per_col']:.4f} "
            f"на {comp_stats['total_edited_cols']} ст.) против "
            f"‖ΔW‖_F = {sgne_stats['total_frob_norm']:.4f} (‖ΔW_cols‖_F²/N_cols = {sgne_stats['frob_sq_per_col']:.4f} "
            f"на {sgne_stats['total_edited_cols']} ст.)."
        )
        if sgne_stats["rel_frob"] > 0:
            print(
                f"  => Относительная норма на собственном носителе ‖ΔW_cols‖_F/‖W_cols‖_F: "
                f"{comp_stats['rel_frob']:.4f} (знаменатель по {comp_stats['total_edited_cols']} ст.) против "
                f"{sgne_stats['rel_frob']:.4f} (знаменатель по {sgne_stats['total_edited_cols']} ст.)."
            )

    if args.verify_only:
        print(f"\n[3/{total_steps}] Режим --verify-only: все криптографические и геометрические инварианты подтверждены.")
        return 0

    out_dir = Path(args.out).expanduser()
    if out_dir.resolve() == base_dir.resolve():
        print(
            f"Ошибка: целевая папка --out ({out_dir.resolve()}) совпадает с папкой базовой модели! "
            "Укажите отдельную директорию.",
            file=sys.stderr,
        )
        return 1

    out_sf = out_dir / "model.safetensors"
    if out_sf.exists() and not args.force:
        print(
            f"Ошибка: целевой файл {out_sf} уже существует. Передайте --force для перезаписи или выберите другой --out.",
            file=sys.stderr,
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[3/4] Наложение патча и сборка тензоров BF16...")
    tensors_out: dict[str, torch.Tensor] = {}
    with safe_open(str(sf_path), framework="pt", device="cpu") as f:
        sf_meta = f.metadata() or {"format": "pt"}
        for key in f.keys():
            w = f.get_tensor(key)
            if key in col_patch_map:
                cols_np, vals_bf16 = col_patch_map[key]
                idx = torch.from_numpy(cols_np).long()
                w = w.to(dtype=torch.bfloat16).clone()
                w[:, idx] = vals_bf16
            elif key in full_patch_map:
                w = full_patch_map[key].clone()
            tensors_out[key] = w

    print(f"[4/4] Сохранение собранной модели в {out_dir}...")
    for item in base_dir.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors") and not item.name.startswith("."):
            shutil.copy2(item, out_dir / item.name)

    save_file(tensors_out, str(out_sf), metadata=sf_meta)
    print(f"Готово: {out_sf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
