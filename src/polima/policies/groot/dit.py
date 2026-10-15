"""GR00T DiT rewrite for Modalix: K/V cache + 4-block graphs + host AdaLN + NCHW form.

The exported DiT is 16 two-block graphs, each re-projecting the constant backbone
features to K/V on every one of the 4 denoising steps. The rewrite (the build the
SO-101 robot runs, measured byte-identical on device to its hand-made original):

* K/V cache: two graphs (`dit_kv_00_14`, `dit_kv_16_30`) project K and V of the 16
  cross-attention blocks once per request; the cross blocks read them through
  `PrecomputedKVProcessor` (diffusers AttnProcessor2_0 with K/V supplied).
* 4 blocks per graph (`ditkv_blocks_SS_EE` N1.6, `ditkv_SS_EE` N1.7), in the LLiMa
  NCHW MLA form: activations [1,C,1,L], Linear as 1x1 Conv, heads by channel
  Split/Concat, attention as two Einsums with Softmax over axis 1.
* Host AdaLN: each block's norm1.linear(silu(temb)) depends only on the denoising
  step, so it is precomputed in FP32 (sidecars ada_embedding_k, 32 blocks x 2C) and
  fed as the graph input `ada_SS_EE` instead of temb.
* The plan runs each denoising step as one `run_bound_chain` over persistent
  bindings (`bound_plan`).

The torch parts run in the family's GR00T environment; `bound_plan` and
`describe_persistent` are plain Python for the packing interpreter.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

PAIRS = 16
KV_GRAPHS = 2
BLOCKS_PER_GRAPH = 4


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def names(family: str) -> dict:
    """Graph and binding names of the rewrite, per model family (bundle-compatible)."""
    prefix = "ditkv_blocks" if family == "n1.6" else "ditkv"
    per = PAIRS // KV_GRAPHS
    kv = [f"dit_kv_{2 * g * per:02d}_{2 * (g + 1) * per - 2:02d}" for g in range(KV_GRAPHS)]
    blocks = [f"{prefix}_{s:02d}_{s + 3:02d}" for s in range(0, 2 * PAIRS, BLOCKS_PER_GRAPH)]
    binds = (["hidden", "temb", "backbone_features", "text_additive_mask", "image_additive_mask"]
             if family == "n1.6" else ["hidden", "temb", "backbone", "mask_even", "mask_odd"])
    return {"kv": kv, "blocks": blocks, "binds": binds,
            "pairs": [f"dit_blocks_{i:02d}_{i + 1:02d}" if family == "n1.6" else f"dit_{i:02d}_{i + 1:02d}"
                      for i in range(0, 2 * PAIRS, 2)]}


# ------------------------------------------------------------------ modules (torch)


def _modules():
    import torch
    import torch.nn.functional as F
    from torch import nn

    class PrecomputedKVProcessor:
        """diffusers 0.35 AttnProcessor2_0.__call__ with (key, value) = (to_k(e), to_v(e))
        supplied; every other operation and its order unchanged."""

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                     temb=None, *args, **kwargs):
            key, value = encoder_hidden_states
            batch, sequence, _ = key.shape
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence, batch)
                attention_mask = attention_mask.view(batch, attn.heads, -1, attention_mask.shape[-1])
            query = attn.to_q(hidden_states)
            head = key.shape[-1] // attn.heads
            query, key, value = (t.view(batch, -1, attn.heads, head).transpose(1, 2) for t in (query, key, value))
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask,
                                                 dropout_p=0.0, is_causal=False)
            out = out.transpose(1, 2).reshape(batch, -1, attn.heads * head).to(query.dtype)
            return attn.to_out[1](attn.to_out[0](out))

    def check_attention(attn):
        if type(attn.processor).__name__ != "AttnProcessor2_0" or attn.spatial_norm is not None \
                or attn.group_norm is not None or attn.norm_q is not None or attn.norm_k is not None \
                or attn.residual_connection or attn.rescale_output_factor != 1.0:
            raise ValueError("DiT attention uses a branch the K/V rewrite does not reproduce")

    class CrossKV(nn.Module):
        """K, V of the given cross-attention blocks (outputs k_BB, v_BB). N1.7 graphs were
        exported from cloned blocks (norms in MatmulNorm form), N1.6 from the attention
        modules; the module paths are the ONNX initializer names, so both are kept."""

        def __init__(self, dit, blocks, clone=None):
            super().__init__()
            if clone is None:
                self.attns = nn.ModuleList([dit.transformer_blocks[b].attn1 for b in blocks])
            else:
                self.blocks = nn.ModuleList([clone(dit.transformer_blocks[b]) for b in blocks])
            for attn in self._attentions():
                check_attention(attn)

        def _attentions(self):
            return list(self.attns) if hasattr(self, "attns") else [b.attn1 for b in self.blocks]

        def forward(self, backbone):
            outputs = []
            for attn in self._attentions():
                source = attn.norm_encoder_hidden_states(backbone) if attn.norm_cross else backbone
                outputs += [attn.to_k(source), attn.to_v(source)]
            return tuple(outputs)

    def conv(linear):
        layer = nn.Conv2d(linear.in_features, linear.out_features, 1, bias=linear.bias is not None)
        layer.weight.data = linear.weight.data.clone()[:, :, None, None]
        if linear.bias is not None:
            layer.bias.data = linear.bias.data.clone()
        return layer

    class NchwAdaBlock(nn.Module):
        """One diffusers BasicTransformerBlock (ada_norm) in LLiMa NCHW form, taking the
        precomputed (scale, shift) of its AdaLayerNorm instead of temb."""

        def __init__(self, block, cross):
            super().__init__()
            a = block.attn1
            self.cross, self.H, self.D, self.C = cross, a.heads, a.inner_dim // a.heads, a.inner_dim
            self.scale = float(a.scale)
            if block.norm_type != "ada_norm" or block.pos_embed is not None or block.norm1.norm.weight is not None:
                raise ValueError("unsupported DiT block form")
            self.eps1 = block.norm1.norm.eps
            self.q = conv(a.to_q)
            if not cross:
                self.k, self.v = conv(a.to_k), conv(a.to_v)
            self.o = conv(a.to_out[0])
            n3 = block.norm3
            self.eps3 = n3.eps
            self.n3w = None if n3.weight is None else nn.Parameter(n3.weight.data.clone()[None, :, None, None])
            self.n3b = None if n3.bias is None else nn.Parameter(n3.bias.data.clone()[None, :, None, None])
            ff = block.ff.net
            if ff[0].approximate != "tanh":
                raise ValueError("DiT feed-forward is not tanh-GELU")
            self.f1, self.f2 = conv(ff[0].proj), conv(ff[2])

        @staticmethod
        def ln(x, eps):  # LayerNorm over the channel axis of [1,C,1,L]
            d = x - x.mean(dim=1, keepdim=True)
            return d / torch.sqrt((d * d).mean(dim=1, keepdim=True) + eps)

        def heads(self, t):  # [1,H*D,1,L] -> [1,D,H,L]
            return torch.cat(torch.split(t, self.D, dim=1), dim=2)

        def forward(self, x, ada, k=None, v=None, mask=None):
            scale, shift = ada[:, :self.C], ada[:, self.C:]
            h = self.ln(x, self.eps1) * (1 + scale) + shift
            q = self.heads(self.q(h)) * self.scale
            if not self.cross:
                k, v = self.heads(self.k(h)), self.heads(self.v(h))
            s = torch.einsum("nchw,nchq->nqhw", q, k)
            if mask is not None:
                s = s + mask
            o = torch.einsum("nchw,nqhc->nqhw", torch.softmax(s, dim=1), v)
            x = self.o(torch.cat(torch.split(o, 1, dim=2), dim=1)) + x
            hn = self.ln(x, self.eps3)
            if self.n3w is not None:
                hn = hn * self.n3w
            if self.n3b is not None:
                hn = hn + self.n3b
            return self.f2(F.gelu(self.f1(hn), approximate="tanh")) + x

    class NchwAda(nn.Module):
        """Blocks start..start+3: hidden [1,C,1,L]; K/V [1,C,1,Lk] per cross block; masks
        [1,Lk,1,1] (additive for N1.6, 0/1 for N1.7 as its pair graphs took them);
        ada [1,4*2C,1,1]."""

        def __init__(self, dit, start, count, binary_masks):
            super().__init__()
            self.blocks = nn.ModuleList([NchwAdaBlock(dit.transformer_blocks[b], (b - start) % 2 == 0)
                                         for b in range(start, start + count)])
            self.binary_masks = binary_masks
            self.D, self.A = self.blocks[0].D, 2 * self.blocks[0].C

        def forward(self, hidden, *args):
            ada, rest = args[-1], args[:-1]
            crosses = len(self.blocks) // 2
            kv = [torch.cat(torch.split(t, self.D, dim=1), dim=2) for t in rest[:2 * crosses]]
            masks = [(1.0 - m) * -10000.0 for m in rest[2 * crosses:]] if self.binary_masks \
                else rest[2 * crosses:]
            for i, block in enumerate(self.blocks):
                a = ada[:, i * self.A:(i + 1) * self.A]
                if i % 2 == 0:
                    hidden = block(hidden, a, kv[i], kv[i + 1], masks[i // 2])
                else:
                    hidden = block(hidden, a)
            return hidden

    return CrossKV, NchwAda


def _export_onnx(module, inputs, path: Path, input_names, output_names) -> None:
    import torch

    module = module.cpu().float().eval().requires_grad_(False)
    with torch.inference_mode():
        torch.onnx.export(module, tuple(inputs), str(path), input_names=list(input_names),
                          output_names=list(output_names), opset_version=17,
                          do_constant_folding=True, dynamo=False)


# ------------------------------------------------------------------- export (torch)


def _load_n16_dit(checkpoint: Path):
    """The checkpoint's action head only (the Eagle backbone needs CUDA to build)."""
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead, Gr00tN1d6Config
    from safetensors.torch import load_file

    head = Gr00tN1d6ActionHead(Gr00tN1d6Config.from_pretrained(str(checkpoint)))
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for shard in sorted({f for k, f in index.items() if k.startswith("action_head.")}):
        state.update({k[len("action_head."):]: v for k, v in load_file(str(checkpoint / shard)).items()
                      if k.startswith("action_head.")})
    missing, unexpected = head.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("position_ids")]
    if missing or unexpected:
        raise ValueError(f"action head load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return head.float().eval().model


def export_variant(family: str, checkpoint: Path, export_dir: Path, out: Path, *, source: Path) -> None:
    """Write out/{onnx,calibration,sidecars}/ + manifest.json for the rewritten DiT."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(8)
    torch.manual_seed(0)
    if out.exists():
        import shutil

        shutil.rmtree(out)  # a stage restarts from scratch
    for sub in ("onnx", "calibration", "sidecars"):
        (out / sub).mkdir(parents=True)
    CrossKV, NchwAda = _modules()
    n = names(family)
    if family == "n1.6":
        dit, clone = _load_n16_dit(checkpoint), None
        cal = {}
        for p in range(PAIRS):  # materialized once: npz members are re-read on every access
            with np.load(export_dir / "calibration" / f"{n['pairs'][p]}.npz") as z:
                cal[p] = {k: z[k] for k in (z.files if p < 2 else ("hidden", "temb"))}
        rows = len(cal[0]["hidden"])
        pick = np.unique(np.linspace(0, rows - 1, 16).astype(int))  # W8A16: rows do not reach the weights
        backbone = cal[0]["backbone_features"]
        masks = [cal[0]["additive_mask"], cal[1]["additive_mask"]]
        steps = [np.fromfile(export_dir / "constants" / f"timestep_embedding_{k}", np.float32) for k in range(4)]
        kv_input, mask_names = "backbone_features", ["text_additive_mask", "image_additive_mask"]
        kv_trace = torch.from_numpy(backbone[0])
        kv_calibration = backbone[pick].astype(np.float32)
        call_rows = [{"backbone": torch.from_numpy(backbone[r]), "masks": [torch.from_numpy(m[r]) for m in masks],
                      "temb": {s: torch.from_numpy(cal[s // 2]["temb"][r]) for s in range(0, 32, 4)},
                      "hidden": {s: torch.from_numpy(cal[s // 2]["hidden"][r]) for s in range(0, 32, 4)}}
                     for r in pick]
        for row in call_rows:  # the host sidecar must be what the device graph would have seen
            if any(sum(np.array_equal(t.numpy().ravel(), c) for c in steps) != 1 for t in row["temb"].values()):
                raise ValueError("calibration temb is not one of the four denoising-step embeddings")
    else:
        from polima.policies.groot.n17 import clone, load_policy

        dit = load_policy(checkpoint, source).model.action_head.model
        cal0 = dict(np.load(export_dir / "calibration" / f"{n['pairs'][0]}.npz"))
        kv_input, mask_names = "backbone", ["mask_even", "mask_odd"]
        kv_trace = torch.from_numpy(cal0["backbone"][0][:, 0])
        kv_calibration = cal0["backbone"][:, :, 0]
        ref = np.load(export_dir / "reference" / "reference.npz")
        f = lambda key: torch.from_numpy(ref[key].astype(np.float32))
        call_rows = [{"backbone": f(f"dit_block_0.{c}.kwargs.encoder_hidden_states"),
                      "temb": {s: f(f"dit_block_0.{c}.kwargs.temb") for s in range(0, 32, 4)},
                      "masks": [f(f"dit_block_{2 * i}.{c}.kwargs.encoder_attention_mask") for i in range(2)],
                      "hidden": {s: f(f"dit_block_{s}.{c}.args.0") for s in range(0, 32, 4)}} for c in range(4)]
        steps = [row["temb"][0].numpy().ravel() for row in call_rows]
    graphs = []
    # ---- K/V prelude: once per request
    kv_values = [{} for _ in call_rows]
    for g, name in enumerate(n["kv"]):
        blocks = [2 * p for p in range(g * PAIRS // KV_GRAPHS, (g + 1) * PAIRS // KV_GRAPHS)]
        module = CrossKV(dit, blocks, clone).float().eval()
        outputs = [f"{t}_{b:02d}" for b in blocks for t in ("k", "v")]
        with torch.inference_mode():
            for values, row in zip(kv_values, call_rows):
                values.update(zip(outputs, module(row["backbone"])))
        _export_onnx(module, [kv_trace], out / "onnx" / f"{name}.onnx", [kv_input], outputs)
        np.savez(out / "calibration" / f"{name}.npz", **{kv_input: kv_calibration.astype(np.float32)})
        graphs.append({"name": name, "kind": "kv", "layout": "NHWC", "blocks": blocks,
                       "inputs": [kv_input], "outputs": outputs})
    # ---- host AdaLN tables: per step k, norm1.linear(silu(temb_k)) of all 32 blocks, FP32
    linears = [b.norm1.linear for b in dit.transformer_blocks]
    @torch.no_grad()  # also called below, outside any inference block
    def ada_of(temb):
        temb = temb.reshape(1, -1)
        return torch.cat([F.linear(F.silu(temb), lin.weight.float(), lin.bias.float()) for lin in linears], 1)

    for k, temb in enumerate(steps):
        ada_of(torch.from_numpy(np.ascontiguousarray(temb))).numpy().astype(np.float32).ravel().tofile(
            out / "sidecars" / f"ada_embedding_{k}")
    # ---- 4-block NCHW graphs
    width = 2 * dit.transformer_blocks[0].attn1.inner_dim
    nchw = lambda t: t.reshape(1, *t.shape[-2:]).permute(0, 2, 1)[:, :, None, :]
    for start, name in zip(range(0, 2 * PAIRS, BLOCKS_PER_GRAPH), n["blocks"]):
        module = NchwAda(dit, start, BLOCKS_PER_GRAPH, binary_masks=family == "n1.7").float().eval()
        pairs = [start // 2, start // 2 + 1]
        kv_names = [f"{t}_{2 * p:02d}" for p in pairs for t in ("k", "v")]
        ada_name = f"ada_{start:02d}_{start + 3:02d}"
        inputs = ["hidden", *kv_names, *[mask_names[p % 2] for p in pairs], ada_name]
        rows = []
        for values, row in zip(kv_values, call_rows):
            ada = ada_of(row["temb"][start])[:, start * width:(start + 4) * width]
            rows.append([nchw(row["hidden"][start]), *[nchw(values[k]) for k in kv_names],
                         *[row["masks"][p % 2].reshape(1, -1, 1, 1) for p in pairs], ada[:, :, None, None]])
        _export_onnx(module, rows[0], out / "onnx" / f"{name}.onnx", inputs, ["hidden_out" if family == "n1.6" else "output"])
        np.savez(out / "calibration" / f"{name}.npz",
                 **{k: torch.stack([r[i] for r in rows]).numpy() for i, k in enumerate(inputs)})
        graphs.append({"name": name, "kind": "blocks", "layout": "NCHW", "blocks": list(range(start, start + 4)),
                       "inputs": inputs, "outputs": ["hidden_out" if family == "n1.6" else "output"],
                       "ada_input": ada_name, "ada_offset": start * width, "ada_count": 4 * width})
    for graph in graphs:
        graph["sha256"] = _sha(out / "onnx" / f"{graph['name']}.onnx")
    manifest = {"family": family, "checkpoint": str(checkpoint), "graphs": graphs, "rows": len(call_rows)}
    manifest["check"] = check_chain(family, export_dir, out, graphs, steps)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not manifest["check"]["ok"]:
        raise ValueError(f"rewritten DiT differs from the exported pair graphs: {manifest['check']}")


def check_chain(family: str, export_dir: Path, out: Path, graphs: list[dict], steps) -> dict:
    """FP32 onnxruntime: the 16 exported pair graphs vs K/V + 4-block NCHW graphs on the
    first calibration row of each denoising step (relative max error)."""
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    session = lambda p: ort.InferenceSession(str(p), options, providers=["CPUExecutionProvider"])
    n = names(family)
    pairs = [session(export_dir / "onnx" / f"{p}.onnx") for p in n["pairs"]]
    new = [(g, session(out / "onnx" / f"{g['name']}.onnx")) for g in graphs]
    tables = [np.fromfile(out / "sidecars" / f"ada_embedding_{k}", np.float32)[None] for k in range(4)]
    cal = [dict(np.load(export_dir / "calibration" / f"{p}.npz")) for p in n["pairs"][:2]]
    worst, rows = 0.0, []
    seen = set()
    for r in range(len(cal[0]["hidden"])):
        temb = np.asarray(cal[0]["temb"][r], np.float32).ravel()
        k = next((k for k, t in enumerate(steps) if np.array_equal(temb, t)), None)
        if k is None or k in seen:
            continue
        seen.add(k)
        feeds = {key: cal[0][key][r] for key in cal[0]}
        if family == "n1.6":
            masks = [cal[0]["additive_mask"][r], cal[1]["additive_mask"][r]]
            backbone, hidden = feeds["backbone_features"], feeds["hidden"]
        else:
            masks = [cal[0]["attention_mask"][r], cal[1]["attention_mask"][r]]
            backbone, hidden = feeds["backbone"], feeds["hidden"]
        h = hidden
        for p, s in enumerate(pairs):
            mask_key = "additive_mask" if family == "n1.6" else "attention_mask"
            h = s.run(None, {**feeds, "hidden": h, mask_key: masks[p % 2]})[0]
        values = {}
        x = np.ascontiguousarray(hidden.reshape(1, *hidden.shape[-2:]).transpose(0, 2, 1)[:, :, None, :])
        for g, s in new:
            if g["kind"] == "kv":
                values.update(zip(g["outputs"], s.run(None, {g["inputs"][0]: backbone.reshape(1, *backbone.shape[-2:])})))
                continue
            feed = {}
            for name in g["inputs"]:
                if name == "hidden":
                    feed[name] = x
                elif name.startswith(("k_", "v_")):
                    feed[name] = np.ascontiguousarray(values[name].transpose(0, 2, 1)[:, :, None, :])
                elif name == g["ada_input"]:
                    feed[name] = tables[k][:, g["ada_offset"]:g["ada_offset"] + g["ada_count"], None, None]
                else:
                    feed[name] = masks[n["binds"].index(name) - 3].reshape(1, -1, 1, 1).astype(np.float32)
            x = s.run(None, feed)[0]
        got = x[:, :, 0, :].transpose(0, 2, 1).reshape(h.shape)
        error = float(np.abs(got - h).max() / np.abs(h).max())
        rows.append({"step": k, "row": r, "relative_max_abs": error})
        worst = max(worst, error)
        if len(seen) == 4:
            break
    return {"ok": bool(rows) and worst < 1e-3, "worst_relative_max_abs": worst, "rows": rows}


# ------------------------------------------------------------- plan rewrite (plain)


def bound_plan(plan: dict, family: str, manifest: dict) -> dict:
    """Base plan (16 per-pair `run_elf` per step) -> one `run_bound_chain` per step with the
    K/V prelude, the 4-block graphs and per-step host AdaLN packs."""
    n = names(family)
    result, steps, i, seen = copy.deepcopy(plan), [], 0, []
    source = plan["steps"]
    while i < len(source):
        step = source[i]
        if family == "n1.6" and step["op"] == "run_elf" and step["args"].get("graph") == n["pairs"][0]:
            block = source[i:i + PAIRS]
            hidden, temb = step["args"]["in"][0], step["args"]["in"][1]
            for j, (item, name) in enumerate(zip(block, n["pairs"], strict=True)):
                want = [hidden if j == 0 else block[j - 1]["out"], temb, "backbone_features",
                        "text_mask" if j % 2 == 0 else "image_mask"]
                if item["op"] != "run_elf" or item["args"] != {"graph": name, "in": want}:
                    raise ValueError(f"unexpected DiT call at plan step {i + j}: {item}")
            host = [hidden, temb, "backbone_features", "text_mask", "image_mask"]
            pre, i = [], i + PAIRS
        elif family == "n1.7" and step["op"] == "pack" and i + 1 < len(source) \
                and source[i + 1]["op"] == "run_elf" and source[i + 1]["args"].get("graph") == n["pairs"][0]:
            block = source[i:i + 2 * PAIRS]
            masks = [block[2 * j]["out"] for j in range(PAIRS)]
            first = block[1]["args"]["in"]  # hidden first; the rest in export or compiler order
            hidden, temb = first[0], next(x for x in first if x.startswith("temb_"))
            backbone = next(x for x in first[1:] if x not in (temb, masks[0]))
            for j, name in enumerate(n["pairs"]):
                pack, run = block[2 * j], block[2 * j + 1]
                carry = hidden if j == 0 else block[2 * j - 1]["out"]
                ins = run["args"].get("in", [])
                if pack["op"] != "pack" or run["op"] != "run_elf" or run["args"].get("graph") != name \
                        or ins[:1] != [carry] or sorted(ins) != sorted([carry, temb, backbone, masks[j]]) \
                        or pack["args"]["parts"][0]["src"] != masks[j]:
                    raise ValueError(f"unexpected N1.7 DiT call at plan step {i + 2 * j}")
            host = [hidden, temb, backbone, masks[0], masks[1]]
            pre, i = [copy.deepcopy(block[0]), copy.deepcopy(block[2])], i + 2 * PAIRS
        else:
            steps.append(copy.deepcopy(step))
            i += 1
            continue
        k = int(host[1].rsplit("_", 1)[1])  # this pass's temb_k buffer
        seen.append(k)
        ada_bufs, ada_names = [], []
        for g in manifest["graphs"]:
            if g["kind"] != "blocks":
                continue
            buf = f"{g['ada_input']}_s{k}"
            result["buffers"][buf] = g["ada_count"]
            steps.append({"op": "pack", "out": buf, "args": {"parts": [
                {"count": g["ada_count"], "dst_offset": 0, "src_offset": g["ada_offset"], "sidecar": True,
                 "src": f"ada_embedding_{k}"}]}})
            ada_bufs.append(buf)
            ada_names.append(g["ada_input"])
        steps[len(steps) - len(ada_bufs):len(steps) - len(ada_bufs)] = pre  # mask packs, then AdaLN packs
        binds = n["binds"]
        steps.append({"op": "run_bound_chain", "out": block[-1]["out"], "args": {
            "in": [host[0], *host[2:], *ada_bufs], "bind": [binds[0], *binds[2:], *ada_names],
            "carry": "hidden", "prelude": n["kv"], "graphs": n["blocks"]}})
    if sorted(seen) != [0, 1, 2, 3]:
        raise ValueError(f"expected four denoising passes, found {seen}")
    result["steps"] = steps
    result["sidecars"] = list(result["sidecars"]) + [f"ada_embedding_{k}" for k in range(4)]
    if family == "n1.7" and "graph_outputs" in result:  # as the persistent-DiT rewrite left them
        result["graph_outputs"] = {f"dit_blocks_{k[4:]}" if k in n["pairs"] else k: v
                                   for k, v in result["graph_outputs"].items()}
    return result


def describe_persistent(name: str, persistent: Path, onnx_outputs: list[str]) -> dict:
    """Named, ordered, audited persistent bindings of one compiled graph (bundle entry)."""
    contract = json.loads((persistent / "contract.json").read_text())
    meta = json.loads((persistent / "mpk.json").read_text())
    mla = next(x for x in meta["plugins"] if x["processor"] == "MLA")
    producers = {v["name"]: v for v in meta["plugins"]}
    order = [producers[v["name"]]["input_nodes"][0]["name"] for v in mla["input_nodes"]]
    if [v["size"] for v in mla["input_nodes"]] != [v["elements"] * 2 for v in contract["inputs"]]:
        raise ValueError(f"{name}: MLA input sizes differ from the persistent contract")
    if len(onnx_outputs) == 1:
        out_order = list(onnx_outputs)
    else:
        mapping = json.loads((persistent / "output_map.json").read_text())
        out_order = [mapping[o["name"]] for o in contract["outputs"]]
        if sorted(out_order) != sorted(onnx_outputs):
            raise ValueError(f"{name}: output map {out_order} does not cover {onnx_outputs}")
    bounds = {}
    with (persistent / "retained" / f"{name}_stage1_mla.mlc").open() as stream:
        for line in stream:  # every DMA descriptor precedes the first .tile section
            if line.startswith(".tile"):
                break
            match = re.match(r'(?:dram_to_l2|l2_to_dram) (\d+) (\d+) .*section="([^"]+)"', line)
            if match and match.group(3).startswith(("ifm.", "ofm.")):
                size, offset, section = match.groups()
                bounds[section] = max(bounds.get(section, 0), int(size) + int(offset))
    for direction, prefix in (("inputs", "ifm"), ("outputs", "ofm")):
        for item in contract[direction]:
            section = f"{prefix}.persistent.{item['binding']}.b0"
            if item["dtype"] != "bfloat16" or section not in bounds or bounds[section] > item["elements"] * 2:
                raise ValueError(f"{name}: invalid persistent binding {section}: {bounds.get(section)}")
    elf = persistent / "retained" / f"{name}_stage1_mla.elf"

    # The runtime presents persistent bindings in LEXICOGRAPHIC binding-name order
    # (input_0, input_1, input_10, ...): measured on device 2026-10-04.
    def ordered(names_, items):
        pairs = sorted(zip(names_, items, strict=True), key=lambda p: p[1]["binding"])
        return [{"name": n_, "elements": c["elements"], "binding": c["binding"]} for n_, c in pairs]

    return {"elf": elf, "sha256": _sha(elf), "dma_bounds": bounds,
            "inputs": ordered(order, contract["inputs"]), "outputs": ordered(out_order, contract["outputs"])}
