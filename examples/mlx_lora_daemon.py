#!/usr/bin/env python3 -u
"""
ANE Real-Time Fine-Tuning Daemon
Based on Ex0byt's pipeline architecture:
  1. MLX inference with live LoRA adapter (GPU/unified memory)
  2. SSE streaming responses back to client
  3. Background ANE training on user+assistant pairs (Neural Engine @ ~40 mW)
  4. LoRA adapter hot-swapped in-memory for next inference

Port 8766 — daemon accepts POST /chat with JSON {messages: [...]}

Usage:
    # Build the bridge first
    cd bridge && make && cd ..

    # Run daemon (defaults to Qwen2.5-3B-Instruct-4bit)
    python examples/mlx_lora_daemon.py

    # Or with custom settings
    ANE_MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit \
    ANE_PORT=8800 \
    python examples/mlx_lora_daemon.py

    # Test it
    curl -s -X POST http://localhost:8766/chat \
      -H "Content-Type: application/json" \
      -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":false}'
"""
import os
import sys
import json
import time
import ctypes
import struct
import threading
import queue
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from mlx_lm import load, generate
from mlx_lm.tuner.lora import LoRALinear
import numpy as np

# ---------- Config ----------
MODEL_NAME = os.environ.get("ANE_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
LORA_RANK = int(os.environ.get("ANE_LORA_RANK", "8"))
LORA_LAYERS = int(os.environ.get("ANE_LORA_LAYERS", "4"))  # last N layers
PORT = int(os.environ.get("ANE_PORT", "8766"))
MAX_TOKENS = int(os.environ.get("ANE_MAX_TOKENS", "512"))

# Resolve paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
BRIDGE_PATH = os.environ.get("ANE_BRIDGE_PATH",
    str(REPO_ROOT / "bridge" / "libane_bridge.dylib"))
DATA_DIR = os.environ.get("ANE_DATA_DIR",
    str(REPO_ROOT / "data" / "training"))
ADAPTER_DIR = os.environ.get("ANE_ADAPTER_DIR",
    str(REPO_ROOT / "data" / "adapter"))

# Ensure ane_lora_kernels is importable from repo root
sys.path.insert(0, str(REPO_ROOT))

# ---------- Training queue ----------
training_queue = queue.Queue()
training_pairs = []  # persistent conversation pairs

# ---------- ANE Bridge (ctypes) ----------
class ANEBridge:
    """Python wrapper around libane_bridge.dylib for ANE kernel dispatch."""

    def __init__(self, dylib_path):
        self.lib = None
        self.available = False
        try:
            self.lib = ctypes.CDLL(dylib_path)
            self._setup_signatures()
            rc = self.lib.ane_bridge_init()
            if rc == 0:
                self.available = True
                print(f"[ANE] Bridge initialized from {dylib_path}")
            else:
                print(f"[ANE] Bridge init failed (rc={rc}) -- ANE training disabled")
        except OSError as e:
            print(f"[ANE] Cannot load bridge: {e} -- ANE training disabled")

    def _setup_signatures(self):
        self.lib.ane_bridge_init.restype = ctypes.c_int
        self.lib.ane_bridge_compile.restype = ctypes.c_void_p
        self.lib.ane_bridge_compile.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.ane_bridge_compile_multi_weights.restype = ctypes.c_void_p
        self.lib.ane_bridge_eval.restype = ctypes.c_bool
        self.lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]
        self.lib.ane_bridge_write_input.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.ane_bridge_read_output.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.ane_bridge_free.argtypes = [ctypes.c_void_p]
        self.lib.ane_bridge_get_compile_count.restype = ctypes.c_int
        self.lib.ane_bridge_reset_compile_count.restype = None
        self.lib.ane_bridge_build_weight_blob.restype = ctypes.POINTER(ctypes.c_uint8)
        self.lib.ane_bridge_build_weight_blob.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t)]

    @property
    def compile_count(self):
        if not self.available:
            return 0
        return self.lib.ane_bridge_get_compile_count()


ane = ANEBridge(BRIDGE_PATH)

# ---------- Model loading ----------
print(f"[MLX] Loading model: {MODEL_NAME}")
model, tokenizer = load(MODEL_NAME)
print(f"[MLX] Model loaded. Applying LoRA (rank={LORA_RANK}) to last {LORA_LAYERS} layers...")

# Apply LoRA to attention layers
def apply_lora(model, rank, num_layers):
    """Apply LoRA adapters to the last N transformer layers."""
    layers = model.model.layers
    total = len(layers)
    start = max(0, total - num_layers)

    for i in range(start, total):
        layer = layers[i]
        attn = layer.self_attn
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if hasattr(attn, proj_name):
                orig = getattr(attn, proj_name)
                if not isinstance(orig, LoRALinear):
                    lora = LoRALinear.from_base(orig, r=rank)
                    setattr(attn, proj_name, lora)

    lora_params = sum(
        p.size for k, p in mlx.utils.tree_flatten(model.trainable_parameters())
    )
    total_params = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
    print(f"[LoRA] {lora_params:,} trainable / {total_params:,} total "
          f"({100*lora_params/total_params:.2f}%)")
    return model

model = apply_lora(model, LORA_RANK, LORA_LAYERS)

# Phase 2b: Replace LoRA layers with ANE-backed versions (conv-based)
ane_kernels = None
if ane.available:
    try:
        from ane_lora_kernels import (
            ANELoRAKernels, set_ane_kernels, replace_lora_with_ane)

        ane_kernels = ANELoRAKernels(BRIDGE_PATH)
        err = ane_kernels.verify_conv()
        if err > 0.1:
            raise RuntimeError(f"ANE conv verification failed: max_error={err:.4f}")
        print(f"[ANE] Conv verification passed (max_error={err:.6f})")

        set_ane_kernels(ane_kernels)
        replaced = replace_lora_with_ane(model)
        print(f"[ANE] {replaced} LoRA layers -> ANE fused gradient dispatch active")
    except Exception as e:
        print(f"[ANE] Layer replacement failed ({e}), using MLX GPU fallback")
        ane_kernels = None

# Load saved adapter if exists
if os.path.exists(ADAPTER_DIR):
    try:
        model.load_weights(ADAPTER_DIR, strict=False)
        print(f"[LoRA] Loaded saved adapter from {ADAPTER_DIR}")
    except Exception as e:
        print(f"[LoRA] Could not load adapter: {e}")

# ---------- Background training thread ----------
def ane_training_worker():
    os.makedirs(DATA_DIR, exist_ok=True)

    while True:
        pair = training_queue.get()
        if pair is None:
            break

        user_text, assistant_text = pair
        training_pairs.append(pair)

        pair_file = os.path.join(DATA_DIR, f"pair_{int(time.time()*1000)}.json")
        with open(pair_file, "w") as f:
            json.dump({"user": user_text, "assistant": assistant_text}, f)

        try:
            loss = _finetune_step(user_text, assistant_text)
        except Exception as e:
            import traceback
            print(f"[FT] Error: {e}")
            traceback.print_exc()

        try:
            os.makedirs(ADAPTER_DIR, exist_ok=True)
            mx.savez(
                os.path.join(ADAPTER_DIR, "adapters.npz"),
                **dict(mlx.utils.tree_flatten(model.trainable_parameters()))
            )
        except Exception as e:
            print(f"[LoRA] Save error: {e}")


def _finetune_step(user_text, assistant_text):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text},
         {"role": "assistant", "content": assistant_text}],
        tokenize=False, add_generation_prompt=False
    )
    tokens = mx.array(tokenizer.encode(prompt))

    def loss_fn(model, tokens):
        logits = model(tokens[None, :-1])
        targets = tokens[1:]
        return nn.losses.cross_entropy(logits.squeeze(0), targets, reduction="mean")

    pre_dispatches = 0
    if ane_kernels:
        pre_dispatches = ane_kernels.total_dispatches

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, tokens)
    mx.eval(loss, grads)

    lr = 1e-4
    trainable_flat = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
    grad_flat = dict(mlx.utils.tree_flatten(grads))
    updates = []
    for k in trainable_flat:
        if k in grad_flat:
            updates.append((k, trainable_flat[k] - lr * grad_flat[k]))
    if updates:
        model.load_weights(updates, strict=False)
        mx.eval(model.parameters())

    loss_val = float(loss)

    if ane_kernels:
        from ane_lora_kernels import get_dispatch_stats
        stats = get_dispatch_stats()
        new_dispatches = ane_kernels.total_dispatches - pre_dispatches
        engine = "ANE" if new_dispatches > 0 else "MLX"
        print(f"[{engine}-FT] loss={loss_val:.4f} | "
              f"ane_dispatches={new_dispatches} "
              f"total_ane={stats.get('total_ane_dispatches', 0)} "
              f"total_steps={stats.get('total_ane_steps', 0)}")
    else:
        print(f"[MLX-FT] loss={loss_val:.4f}")

    return loss_val


training_thread = threading.Thread(target=ane_training_worker, daemon=True)
training_thread.start()

# ---------- HTTP Server ----------
class DaemonHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/chat":
            self._handle_chat()
        elif self.path == "/status":
            self._handle_status()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/status":
            self._handle_status()
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_error(404)

    def _handle_status(self):
        ane_stats = {}
        if ane_kernels is not None:
            from ane_lora_kernels import get_dispatch_stats
            ane_stats = get_dispatch_stats()

        status = {
            "model": MODEL_NAME,
            "lora_rank": LORA_RANK,
            "lora_layers": LORA_LAYERS,
            "ane_available": ane.available,
            "ane_phase": "fused-kernel" if ane_kernels is not None else ("fallback" if not ane.available else "init"),
            "ane_stats": ane_stats,
            "training_pairs_total": len(training_pairs),
            "training_queue_size": training_queue.qsize(),
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())

    def _handle_chat(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        messages = body.get("messages", [])
        stream = body.get("stream", True)

        if not messages:
            self.send_error(400, "No messages provided")
            return

        user_msg = None
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m["content"]
                break

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if stream:
            self._stream_response(prompt, user_msg)
        else:
            self._batch_response(prompt, user_msg)

    def _stream_response(self, prompt, user_msg):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        full_response = []
        try:
            response_text = generate(
                model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS,
                verbose=False
            )
            words = response_text.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == 0 else " " + word
                data = json.dumps({"content": chunk, "done": False})
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
                full_response.append(chunk)

            done_data = json.dumps({"content": "", "done": True})
            self.wfile.write(f"data: {done_data}\n\n".encode())
            self.wfile.flush()

            assistant_text = "".join(full_response)
            if user_msg and assistant_text:
                training_queue.put((user_msg, assistant_text))

        except Exception as e:
            error_data = json.dumps({"error": str(e), "done": True})
            self.wfile.write(f"data: {error_data}\n\n".encode())
            self.wfile.flush()

    def _batch_response(self, prompt, user_msg):
        try:
            response_text = generate(
                model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS,
                verbose=False
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "content": response_text,
                "model": MODEL_NAME,
                "ane_training": ane.available
            }).encode())

            if user_msg and response_text:
                training_queue.put((user_msg, response_text))

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def log_message(self, format, *args):
        print(f"[HTTP] {args[0]} {args[1]} {args[2]}")


# ---------- Main ----------
if __name__ == "__main__":
    ane_mode = "FUSED KERNEL (1 dispatch/module)" if ane_kernels else (
        "FALLBACK (MLX GPU)" if not ane.available else "INIT FAILED")
    print(f"\n{'='*60}")
    print(f"  ANE Real-Time Fine-Tuning Daemon")
    print(f"  Model:     {MODEL_NAME}")
    print(f"  LoRA:      rank={LORA_RANK}, layers={LORA_LAYERS}")
    print(f"  ANE:       {ane_mode}")
    if ane_kernels:
        print(f"  Method:    fused packed-IOSurface, compile-once")
    print(f"  Port:      {PORT}")
    print(f"  Adapter:   {ADAPTER_DIR}")
    print(f"{'='*60}\n")

    server = HTTPServer(("0.0.0.0", PORT), DaemonHandler)
    print(f"[DAEMON] Listening on 0.0.0.0:{PORT}")
    print(f"[DAEMON] POST /chat  -- inference + background training")
    print(f"[DAEMON] GET  /status -- daemon status")
    print(f"[DAEMON] GET  /health -- health check")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[DAEMON] Shutting down...")
        training_queue.put(None)
        training_thread.join(timeout=5)
        server.server_close()
