"""Minimal OpenAI-compatible server for latent-safety local evaluation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any

from omegaconf import OmegaConf
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    GenerationConfig,
)

from latent_safety.models.latent_memory.modeling import MemGenModel
from latent_safety.models.latent_memory.native_boundary import (
    BOUNDARY_INJECTION_MODE,
    render_generation_boundary_prompt,
)
from latent_safety.server.qwen_native_protocol import (
    QwenNativeParseResult,
    parse_qwen_native_response,
)
from latent_safety.server.mistral_native_protocol import parse_mistral_native_response
from latent_safety.server.gemma_native_protocol import parse_gemma_native_response
from redir.engine.trainer import normalize_assistant_generation_prefix


LOGGER = logging.getLogger(__name__)
_GENERATION_LOCK = threading.Lock()


def _resolve_device(device: str) -> torch.device:
    return torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")


def _payload_tools(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = payload.get("tools")
    return tools if isinstance(tools, list) and tools else None


def _payload_tool_names(payload: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for tool in _payload_tools(payload) or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _native_chat_template(tokenizer: Any) -> str:
    """Recover the model-native template replaced by ``MemGenModel``."""

    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    template = init_kwargs.get("chat_template") if isinstance(init_kwargs, dict) else None
    if not isinstance(template, str) or not template.strip():
        raise ValueError("tokenizer does not expose its model-native chat template")
    return template


def _native_protocol_name(model_family: Any) -> str:
    normalized = str(model_family or "").strip().lower()
    if normalized in {"mistral", "mistral3", "ministral3"}:
        return "mistral"
    if normalized in {"gemma", "gemma4"}:
        return "gemma"
    return "qwen"


def _decode_native_completion(
    tokenizer: Any,
    completion_ids: torch.Tensor,
    *,
    protocol: str,
) -> str:
    # Mistral and Gemma represent native tool framing with special tokens.
    # The protocol parser must see those tokens before they are stripped.
    return tokenizer.decode(
        completion_ids,
        skip_special_tokens=protocol not in {"mistral", "gemma"},
    )


def _flatten_native_content(content: Any, *, message_index: int) -> str:
    """Apply vLLM's string-format preprocessing to OpenAI content parts."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(
            f"messages[{message_index}].content must be a string, list, or null"
        )

    texts: list[str] = []
    for part_index, part in enumerate(content):
        if isinstance(part, str):
            texts.append(part)
            continue
        if not isinstance(part, dict):
            raise ValueError(
                f"messages[{message_index}].content[{part_index}] must be an object"
            )
        part_type = part.get("type")
        if part_type in {"text", "input_text", "output_text"}:
            value = part.get("text")
        elif part_type == "refusal":
            value = part.get("refusal")
        elif part_type == "thinking":
            value = part.get("thinking")
        else:
            raise ValueError(
                f"messages[{message_index}].content[{part_index}] has unsupported "
                f"part type {part_type!r}; the latent server is text-only"
            )
        if value is None and part_type in {"text", "refusal"}:
            continue
        if not isinstance(value, str):
            raise ValueError(
                f"messages[{message_index}].content[{part_index}] text must be a string"
            )
        texts.append(value)
    return "\n".join(texts)


def _normalize_native_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Mirror vLLM's OpenAI-message preprocessing needed by Qwen's template."""

    normalized: list[dict[str, Any]] = []
    for message_index, raw_message in enumerate(messages):
        if not isinstance(raw_message, dict):
            raise ValueError(f"messages[{message_index}] must be an object")
        message = dict(raw_message)
        message["content"] = _flatten_native_content(
            message.get("content"),
            message_index=message_index,
        )
        if message.get("role") == "assistant":
            reasoning = message.get("reasoning")
            message.pop("reasoning_content", None)
            if isinstance(reasoning, str):
                message["reasoning_content"] = reasoning

            raw_calls = message.get("tool_calls")
            if raw_calls is not None:
                if not isinstance(raw_calls, list):
                    raise ValueError(
                        f"messages[{message_index}].tool_calls must be a list"
                    )
                if not raw_calls:
                    message.pop("tool_calls", None)
                    normalized.append(message)
                    continue
                calls: list[dict[str, Any]] = []
                for call_index, raw_call in enumerate(raw_calls):
                    if not isinstance(raw_call, dict):
                        raise ValueError(
                            f"messages[{message_index}].tool_calls[{call_index}] "
                            "must be an object"
                        )
                    call = dict(raw_call)
                    raw_function = call.get("function")
                    if not isinstance(raw_function, dict):
                        raise ValueError(
                            f"messages[{message_index}].tool_calls[{call_index}].function "
                            "must be an object"
                        )
                    function = dict(raw_function)
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            function["arguments"] = json.loads(arguments or "{}")
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"messages[{message_index}].tool_calls[{call_index}] "
                                "has invalid JSON arguments"
                            ) from exc
                    elif arguments is None:
                        function["arguments"] = {}
                    call["function"] = function
                    calls.append(call)
                message["tool_calls"] = calls
        normalized.append(message)
    return normalized


def _apply_chat_template(
    tokenizer: Any,
    messages: list[Any],
    payload: dict[str, Any],
) -> dict[str, torch.Tensor]:
    tools = _payload_tools(payload)
    template_kwargs: dict[str, Any] = {}
    template_messages = messages
    if tools:
        template_messages = _normalize_native_messages(messages)
        template_kwargs["chat_template"] = _native_chat_template(tokenizer)
    return tokenizer.apply_chat_template(
        template_messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        **template_kwargs,
    )


def _message_request_summary(messages: list[Any]) -> tuple[list[str], list[int]]:
    roles: list[str] = []
    content_chars: list[int] = []
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            roles.append(type(raw_message).__name__)
            content_chars.append(0)
            continue
        roles.append(str(raw_message.get("role", "missing")))
        content = raw_message.get("content")
        if isinstance(content, str):
            content_chars.append(len(content))
        elif isinstance(content, list):
            content_chars.append(
                sum(
                    len(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                )
            )
        else:
            content_chars.append(0)
    return roles, content_chars


def _completion_hit_length(
    completion_ids: torch.Tensor,
    *,
    max_tokens: int,
    eos_token_id: int | list[int] | None,
) -> bool:
    if completion_ids.numel() == 0 or completion_ids.size(1) < max_tokens:
        return False
    if eos_token_id is None:
        return True
    eos_ids = (
        {int(eos_token_id)}
        if isinstance(eos_token_id, int)
        else {int(value) for value in eos_token_id}
    )
    return int(completion_ids[0, -1].item()) not in eos_ids


def _parse_native_completion(
    payload: dict[str, Any],
    raw_completion: str,
    *,
    protocol: str = "qwen",
) -> QwenNativeParseResult | None:
    tools = _payload_tools(payload)
    if not tools:
        return None
    if protocol == "mistral":
        return parse_mistral_native_response(raw_completion, tools)
    if protocol == "gemma":
        return parse_gemma_native_response(raw_completion, tools)
    return parse_qwen_native_response(raw_completion, tools)


def _log_response_audit(
    *,
    payload: dict[str, Any],
    raw_completion: str,
    parsed: QwenNativeParseResult | None,
    finish_reason: str,
) -> None:
    anomalies = []
    if parsed is not None:
        anomalies = [
            {
                "code": anomaly.code,
                "message": anomaly.message,
                "tool_index": anomaly.tool_index,
                "function_name": anomaly.function_name,
                "parameter_name": anomaly.parameter_name,
            }
            for anomaly in parsed.anomalies
        ]
    record = {
        "request_tool_names": _payload_tool_names(payload),
        "request_tools_count": len(_payload_tools(payload) or []),
        "raw_completion": raw_completion[:500],
        "raw_completion_chars": len(raw_completion),
        "parsed_tool_calls": [] if parsed is None else parsed.tool_calls,
        "reasoning_content_chars": (
            0 if parsed is None or parsed.reasoning_content is None else len(parsed.reasoning_content)
        ),
        "finish_reason": finish_reason,
        "parse_status": "not_requested" if parsed is None else parsed.parse_status,
        "parse_error": None if parsed is None else parsed.parse_error,
        "parse_anomalies": anomalies,
    }
    LOGGER.info("native_response_audit=%s", json.dumps(record, ensure_ascii=False, sort_keys=True))


@contextmanager
def _seeded_generation(payload: dict[str, Any], device: torch.device):
    """Make an optional OpenAI `seed` deterministic without leaking RNG state."""
    raw_seed = payload.get("seed")
    if raw_seed is None:
        yield None
        return

    seed = int(raw_seed)
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with _GENERATION_LOCK, torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(seed)
        yield seed


def _append_assistant_generation_prefix(
    tokenizer,
    prompt: dict[str, torch.Tensor],
    *,
    device: torch.device,
    assistant_generation_prefix: str,
) -> dict[str, torch.Tensor]:
    if not assistant_generation_prefix:
        return prompt
    prefix_ids = tokenizer(
        assistant_generation_prefix,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)
    if (
        prefix_ids.numel() > 0
        and prompt["input_ids"].size(1) >= prefix_ids.size(1)
        and torch.equal(prompt["input_ids"][:, -prefix_ids.size(1) :], prefix_ids)
    ):
        return prompt
    prefix_attention = torch.ones(
        prefix_ids.shape,
        dtype=prompt["attention_mask"].dtype,
        device=device,
    )
    return {
        **prompt,
        "input_ids": torch.cat([prompt["input_ids"], prefix_ids], dim=1),
        "attention_mask": torch.cat([prompt["attention_mask"], prefix_attention], dim=1),
    }


def _build_response(
    *,
    payload: dict[str, Any],
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    response_prefix: str,
    parsed: QwenNativeParseResult | None = None,
    hit_length: bool = False,
) -> dict[str, Any]:
    now = int(time.time())
    response_content = content if parsed is None else parsed.content
    if (
        parsed is not None
        and parsed.tool_calls
        and response_content
        and not response_content.strip()
    ):
        # vLLM's serving layer normalizes a whitespace-only prefix before a
        # successfully parsed tool call to null.
        response_content = None
    message: dict[str, Any] = {
        "role": "assistant",
        "content": response_content,
    }
    finish_reason = "stop" if parsed is None else parsed.finish_reason
    if parsed is not None:
        if parsed.reasoning_content is not None:
            # vLLM exposes this provider field as `reasoning`; LiteLLM maps it
            # to `reasoning_content` while retaining the provider-specific value.
            message["reasoning"] = parsed.reasoning_content
        if parsed.tool_calls:
            message["tool_calls"] = parsed.tool_calls
    if hit_length and finish_reason != "tool_calls":
        finish_reason = "length"
    _log_response_audit(
        payload=payload,
        raw_completion=content,
        parsed=parsed,
        finish_reason=finish_reason,
    )
    return {
        "id": f"chatcmpl-{response_prefix}-{now}",
        "object": "chat.completion",
        "created": now,
        "model": payload.get("model", f"{response_prefix}-safety-local"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class LatentChatBackend:
    def __init__(
        self,
        config_path: Path,
        model_path: str | None = None,
        device: str = "cuda",
        weaver_device: str | None = None,
        assistant_generation_prefix: str = "",
    ):
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        model_config = config["model"]
        if model_path:
            model_config["load_model_path"] = model_path
        self.device = _resolve_device(device)
        self.weaver_device = torch.device(weaver_device) if weaver_device else self.device
        self.model = MemGenModel.from_config(model_config)
        self.model.place_components(self.device, self.weaver_device)
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.native_protocol = _native_protocol_name(
            getattr(self.model.config, "model_family", "qwen")
        )
        self.assistant_generation_prefix = normalize_assistant_generation_prefix(assistant_generation_prefix)
        if self.assistant_generation_prefix:
            LOGGER.info("assistant_generation_prefix=%r", self.assistant_generation_prefix)

    @torch.no_grad()
    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 512)
        temperature = float(payload.get("temperature") or 0.0)
        do_sample = temperature > 0.0
        boundary_mode = (
            self.model.config.latent_injection_mode == BOUNDARY_INJECTION_MODE
        )
        if boundary_mode:
            tools = _payload_tools(payload)
            template_messages = (
                _normalize_native_messages(messages) if tools else messages
            )
            prompt = render_generation_boundary_prompt(
                self.tokenizer,
                template_messages,
                tools=tools,
                device=self.device,
                chat_template=_native_chat_template(self.tokenizer) if tools else None,
                assistant_generation_prefix=self.assistant_generation_prefix,
                allow_empty_generation_suffix=bool(
                    getattr(self.model.config, "allow_empty_generation_suffix", False)
                ),
            )
            latent_insertion_index = int(prompt["latent_insertion_index"])
        else:
            prompt = _apply_chat_template(self.tokenizer, messages, payload)
            prompt = {key: value.to(self.device) for key, value in prompt.items()}
            prompt = _append_assistant_generation_prefix(
                self.tokenizer,
                prompt,
                device=self.device,
                assistant_generation_prefix=self.assistant_generation_prefix,
            )
            latent_insertion_index = None
        generation_config = GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=False,
        )
        with _seeded_generation(payload, self.device) as seed:
            generated = self.model.generate_with_prompt_latent(
                prompt["input_ids"],
                prompt["attention_mask"],
                generation_config,
                latent_insertion_index=latent_insertion_index,
            )
        completion_ids = generated[:, prompt["input_ids"].size(1) :]
        content = _decode_native_completion(
            self.tokenizer,
            completion_ids[0],
            protocol=self.native_protocol,
        )
        parsed = _parse_native_completion(
            payload,
            content,
            protocol=self.native_protocol,
        )
        message_roles, message_content_chars = _message_request_summary(messages)
        LOGGER.info(
            "latent completion prompt_tokens=%d completion_tokens=%d content_chars=%d "
            "tools=%d messages=%d roles=%s message_content_chars=%s seed=%s",
            int(prompt["input_ids"].numel()),
            int(completion_ids.numel()),
            len(content),
            len(_payload_tools(payload) or []),
            len(messages),
            message_roles,
            message_content_chars,
            seed,
        )
        return _build_response(
            payload=payload,
            content=content,
            prompt_tokens=int(prompt["input_ids"].numel()),
            completion_tokens=int(completion_ids.numel()),
            response_prefix="latent",
            parsed=parsed,
            hit_length=_completion_hit_length(
                completion_ids,
                max_tokens=max_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
            ),
        )


class BaseChatBackend:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = _resolve_device(device)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
        self.native_protocol = _native_protocol_name(model_type)
        tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.native_protocol == "mistral":
            tokenizer_kwargs["fix_mistral_regex"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_loader = (
            AutoModelForImageTextToText
            if self.native_protocol == "mistral"
            else AutoModelForCausalLM
        )
        self.model = model_loader.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 512)
        temperature = float(payload.get("temperature") or 0.0)
        do_sample = temperature > 0.0
        prompt = _apply_chat_template(self.tokenizer, messages, payload)
        prompt = {key: value.to(self.device) for key, value in prompt.items()}
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = float(payload.get("top_p") or 1.0)
        with _seeded_generation(payload, self.device) as seed:
            generated = self.model.generate(**prompt, **generation_kwargs)
        completion_ids = generated[:, prompt["input_ids"].size(1) :]
        content = _decode_native_completion(
            self.tokenizer,
            completion_ids[0],
            protocol=self.native_protocol,
        )
        parsed = _parse_native_completion(
            payload,
            content,
            protocol=self.native_protocol,
        )
        message_roles, message_content_chars = _message_request_summary(messages)
        LOGGER.info(
            "base completion prompt_tokens=%d completion_tokens=%d content_chars=%d "
            "tools=%d messages=%d roles=%s message_content_chars=%s seed=%s",
            int(prompt["input_ids"].numel()),
            int(completion_ids.numel()),
            len(content),
            len(_payload_tools(payload) or []),
            len(messages),
            message_roles,
            message_content_chars,
            seed,
        )
        return _build_response(
            payload=payload,
            content=content,
            prompt_tokens=int(prompt["input_ids"].numel()),
            completion_tokens=int(completion_ids.numel()),
            response_prefix="base",
            parsed=parsed,
            hit_length=_completion_hit_length(
                completion_ids,
                max_tokens=max_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
            ),
        )


class Handler(BaseHTTPRequestHandler):
    backend: LatentChatBackend
    completion_log_dir: Path | None = None
    completion_log_lock = threading.Lock()
    completion_log_counter = 0

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._send_json({"object": "list", "data": [{"id": "latent-safety-local", "object": "model"}]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            response = self.backend.complete(payload)
            self._write_completion_log(payload, response)
            self._send_json(response)
        except Exception as exc:
            LOGGER.exception("chat completion failed")
            self._send_json({"error": {"message": str(exc), "type": "server_error"}}, status=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_completion_log(
        self, request_payload: dict[str, Any], response_payload: dict[str, Any]
    ) -> None:
        directory = type(self).completion_log_dir
        if directory is None:
            return
        with type(self).completion_log_lock:
            type(self).completion_log_counter += 1
            index = type(self).completion_log_counter
        record = {
            "timestamp": time.time(),
            "messages": request_payload.get("messages") or [],
            "kwargs": {"tools": request_payload.get("tools") or []},
            "request": request_payload,
            "response": response_payload,
        }
        path = directory / f"completion_{index:08d}.json"
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["latent", "base"], default="latent")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weaver-device", default=None)
    parser.add_argument("--assistant-generation-prefix", default="")
    parser.add_argument("--completion-log-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.backend == "latent":
        if args.config is None:
            raise ValueError("--config is required for latent backend")
        Handler.backend = LatentChatBackend(
            Path(args.config),
            args.model_path,
            args.device,
            args.weaver_device,
            args.assistant_generation_prefix,
        )
    else:
        if args.model_path is None:
            raise ValueError("--model-path is required for base backend")
        Handler.backend = BaseChatBackend(args.model_path, args.device)
    if args.completion_log_dir is not None:
        args.completion_log_dir.mkdir(parents=True, exist_ok=True)
        Handler.completion_log_dir = args.completion_log_dir.resolve()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    LOGGER.info("Serving %s safety endpoint at http://%s:%d/v1", args.backend, args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
