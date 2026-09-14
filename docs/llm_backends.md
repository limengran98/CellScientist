# LLM backbones and API configuration

CellScientist separates the LLM decision policy from candidate fitting and
evaluation. An LLM chooses an allow-listed candidate–address pair; the controller
validates that pair, applies the local edit and records its outcome.

## Paper backbones

| Experiment | Backbone | Configuration |
| :--- | :--- | :--- |
| Manuscript default | Gemini 3 Pro | `llm.model: "gemini-3-pro-preview"` in the BBBC base configuration |
| Open-weight reproduction of the controlled audit | Qwen2.5-0.5B-Instruct | Set the model ID exposed by your serving endpoint, for example `Qwen/Qwen2.5-0.5B-Instruct` |

The model identifier is an API deployment setting. Use the identifier your
server exposes, and keep the backbone, serving revision and decoding settings
with the resulting experiment records.

## Hosted and local endpoints

The transport sends `POST <base_url>/chat/completions` with a JSON `model`,
`messages` and `stream: false`. It reads a text response from
`choices[0].message.content`. Hosted APIs, compatibility gateways and local
servers that implement this contract use the same controller.

The base URL includes the API prefix (usually `/v1`). Configure it through
`CELLSCIENTIST_API_BASE`; configure bearer authentication through
`CELLSCIENTIST_API_KEY`. These variable names are themselves configurable in
`llm.base_url_env` and `llm.api_key_env`. A `llm.base_url` value can provide a
fixed endpoint; a populated environment variable takes precedence.

### Qwen on a local model server

With a Qwen endpoint serving `Qwen/Qwen2.5-0.5B-Instruct`:

```bash
python scripts/configure_run.py --name qwen_audit --model Qwen/Qwen2.5-0.5B-Instruct --local-api --output configs/qwen_audit.json
export CELLSCIENTIST_API_BASE=http://localhost:8000/v1
```

The paper's CUDA predictor setting is inherited. `--local-api` makes API-key
authentication optional; supply `CELLSCIENTIST_API_KEY` if the server uses it.
The LLM server's compute device and the predictor's `model.backend` are
independent settings.

### Another hosted LLM

```bash
python scripts/configure_run.py --name alternate_llm --model your-served-model-id --output configs/alternate_llm.json
export CELLSCIENTIST_API_BASE=https://your-compatible-endpoint/v1
export CELLSCIENTIST_API_KEY=your_key
```

Each named configuration receives its own protocol ID, output directory and
cache directory. The helper preserves the base dataset splits, candidate
budget, seeds and decoding values unless an explicit option changes them.

### Request parameter options

| JSON setting | Behavior |
| :--- | :--- |
| `llm.temperature: 0.0` | Send a temperature of zero (base configuration) |
| `llm.temperature: null` | Let the server apply its temperature default |
| `llm.max_tokens: 2048` | Requested output-token budget |
| `llm.max_tokens_parameter: "max_tokens"` | Send the budget as `max_tokens` (default) |
| `llm.max_tokens_parameter: "max_completion_tokens"` | Send the alternative token-budget field |
| `llm.require_credential: false` | Allow requests without an API key |
| `llm.timeout_seconds`, `llm.max_attempts` | Bound request duration and retry count |

The configuration helper exposes `--omit-temperature` and
`--token-parameter max_completion_tokens` for endpoints using these settings.
All request options are part of the saved configuration and its lock.

## Validate and run

Set `CELLSCIENTIST_DATA_ROOT` to the prepared HDF5 root, then run:

```bash
python -m cellscientist preflight --config configs/qwen_audit.json --check-llm
python -m cellscientist inspect --config configs/qwen_audit.json
python -m cellscientist freeze --config configs/qwen_audit.json --lock configs/qwen_audit.lock.json
python -m cellscientist run --config configs/qwen_audit.json --lock configs/qwen_audit.lock.json --task BBBC036_smiles --seed 11
```

To evaluate the component-routing cases directly:

```bash
python -m cellscientist routing-audit --config configs/qwen_audit.json --output audit_outputs/qwen_routing.json
```

Routing cases are self-contained; this command reads the LLM configuration
and can run independently of HDF5 preparation.

## Inspect decisions

`llm_trace.jsonl` records the model's request, response, usage when reported,
HTTP attempts, validation status and fallback events. A transport or invalid
decision follows the controller's recorded legal fallback, preserving the
candidate budget. Use these fields when comparing backbones alongside the
validation and held-out metrics. The environment's endpoint URL and API key
are resolved at request time; record the provider/deployment identity and
server version in your experiment notes.
