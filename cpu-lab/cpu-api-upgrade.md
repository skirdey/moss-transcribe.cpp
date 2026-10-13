# Use the tested CPU build through the API

The API now runs the opt48 build checked by the [24-clip benchmark](cpu-speedup.md).
That benchmark measured **1.58× overall speed**, with **37% less processing time**,
and matching outputs. This change makes that build available to the laptop client.
It does not add a new speed claim or enable failed experimental kernels.

The server checks the exact binary and weights at startup. The installer also
checks the frozen source tree and all three shared libraries. Requests use
16 CPU threads, the tested optimization setting and the existing bounded retry
policy. The laptop client checks the runtime identity on returned results.
Completed transcript caches stay valid; the model and decoding policy are unchanged.

## Checks

The [numeric rollout report](cpu-api-upgrade-v1.json) keeps the checks and small
API timing observations. Three uploads from a laptop match the retained benchmark
answers, parsed speaker/timestamp segments, generation counts and EOS. The real
client's audio conversion and upload route also passes. A second service restart
returns the same runtime and a matching answer. The service is active and enabled,
and user lingering is enabled for boot startup. A whole-machine reboot was not tested.

The first verification attempt used a local Python environment without NumPy.
The raw uploads passed, but the real-client check stopped and the old service and
client were restored. The complete retry used the project's existing environment.
A separate check rejects the old binary before changing any settings or writing
client credentials. Both attempts remain in the report.

The original executable and libraries are kept at their existing paths. The new
release lives in a separate directory; server code, settings and the service unit
have a private backup. Automatic ingestion is restored after verification.

## Install this validated release

This installer requires the exact prebuilt binary and libraries from the linked
benchmark, and the matching source tree and weights. A fresh build may have a
different hash and must be benchmarked and pinned separately before installation.
The measured release uses CPUs 0 through 15 and was built for the CPU described in
the benchmark; test compatibility and performance on other hardware first.

Supply your own settings below. These are neutral placeholders, not connection
information for a deployed service. Pause callers and wait until the worker is
idle before upgrading. Store the generated client file outside Git.

```sh
python3 cpu-lab/setup_moss_api.py \
  --host "$SSH_TARGET" \
  --source "$VALIDATED_SOURCE" \
  --build "$VALIDATED_BUILD" \
  --model "$MODEL_FILE" \
  --service-root "$SERVICE_ROOT" \
  --bind-address "$PRIVATE_BIND_ADDRESS" \
  --api-url "$PRIVATE_API_URL" \
  --client-dir "$PRIVATE_CLIENT_DIRECTORY"
```

Use the generated `client.env` values in the laptop application's configuration.
The endpoints are authenticated `GET /health`, `GET /v1/model` and
`POST /v1/audio/transcriptions?max_tokens=4096`. Upload mono 16 kHz PCM16 WAV bytes
with `Content-Type: audio/wav` and `Authorization: Bearer <token>`; the supplied
client converts audio for you. Audio is limited to 120 seconds per request,
one inference runs at a time, and busy callers receive 503 with a retry hint.
Partial output is rejected. The API is intended for a private network.
