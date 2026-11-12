# Synthesis through Stado's inference alternatives

How `ster pairs synthesize --generator brama` and `ster decisions synthesize`
reach a hosted model. The [pair sets guide](../pair-sets.md) links here from
its synthesis section.

Ster names no model service, gateway, key or provider, and reads no bearer.
Every request is handed to `stado capabilities call inference --consumer ster
--json`, which sends it to the first alternative in Stado's selected order
that answers it — Brama, or a statement recorded for that exact request
(`stado capabilities attestation record`) — and prints which alternative
answered beside its answer. An alternative Stado adds, or an order it selects
(`stado capabilities select inference`), changes what writes Ster's text
without Ster changing. Ster is a consumer of `inference` in Stado's service
directory; the credentials an alternative needs are Stado's and that
alternative's, never Ster's. The report names the generator as
`<alternative>:<route>`, the alternative Stado says answered, such as
`brama:wisent-backend/chat/primary` or `attestation:wisent-backend/chat/primary`.

`--generator-model` names the route asked for: a Brama alias, such as Wisent's
own chat alias `wisent-backend/chat/primary` or its sibling
`wisent-backend/chat/fallback`; a canonical `provider/model` route, such as
`anthropic/claude-3-5-haiku-latest`; or a selector. A blank route is refused
before Stado is asked. Whether the answering alternative serves that route is
its own decision, and its refusal reaches you in Stado's sentence.

The request carries exactly `model`, `messages` with one user message,
`max_tokens` and `temperature`, an OpenAI chat-completions request every
inference alternative takes. Neither `--seed` nor `--top-p` travels, and a
hosted run is not seed-reproducible: the alternative owns its sampler, and the
running deduplicator is what suppresses repeated draws. The bounds on
`max_tokens` and `temperature` are the answering alternative's.

## How it fails

Every way this layer refuses is `ster.inference`, listed with its sentence,
code, exit status and test on [errors/inference.md](../errors/inference.md):

- a blank route (exit 2);
- no `stado` at any place Stado states (`STADO_BIN`, else `PATH`, else
  `${STADO_BIN_DIR:-$HOME/.stado/bin}/stado`), a `STADO_BIN` naming no file,
  or a `stado` that this account may not execute or whose interpreter is
  missing (exit 1);
- no alternative answers (exit 69, because asking again once an alternative
  is reachable or a statement is recorded can help): the refusal reads
  `no inference alternative answered \`stado capabilities call inference
  --consumer ster --json\` (<status>): <Stado's sentence>`, and Stado's
  sentence names every alternative in its order with its cause and the
  `stado capabilities attestation record inference --consumer ster <request>`
  command that answers the request it kept;
- Stado refusing the request itself (exit 1), or printing something other than
  its `--json` form;
- an answer with no assistant message at `choices[0].message.content`.

The real journeys are `tests/synthesis/inference.sh` (the route and Stado
refusals) and `tests/synthesis/resolved.sh` (nothing answers, then every kept
request answered by a recorded statement).

No call is cut off after a fixed time, and none waits without a word: Stado's
wait mechanism writes a `czekam` line to stderr naming the call before it
starts, then `koniec czekania` when it ends or `blad czekania` with the cause
when it does not. The `--reward` scorer of `ster tune grpo` and the Hugging
Face reads of `ster decisions fetch` announce their calls the same way.
