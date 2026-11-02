# Synthesis through Brama

How `ster pairs synthesize --generator brama` reaches a hosted model. The
[pair sets guide](../pair-sets.md) links here from its synthesis section.

The hosted route never dials Brama itself. An exported `BRAMA_URL`, the
gateway base, is the operator's choice and is used as given, with
`BRAMA_BEARER`, the caller's bearer. Without `BRAMA_URL`, Ster names no model
service: each request is handed to `stado capabilities call inference
--consumer ster`, which sends it to the first alternative in Stado's selected
order that answers it — Brama, or a statement recorded for that exact request
(`stado capabilities attestation record`) — and prints its answer. An
alternative Stado adds, or an order it selects, changes what writes Ster's
pairs without Ster changing, and no bearer of Ster's is read then. The report
names the generator as `<alternative>:<route>`: `brama:<route>` for an exported
`BRAMA_URL`, else the alternative Stado says answered, such as
`attestation:<route>`. When no alternative answers, the run is refused with
`BRAMA_URL is unset and no inference alternative answered \`stado capabilities
call inference --consumer ster --json\` (<status>): <Stado's sentence>; answer
it as Stado's sentence says, or export BRAMA_URL`; Stado's sentence names every
alternative with its cause and the command that answers the kept request.
With `BRAMA_URL` exported, nothing exports `BRAMA_BEARER` for you: there is no
launcher script in this repository, no wrapper on the fleet, and no service
unit that sets it, and Ster Desktop runs `ster request` with the environment it
inherited. An empty bearer is refused with
`BRAMA_BEARER is unset or empty; export Ster's own Brama bearer before running this command`,
a bearer that is not Unicode, without printing it, with
`BRAMA_BEARER is not Unicode; export Ster's own Brama bearer as text before running this command`,
and a base that is neither https nor explicit loopback with
`the model service base <base> must be https:// or an explicit http:// loopback address, because Brama answers plain http elsewhere with 426 secure_transport_required`.
Loopback is `localhost` or any loopback IP address, IPv6 in brackets. The real
journey is `tests/synthesis/resolved.sh`.

The bearer to export is Ster's own client identity, never another product's.
Ster is a declared consumer of `brama` in Stado's service directory —
`stado resolver resolve brama --consumer ster` answers
`stado://service/brama capabilities=model-routing` — and the bearer itself is a
Skarbiec grant for the consumer `ster` whose capability is `call:brama#<route>`.
Brama resolves a bearer its own start did not preload by introspecting Skarbiec
for exactly that grant, so the route the grant names is the only route
`--generator-model` may then use.

Which Skarbiec matters, and this is the part that is easy to get wrong: the
gateway introspects the vault on its **own** host, not yours. A grant minted in
a workstation's vault authenticates against that workstation and is answered
`401 unauthenticated` by the gateway, which never consults it. Mint it where the
gateway will look, through Stado's managed channel rather than over ssh:

```bash
export BRAMA_BEARER="$(stado host vault-token-mint <gateway-host> ster \
  --capabilities 'call:brama#openai/gpt-5-mini' --audience ster --raw-token)"
```

`--raw-token` prints the bearer and nothing else, for a direct pipe into a
variable or a secret store; Stado never writes it to disk or to a command line,
and neither should you. Drop `--raw-token` for non-secret metadata only. Read
the value at the moment of the call, keep it in a process-only variable, and let
it die with the process: a bearer belongs in neither a file nor an argv.

`--generator-model` may name exactly four things: a declared alias, such as
Wisent's own chat alias `wisent-backend/chat/primary` or its sibling
`wisent-backend/chat/fallback`; the delegation alias `best`; a canonical
`provider/model` route, such as `anthropic/claude-3-5-haiku-latest`; or a
selector, `any` or `task:<name>`. `best` and the selectors additionally require
an agent-signed request, which Ster does not construct — it sends a bearer and
nothing more — so a Ster run uses a declared alias or a canonical route.
Anything outside that vocabulary is refused with
`the generator model must be a Brama alias, a canonical provider/model route, or a selector`.

The request to the gateway carries exactly `model`, `messages` with one user
message, `max_tokens` and `temperature`, because Brama refuses unknown fields
by name. Neither `--seed` nor `--top-p` therefore travels, and a hosted run is
not seed-reproducible: the provider owns its sampler, and the running
deduplicator is what suppresses repeated draws. The bounds on `max_tokens` and
`temperature` are Brama's alone: a value outside them is refused by the
gateway on the first pair, in its own words. A refusal is surfaced in the
answering service's own words, as `brama refused the completion: 401 unauthorized`
— the status followed by the `message` from its
`{"error":{...}}` envelope. A body without that envelope becomes
`<provider> refused the completion with {status} and a body that is not its error envelope: {body}`.
The body is retained whole, including HTML. An unreadable refusal names the
gateway and HTTP status alongside the read cause; malformed successful replies
retain their complete body beside the JSON or missing-message diagnostic.

No call to the gateway is cut off after a fixed time, and none waits without
a word: each completion goes through Stado's wait mechanism, which writes a
`czekam` line to stderr naming the route and the gateway URL before the call,
then `koniec czekania` when the answer arrives or `blad czekania` with the
cause when it does not. The `--reward` scorer of `ster tune grpo` and the
Hugging Face reads of `ster decisions fetch` announce their calls the same way.
