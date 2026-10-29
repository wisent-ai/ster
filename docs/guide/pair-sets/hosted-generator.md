# Synthesis through Brama

How `ster pairs synthesize --generator brama` reaches a hosted model. The
[pair sets guide](../pair-sets.md) links here from its synthesis section.

The hosted route reads Brama's own documented client variables: `BRAMA_URL`,
the gateway base, and `BRAMA_BEARER`, the caller's bearer. Both are required;
there is no built-in gateway. Ster reads both from its own environment
and never reads a vault itself. Nothing puts them there for you: there is no
launcher script in this repository, no wrapper on the fleet, and no service
unit that exports them, and Ster Desktop runs `ster request` with the
environment it inherited without setting a variable of its own. You export
both before the run. An unset base is refused with
`BRAMA_URL is unset or empty; export the Brama gateway base Ster should call`,
an empty bearer with
`BRAMA_BEARER is unset or empty; export Ster's own Brama bearer before running this command`,
a bearer that is not Unicode, without printing it, with
`BRAMA_BEARER is not Unicode; export Ster's own Brama bearer as text before running this command`,
and a base that is neither https nor explicit loopback with
`BRAMA_URL must be an https:// base or an explicit http:// loopback address, because Brama answers plain http elsewhere with 426 secure_transport_required`.
Loopback is `localhost` or any loopback IP address, IPv6 in brackets.

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
gateway on the first pair, in its own words. A gateway refusal is
surfaced in Brama's own words, as `brama refused the completion: 401 unauthorized`
— the status followed by the `message` from Brama's `{"error":{...}}` envelope.
A body without that envelope becomes
`brama refused the completion with {status} and a body that is not its error envelope: {body}`.
The body is retained whole, including HTML. An unreadable refusal names the
gateway and HTTP status alongside the read cause; malformed successful replies
retain their complete body beside the JSON or missing-message diagnostic.

No call to the gateway is cut off after a fixed time, and none waits without
a word: each completion goes through Stado's wait mechanism, which writes a
`czekam` line to stderr naming the route and the gateway URL before the call,
then `koniec czekania` when the answer arrives or `blad czekania` with the
cause when it does not. The `--reward` scorer of `ster tune grpo` and the
Hugging Face reads of `ster decisions fetch` announce their calls the same way.
