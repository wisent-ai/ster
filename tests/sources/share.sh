#!/bin/sh
# Real test of `<program> sources share [yes | no]` (`<program>.sources`) through the built
# binary: alone and unanswered it asks the question and keeps nothing; `yes` keeps the
# answer with a random subject in $XDG_CONFIG_HOME/<program>/sources-share.json, `no`
# changes the answer under the same subject, and the question then shows the answer kept;
# an answer that cannot be written is refused with its declared sentence.
#
# Usage: tests/sources/share.sh   (<environment prefix>BIN selects the binary, see tests/support/names.sh)
cd "$(dirname "$0")/../.."
. tests/sources/harness.sh share
KEPT="$XDG_CONFIG_HOME/$PROGRAM/sources-share.json"

the_unanswered_question_is_asked_and_nothing_kept() {
  run 0 sources share
  [ "$(printf '%s' "$out" | jq -r .answer_with)" = "$PROGRAM sources share yes | no" ] || fail "the question names no answer: $out"
  [ "$(printf '%s' "$out" | jq -r .sends.adapter)" = dpo ] || fail "the question's example event does not name dpo: $out"
  [ -z "$(kept_files)" ] || fail "asking kept files: $(kept_files)"
  passed the_unanswered_question_is_asked_and_nothing_kept
}

an_answer_is_kept_and_changed_under_one_subject() {
  run 0 sources share yes
  [ "$(jq .share "$KEPT")" = true ] || fail "yes was not kept in $KEPT: $(cat "$KEPT")"
  subject=$(jq -r .subject "$KEPT")
  printf '%s' "$subject" | grep -Eq '^[0-9a-f]{64}$' || fail "the kept subject is not 32 random bytes in hex: $subject"
  run 0 sources share no
  [ "$(jq .share "$KEPT")" = false ] || fail "no was not kept in $KEPT: $(cat "$KEPT")"
  [ "$(jq -r .subject "$KEPT")" = "$subject" ] || fail "changing the answer changed the subject: $(cat "$KEPT")"
  run 0 sources share
  [ "$(printf '%s' "$out" | jq .share)" = false ] || fail "the question does not show the answer kept: $out"
  passed an_answer_is_kept_and_changed_under_one_subject
}

an_answer_to_sharing_that_cannot_be_kept_is_refused() {
  # XDG_CONFIG_HOME names a file, so the folder the answer goes in cannot be made.
  printf 'a file where the answer folder would go\n' >"$SCRATCH/config-is-a-file"
  export XDG_CONFIG_HOME="$SCRATCH/config-is-a-file"
  run 1 sources share no
  export XDG_CONFIG_HOME="$SCRATCH/config"
  said "keeping whether discoveries report their counts"
  said "could not be written"
  passed an_answer_to_sharing_that_cannot_be_kept_is_refused
}

the_unanswered_question_is_asked_and_nothing_kept
an_answer_is_kept_and_changed_under_one_subject
an_answer_to_sharing_that_cannot_be_kept_is_refused
finished
