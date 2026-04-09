from __future__ import annotations

import json
import random
import uuid
from typing import Any

from datasets import Dataset

import verifiers as vf

from redesign_scope import (
    STRATEGIES,
    build_scope05_rows,
    candidate_id,
    design_variants_from_strategy,
    format_candidate_table,
    make_full_screen_report,
    make_quick_screen_report,
    parse_candidate_id,
)
from synthetic_dataset import normalize_sequence

SYSTEM_PROMPT = """You are running a budgeted peptide-binder redesign campaign.
Work strategically: inspect the candidate table, create variants with design strategies, screen promising candidates, and submit the best screened candidate.
Reserve one turn for the final answer. If budget is getting low or you already have a strong screened candidate, stop searching and answer.
Always finish with both tags, for example:
<answer>C0003</answer>
<sequence>ACDEFGHIK</sequence>"""


def _parse_submission(completion: vf.Messages, parser: vf.XMLParser) -> tuple[str, str]:
    selected_id = ""
    selected_sequence = ""

    for msg in reversed(parser.get_assistant_messages(completion)):
        content = parser._content_to_text(
            msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
        )
        parsed = parser.parse(content, last=True)

        raw_id = (getattr(parsed, "answer", None) or "").strip().upper()
        if parse_candidate_id(raw_id) is not None:
            selected_id = f"C{parse_candidate_id(raw_id):04d}"

        raw_sequence = getattr(parsed, "sequence", None) or ""
        if raw_sequence:
            selected_sequence = normalize_sequence(raw_sequence)

        if selected_id or selected_sequence:
            break

    return selected_id, selected_sequence


async def selection_reward(completion: vf.Messages, state: vf.State, parser: vf.XMLParser) -> float:
    selected_id, selected_sequence = _parse_submission(completion, parser)
    if not selected_id or not selected_sequence:
        return 0.0

    screened_ids = set(state.get("screened_ids", []))
    if selected_id not in screened_ids:
        return 0.0

    candidate_sequences = state.get("candidate_sequences", {})
    if candidate_sequences.get(selected_id) != selected_sequence:
        return 0.0

    candidate_truths = state.get("candidate_truths", {})
    true_score = float(candidate_truths.get(selected_id, 0.0))
    seed_score = float(state.get("seed_true_score", 0.0))
    budget_total = float(state.get("budget_total", 1.0))
    budget_remaining = float(state.get("budget_remaining", budget_total))
    budget_efficiency = budget_remaining / max(budget_total, 1e-8)
    improvement = max(0.0, true_score - seed_score)

    reward = 0.7 * true_score + 0.2 * improvement + 0.1 * budget_efficiency
    return max(0.0, min(1.0, reward))


async def chosen_true_score(completion: vf.Messages, state: vf.State, parser: vf.XMLParser) -> float:
    selected_id, _ = _parse_submission(completion, parser)
    if not selected_id:
        return 0.0
    return float(state.get("candidate_truths", {}).get(selected_id, 0.0))


async def chosen_improvement(completion: vf.Messages, state: vf.State, parser: vf.XMLParser) -> float:
    selected_id, _ = _parse_submission(completion, parser)
    if not selected_id:
        return 0.0
    true_score = float(state.get("candidate_truths", {}).get(selected_id, 0.0))
    seed_score = float(state.get("seed_true_score", 0.0))
    return max(0.0, true_score - seed_score)


async def screened_selection_metric(completion: vf.Messages, state: vf.State, parser: vf.XMLParser) -> float:
    selected_id, _ = _parse_submission(completion, parser)
    if not selected_id:
        return 0.0
    return 1.0 if selected_id in set(state.get("screened_ids", [])) else 0.0


async def chose_full_screened(completion: vf.Messages, state: vf.State, parser: vf.XMLParser) -> float:
    selected_id, _ = _parse_submission(completion, parser)
    if not selected_id:
        return 0.0
    return 1.0 if selected_id in set(state.get("full_screened_ids", [])) else 0.0


async def budget_efficiency_metric(completion: vf.Messages, state: vf.State) -> float:
    budget_total = float(state.get("budget_total", 1.0))
    budget_remaining = float(state.get("budget_remaining", budget_total))
    return budget_remaining / max(budget_total, 1e-8)


async def best_screened_score_metric(completion: vf.Messages, state: vf.State) -> float:
    screened_ids = set(state.get("screened_ids", []))
    truths = state.get("candidate_truths", {})
    if not screened_ids:
        return 0.0
    return max(float(truths[candidate]) for candidate in screened_ids if candidate in truths)


async def candidate_count_metric(completion: vf.Messages, state: vf.State) -> float:
    return float(state.get("candidate_count", 0))


async def design_calls_metric(completion: vf.Messages, state: vf.State) -> float:
    return float(state.get("design_calls", 0))


async def quick_screen_calls_metric(completion: vf.Messages, state: vf.State) -> float:
    return float(state.get("quick_screen_calls", 0))


async def full_screen_calls_metric(completion: vf.Messages, state: vf.State) -> float:
    return float(state.get("full_screen_calls", 0))


async def submitted_sequence_matches_candidate(completion: vf.Messages, state: vf.State, parser: vf.XMLParser) -> float:
    selected_id, selected_sequence = _parse_submission(completion, parser)
    if not selected_id or not selected_sequence:
        return 0.0
    candidate_sequences = state.get("candidate_sequences", {})
    return 1.0 if candidate_sequences.get(selected_id) == selected_sequence else 0.0


class ProteinBinderScopeEnv(vf.StatefulToolEnv):
    def __init__(self, **kwargs):
        super().__init__(tools=[], **kwargs)
        self.add_tool(self.list_candidates, args_to_skip=["rollout_id"])
        self.add_tool(self.design_variants, args_to_skip=["rollout_id"])
        self.add_tool(self.quick_screen, args_to_skip=["rollout_id"])
        self.add_tool(self.full_screen, args_to_skip=["rollout_id"])
        self._sessions: dict[str, dict[str, Any]] = {}

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        rollout_id = uuid.uuid4().hex
        info = dict(state.get("info", {}) or {})
        seed_sequence = info["seed_sequence"]
        seed_score = float(info["seed_score"])

        session = {
            "info": info,
            "budget_total": float(info["budget"]),
            "budget_remaining": float(info["budget"]),
            "next_candidate_index": 1,
            "rng": random.Random(hash((rollout_id, info["target_id"])) & 0xFFFFFFFF),
            "design_calls": 0,
            "quick_screen_calls": 0,
            "full_screen_calls": 0,
            "candidates": {
                "C0000": {
                    "candidate_id": "C0000",
                    "parent_id": None,
                    "sequence": seed_sequence,
                    "mutations": ["seed"],
                    "true_score": seed_score,
                    "quick_score": None,
                    "full_score": None,
                    "quick_report": None,
                    "full_report": None,
                    "screened": False,
                }
            },
        }
        self._sessions[rollout_id] = session
        state["rollout_id"] = rollout_id
        self._sync_state(state, session)
        return await super().setup_state(state, **kwargs)

    @vf.cleanup
    async def cleanup_session(self, state: vf.State):
        rollout_id = state.get("rollout_id")
        if rollout_id and rollout_id in self._sessions:
            del self._sessions[rollout_id]

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        tool_args["rollout_id"] = state["rollout_id"]
        return tool_args

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> vf.Messages:
        result = await super().env_response(messages, state, **kwargs)
        rollout_id = state.get("rollout_id")
        if rollout_id and rollout_id in self._sessions:
            self._sync_state(state, self._sessions[rollout_id])
        return result

    def _sync_state(self, state: vf.State, session: dict[str, Any]) -> None:
        candidates = session["candidates"]
        truths = {candidate_id: data["true_score"] for candidate_id, data in candidates.items()}
        sequences = {candidate_id: data["sequence"] for candidate_id, data in candidates.items()}
        screened_ids = [candidate_id for candidate_id, data in candidates.items() if data.get("screened")]
        full_screened_ids = [candidate_id for candidate_id, data in candidates.items() if data.get("full_report") is not None]
        state["budget_total"] = session["budget_total"]
        state["budget_remaining"] = session["budget_remaining"]
        state["candidate_truths"] = truths
        state["candidate_sequences"] = sequences
        state["screened_ids"] = screened_ids
        state["full_screened_ids"] = full_screened_ids
        state["candidate_count"] = len(candidates)
        state["design_calls"] = session["design_calls"]
        state["quick_screen_calls"] = session["quick_screen_calls"]
        state["full_screen_calls"] = session["full_screen_calls"]
        state["seed_true_score"] = candidates["C0000"]["true_score"]
        state["best_known_candidate"] = max(truths, key=truths.get)

    def _session(self, rollout_id: str) -> dict[str, Any]:
        session = self._sessions.get(rollout_id)
        if session is None:
            raise ValueError("Invalid rollout_id")
        return session

    def _best_screened_candidate(self, session: dict[str, Any]) -> dict[str, Any] | None:
        screened = [candidate for candidate in session["candidates"].values() if candidate.get("screened")]
        if not screened:
            return None
        best = max(
            screened,
            key=lambda candidate: candidate.get("full_score") if candidate.get("full_score") is not None else candidate.get("quick_score") if candidate.get("quick_score") is not None else -1.0,
        )
        return {
            "candidate_id": best["candidate_id"],
            "known_score": best.get("full_score") if best.get("full_score") is not None else best.get("quick_score"),
            "screen_type": "full" if best.get("full_score") is not None else "quick",
        }

    def _budget_error(self, session: dict[str, Any], required_cost: float) -> str:
        return json.dumps(
            {
                "error": "insufficient_budget",
                "required_cost": round(required_cost, 3),
                "budget_remaining": round(session["budget_remaining"], 3),
                "suggested_action": "submit_best_screened_candidate_now",
                "best_screened_candidate": self._best_screened_candidate(session),
            },
            indent=2,
        )

    def _parse_candidate_ids(self, raw_ids: str, session: dict[str, Any]) -> tuple[list[str], list[str]]:
        normalized_ids = [f"C{parse_candidate_id(item):04d}" for item in [part.strip() for part in raw_ids.split(",") if part.strip()] if parse_candidate_id(item) is not None]
        unique_ids = list(dict.fromkeys(normalized_ids))
        missing = [candidate for candidate in unique_ids if candidate not in session["candidates"]]
        return unique_ids, missing

    async def list_candidates(self, rollout_id: str) -> str:
        """Show the current candidate table and remaining budget.

        Returns:
            JSON with the current candidate table, including latest quick/full screen scores.
        """
        session = self._session(rollout_id)
        return json.dumps(
            {
                "budget_remaining": round(session["budget_remaining"], 3),
                "best_screened_candidate": self._best_screened_candidate(session),
                "candidates": json.loads(format_candidate_table(session["candidates"])),
            },
            indent=2,
        )

    async def design_variants(self, parent_id: str, strategy: str, num_variants: int, rollout_id: str) -> str:
        """Create new binder variants from an existing candidate.

        Args:
            parent_id: Candidate ID to mutate, for example C0000.
            strategy: One of balanced, anchor, composition, charge, or explore.
            num_variants: Number of variants to create in one batch (1-4 recommended).

        Returns:
            JSON with created candidate IDs, sequences, mutations, and remaining budget.
        """
        session = self._session(rollout_id)
        strategy = (strategy or "balanced").strip().lower()
        if strategy not in STRATEGIES:
            return json.dumps({"error": "unknown_strategy", "allowed": STRATEGIES}, indent=2)

        parent_idx = parse_candidate_id(parent_id)
        if parent_idx is None:
            return json.dumps({"error": "invalid_parent_id", "parent_id": parent_id}, indent=2)
        normalized_parent_id = candidate_id(parent_idx)
        parent = session["candidates"].get(normalized_parent_id)
        if parent is None:
            return json.dumps({"error": "unknown_parent_id", "parent_id": normalized_parent_id}, indent=2)

        clamped_num_variants = max(1, min(int(num_variants), 4))
        design_cost = float(session["info"]["costs"]["design"])
        if session["budget_remaining"] < design_cost:
            return self._budget_error(session, design_cost)

        existing_sequences = {candidate["sequence"] for candidate in session["candidates"].values()}
        created = design_variants_from_strategy(
            parent_sequence=parent["sequence"],
            strategy=strategy,
            num_variants=clamped_num_variants,
            spec=session["info"],
            rng=session["rng"],
            existing_sequences=existing_sequences,
        )
        if not created:
            return json.dumps({"error": "no_variants_created", "parent_id": normalized_parent_id}, indent=2)

        session["budget_remaining"] -= design_cost
        session["design_calls"] += 1

        created_payload = []
        for variant in created:
            new_id = candidate_id(session["next_candidate_index"])
            session["next_candidate_index"] += 1
            session["candidates"][new_id] = {
                "candidate_id": new_id,
                "parent_id": normalized_parent_id,
                "sequence": variant["sequence"],
                "mutations": variant["mutations"],
                "true_score": variant["true_score"],
                "quick_score": None,
                "full_score": None,
                "quick_report": None,
                "full_report": None,
                "screened": False,
            }
            created_payload.append(
                {
                    "candidate_id": new_id,
                    "parent_id": normalized_parent_id,
                    "sequence": variant["sequence"],
                    "mutations": variant["mutations"],
                }
            )

        return json.dumps(
            {
                "strategy": strategy,
                "created": created_payload,
                "budget_remaining": round(session["budget_remaining"], 3),
                "best_screened_candidate": self._best_screened_candidate(session),
            },
            indent=2,
        )

    async def quick_screen(self, candidate_ids: str, rollout_id: str) -> str:
        """Run a cheap, noisy screen on one or more candidates.

        Args:
            candidate_ids: Comma-separated candidate IDs, for example C0001,C0002.

        Returns:
            JSON with approximate scores and coarse anchor/charge/position fit metrics.
        """
        session = self._session(rollout_id)
        normalized_ids, missing = self._parse_candidate_ids(candidate_ids, session)
        if missing:
            return json.dumps({"error": "unknown_candidate_ids", "candidate_ids": missing}, indent=2)
        if not normalized_ids:
            return json.dumps({"error": "no_candidate_ids"}, indent=2)

        required_cost = float(session["info"]["costs"]["quick"]) * len(normalized_ids)
        if session["budget_remaining"] < required_cost:
            return self._budget_error(session, required_cost)

        session["budget_remaining"] -= required_cost
        session["quick_screen_calls"] += 1
        results = []
        for candidate_key in normalized_ids:
            candidate = session["candidates"][candidate_key]
            if candidate["quick_report"] is None:
                candidate["quick_report"] = make_quick_screen_report(candidate["sequence"], session["info"], session["rng"])
                candidate["quick_score"] = candidate["quick_report"]["score"]
            candidate["screened"] = True
            results.append({"candidate_id": candidate_key, **candidate["quick_report"]})

        results.sort(key=lambda row: row["score"], reverse=True)
        return json.dumps(
            {
                "results": results,
                "budget_remaining": round(session["budget_remaining"], 3),
                "best_screened_candidate": self._best_screened_candidate(session),
            },
            indent=2,
        )

    async def full_screen(self, candidate_ids: str, rollout_id: str) -> str:
        """Run a stronger, more expensive screen on one or more candidates.

        Args:
            candidate_ids: Comma-separated candidate IDs, for example C0003,C0005.

        Returns:
            JSON with stronger scores and detailed proxy breakdowns.
        """
        session = self._session(rollout_id)
        normalized_ids, missing = self._parse_candidate_ids(candidate_ids, session)
        if missing:
            return json.dumps({"error": "unknown_candidate_ids", "candidate_ids": missing}, indent=2)
        if not normalized_ids:
            return json.dumps({"error": "no_candidate_ids"}, indent=2)

        required_cost = float(session["info"]["costs"]["full"]) * len(normalized_ids)
        if session["budget_remaining"] < required_cost:
            return self._budget_error(session, required_cost)

        session["budget_remaining"] -= required_cost
        session["full_screen_calls"] += 1
        results = []
        for candidate_key in normalized_ids:
            candidate = session["candidates"][candidate_key]
            if candidate["full_report"] is None:
                candidate["full_report"] = make_full_screen_report(candidate["sequence"], session["info"], session["rng"])
                candidate["full_score"] = candidate["full_report"]["score"]
            candidate["screened"] = True
            results.append({"candidate_id": candidate_key, **candidate["full_report"]})

        results.sort(key=lambda row: row["score"], reverse=True)
        return json.dumps(
            {
                "results": results,
                "budget_remaining": round(session["budget_remaining"], 3),
                "best_screened_candidate": self._best_screened_candidate(session),
            },
            indent=2,
        )


def load_environment(
    num_train_examples: int = 96,
    num_eval_examples: int = 24,
    train_seed: int = 7,
    eval_seed: int = 17,
    max_turns: int = 10,
) -> vf.Environment:
    """Build the scope 0.5 budgeted binder-redesign environment."""

    train_dataset = Dataset.from_list(build_scope05_rows(num_train_examples, train_seed, split="train"))
    eval_dataset = Dataset.from_list(build_scope05_rows(num_eval_examples, eval_seed, split="eval"))

    parser = vf.XMLParser(["answer", "sequence"], answer_field="answer")
    rubric = vf.Rubric(funcs=[selection_reward, parser.get_format_reward_func()], weights=[1.0, 0.1], parser=parser)
    rubric.add_metric(chosen_true_score)
    rubric.add_metric(chosen_improvement)
    rubric.add_metric(screened_selection_metric)
    rubric.add_metric(chose_full_screened)
    rubric.add_metric(submitted_sequence_matches_candidate)
    rubric.add_metric(budget_efficiency_metric)
    rubric.add_metric(best_screened_score_metric)
    rubric.add_metric(candidate_count_metric)
    rubric.add_metric(design_calls_metric)
    rubric.add_metric(quick_screen_calls_metric)
    rubric.add_metric(full_screen_calls_metric)

    return ProteinBinderScopeEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        parser=parser,
        system_prompt=SYSTEM_PROMPT,
        max_turns=max_turns,
    )
