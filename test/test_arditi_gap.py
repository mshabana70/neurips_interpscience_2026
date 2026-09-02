import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from arditi_gap.run import (  # noqa: E402
    choose_site,
    has_refusal_substring,
    heldout_prompts,
    parse_site,
    refusal_token_ids,
    site_key,
)


class ToyTokenizer:
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [10 if text.startswith("I") else 20, 99]


def _cell(layer, position, global_bypass, local_bypass, induce=1.0, kl=0.01):
    return {
        "site": site_key(layer, position),
        "layer": layer,
        "position": position,
        "global_bypass": {"mean": global_bypass},
        "local_bypass": {"mean": local_bypass},
        "addition_induce": {"mean": induce},
        "global_kl_harmless": {"mean": kl},
        "local_kl_harmless": {"mean": kl},
    }


def test_site_roundtrip():
    assert parse_site(site_key(15, -2)) == (15, -2)
    assert parse_site("15@-2") == (15, -2)


def test_refusal_tokens_are_derived_and_deduplicated():
    assert refusal_token_ids(ToyTokenizer()) == [10, 20]


def test_refusal_substrings_cover_straight_and_curly_apostrophes():
    assert has_refusal_substring("I'm sorry, but I cannot help with that.")
    assert has_refusal_substring("I’m unable to assist.")
    assert not has_refusal_substring("Here is a benign explanation.")


def test_heldout_prompts_uses_an_exact_disjoint_slice(tmp_path):
    rows = [{"instruction": f"p{i}"} for i in range(12)]
    (tmp_path / "harmful_train.json").write_text(json.dumps(rows))
    assert heldout_prompts("harmful", 4, 3, tmp_path) == ["p4", "p5", "p6"]
    with pytest.raises(ValueError, match="cannot take"):
        heldout_prompts("harmful", 10, 3, tmp_path)


def test_selection_enforces_arditi_constraints_and_depth():
    cells = [
        _cell(1, -1, global_bypass=-2, local_bypass=-1),
        _cell(2, -1, global_bypass=-5, local_bypass=-4, induce=-1),
        _cell(9, -1, global_bypass=-9, local_bypass=-9),  # excluded at 0.8 * 10
    ]
    selected = choose_site(cells, n_layers=10, scope="global")
    assert selected["site"] == "L1@-1"
    assert selected["selection_status"] == "constraints_passed"


def test_selection_records_fallback_when_no_cell_passes():
    cells = [
        _cell(1, -1, global_bypass=-2, local_bypass=-1, induce=-1),
        _cell(2, -1, global_bypass=-5, local_bypass=-4, kl=0.2),
    ]
    selected = choose_site(cells, n_layers=10, scope="local")
    assert selected["site"] == "L2@-1"
    assert selected["selection_status"] == "fallback_no_candidate_passed"
