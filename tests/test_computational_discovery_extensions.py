from open_deep_research.computational.configuration import ComputationalConfiguration
from open_deep_research.computational.experiment_packs import get_experiment_pack, list_experiment_packs
from open_deep_research.computational.state import (
    HypothesisReflectionRecord,
    NoveltyCheckRecord,
    PaperDraftRecord,
    PaperReviewRecord,
)


def test_new_configuration_defaults_present():
    cfg = ComputationalConfiguration()
    assert cfg.enable_novelty_engine is True
    assert cfg.enable_hypothesis_reflection is True
    assert cfg.enable_paper_pipeline is True
    assert cfg.orchestrator_max_runs >= 1


def test_experiment_pack_registry_has_astronomy_default():
    pack = get_experiment_pack("astronomy_default")
    assert pack.id == "astronomy_default"
    assert "statistical_result" in pack.required_outputs
    assert any(p.id == "astronomy_default" for p in list_experiment_packs())


def test_new_state_records_can_be_created():
    novelty = NoveltyCheckRecord(hypothesis_text="A test hypothesis")
    reflection = HypothesisReflectionRecord(
        original_hypothesis="H0",
        refined_hypothesis="H1",
    )
    draft = PaperDraftRecord(body_markdown="# Draft")
    review = PaperReviewRecord(draft_id=draft.id)

    assert novelty.id
    assert reflection.refined_hypothesis == "H1"
    assert draft.body_markdown.startswith("#")
    assert review.draft_id == draft.id

