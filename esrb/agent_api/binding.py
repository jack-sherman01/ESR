"""Validate proposals before any candidate execution."""

from copy import deepcopy
from dataclasses import asdict

from .contracts import ParentView, Proposal, SCHEMA, identity, require


def apply_proposal(proposer, manifest_id, components, feedback, validators):
    """Validate a proposal against the host's writable roles and trusted consumers."""
    manifest = proposer.manifest
    require(manifest.id == manifest_id, "bound proposal manifest changed")
    consumers = {item.role: item for item in manifest.consumers}
    require(set(consumers) <= set(validators) and set(consumers) <= set(components),
            "proposal manifest declares a role without a host consumer")
    require(all(consumer == validators[role][0] for role, consumer in consumers.items()),
            "proposal manifest declares an unsupported host consumer format")
    feedback_id = feedback.validate()
    public_view = deepcopy(feedback)
    version = identity(components)
    parent = ParentView(version, deepcopy(components), manifest_id)
    proposal = proposer.propose(parent, public_view)
    require(identity(parent.components) == version, "proposer mutated the parent view")
    require(public_view.validate() == feedback_id, "proposer mutated public feedback")
    require(proposer.manifest.id == manifest_id, "bound proposal manifest changed")
    require(isinstance(proposal, Proposal) and proposal.schema == SCHEMA, "unsupported proposal schema")
    require(proposal.base_version == version, "stale proposal parent version")
    require(proposal.manifest_id == manifest_id, "foreign proposal manifest")
    require(type(proposal.replacements) is dict and proposal.replacements
            and set(proposal.replacements) <= set(consumers), "unauthorized proposal role")
    require(type(proposal.formats) is dict and set(proposal.formats) == set(proposal.replacements),
            "proposal formats must match the replacement roles")
    payload = asdict(proposal)
    proposal_id = identity(payload)
    child = deepcopy(components)
    for role, replacement in proposal.replacements.items():
        require(proposal.formats[role] == consumers[role].format, "unsupported proposal format")
        child[role] = validators[role][1](replacement)
    return child, {
        "manifest": asdict(manifest), "manifest_id": manifest_id,
        "base_version": version, "candidate_version": identity(child),
        "parent_components": deepcopy(components), "candidate_components": child,
        "proposal_id": proposal_id, "proposal": payload, "feedback_id": feedback_id,
        "public_feedback": asdict(feedback),
    }
