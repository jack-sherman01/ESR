"""Consolidation preserves unique bytes and requires complete source absorption."""
from pathlib import Path
import pytest
from esrb.runtime.io import read
from esrb.tools.data_consolidation import overlay, protect_sources, publish, resolve_original


def test_merge_replaces_current_files_without_modifying_linked_input(tmp_path):
    old, new, stage = [tmp_path / n for n in ('old','new','stage')]
    for p in old,new,stage: p.mkdir()
    (old/'result.json').write_text('unknown')
    (new/'result.json').write_text('explained')
    overlay(old, stage)
    overlay(new, stage)
    assert (old/'result.json').read_text() == 'unknown'
    assert (stage/'result.json').read_text() == 'explained'


def test_verified_publish_removes_absorbed_revision_and_keeps_old_receipt(tmp_path):
    old, new, stage = [tmp_path/n for n in ('main','repair','stage')]
    for p in old,new,stage: p.mkdir()
    (old/'result').write_text('unknown')
    (new/'result').write_text('reason')
    (old/'raw').write_bytes(b'physics')
    (new/'raw').write_bytes(b'physics')
    overlay(old, stage); overlay(new, stage)
    protect_sources(stage, [old,new])
    publish(stage, old, [old,new])
    assert (old/'result').read_text() == 'reason'
    assert not new.exists()
    assert Path(resolve_original(old, old/'result')).read_text() == 'unknown'
    assert read(old/'cleanup.json')['status'] == 'COMPLETE'


def test_changed_source_aborts_cleanup_and_preserves_every_source(tmp_path):
    old, new, stage = [tmp_path/n for n in ('main','repair','stage')]
    for p in old,new,stage: p.mkdir()
    (old/'result').write_text('old'); (new/'result').write_text('new')
    overlay(new,stage); protect_sources(stage,[old,new])
    # Simulate an actively modified revision; no revision may be deleted.
    (new/'new-file').write_text('concurrent')
    with pytest.raises(ValueError, match='file set changed'):
        publish(stage,old,[old,new])
    assert new.exists()
    assert old.with_name('.main.pre-merge').exists()


def test_symlink_target_is_indexed_without_copying_external_files(tmp_path):
    source, stage, external = [tmp_path/n for n in ('source','stage','external')]
    for p in source,stage,external: p.mkdir()
    (external/'data').write_text('not owned')
    (source/'external').symlink_to(external, target_is_directory=True)
    overlay(source,stage)
    rows = protect_sources(stage,[source])
    assert len(rows)==1 and rows[0]['kind']=='symlink'
    assert not list((stage/'provenance').glob('objects/*/*'))


def test_deduplicates_same_physics_even_if_independent_inodes(tmp_path):
    source, stage = tmp_path/'source',tmp_path/'stage'
    source.mkdir();stage.mkdir()
    (source/'a').write_bytes(b'physics');(source/'b').write_bytes(b'physics')
    rows=protect_sources(stage,[source])
    assert rows[0]['stored']==rows[1]['stored']
    assert len(list((stage/'provenance/objects').glob('*/*')))==1


def test_absorbed_loose_log_can_be_removed_only_after_its_bytes_are_retained(tmp_path):
    old, stage = tmp_path/'main',tmp_path/'stage'
    old.mkdir();stage.mkdir()
    (old/'data').write_bytes(b'physics')
    log=tmp_path/'repair.log';log.write_text('failed attempt')
    overlay(old,stage);protect_sources(stage,[old,log]);publish(stage,old,[old,log])
    assert not log.exists()
    assert Path(resolve_original(old,log)).read_text()=='failed attempt'
