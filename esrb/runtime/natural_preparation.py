"""Audited preparation and explicitly authorized context-only runtime revisions."""
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import shutil

from .io import read, save, digest
from .natural import REPO, source_files, natural_api


def revise_context(campaign, *, authorization):
    """Keep original seals/sources; change only the model's history policy at rest.

    This is an explicit protocol amendment, not a normal resume or source refresh.
    No task/physics/API/model settings, proposals, results or lineage are rewritten.
    """
    from .natural import CONTEXT_POLICY, exclusive, validate, validate_lineage
    if not authorization.strip():
        raise ValueError('explicit context revision authorization required')
    campaign = Path(campaign).resolve()
    allowed = {'esrb/runtime/natural.py', 'esrb/runtime/natural_model.py',
               'esrb/runtime/natural_preparation.py'}
    with ExitStack() as stack:
        stack.enter_context(exclusive(campaign))
        entries = read(campaign/'manifest.json')['chains']
        checked = []
        # Validate every chain before mutating any of them.
        for item in entries:
            root = Path(item['root']).resolve()
            stack.enter_context(exclusive(root))
            protocol = read(root/'protocol.json')
            receipt = read(root/'freeze_receipt.json')
            if receipt['protocol_sha256'] != digest(root/'protocol.json'):
                raise ValueError('sealed natural protocol changed')
            if protocol.get('context_policy') == CONTEXT_POLICY:
                validate(root)
                continue
            if protocol.get('context_policy'):
                raise ValueError('unexpected prior context policy')
            if digest(root/'API.txt') != protocol['api_sha256']:
                raise ValueError('frozen API changed')
            changes = {}
            for name, expected in protocol['sources'].items():
                if digest(root/'frozen_sources'/name) != expected:
                    raise ValueError(f'frozen snapshot changed: {name}')
                current = digest(REPO/name)
                if current != expected:
                    if name not in allowed:
                        raise ValueError(f'context revision cannot change source: {name}')
                    changes[name] = {'before': expected, 'after': current}
            for name, expected in protocol.get('upstream_sources', {}).items():
                if digest(REPO/name) != expected or digest(root/'frozen_sources'/name) != expected:
                    raise ValueError(f'frozen ASPIRE source changed: {name}')
            state = validate_lineage(root)
            checked.append((root, protocol, state, changes))
        if not checked:
            return []
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        records = []
        for root, protocol, state, changes in checked:
            archive = root/'context_revisions'/stamp
            archive.mkdir(parents=True)
            for name in ('protocol.json', 'freeze_receipt.json', 'state.json'):
                shutil.copyfile(root/name, archive/name)
            # Small complete source snapshot retains independently verifiable old seals.
            shutil.copytree(root/'frozen_sources', archive/'frozen_sources')
            record = {'schema': 'esrb.natural_context_revision/1', 'revision': stamp,
                      'authorization': authorization, 'previous_protocol_sha256': digest(root/'protocol.json'),
                      'previous_receipt_sha256': digest(root/'freeze_receipt.json'),
                      'state_sha256': digest(root/'state.json'), 'completed_slots': len(state['events']),
                      'effective_from_version': len(state['events'])+1,
                      'context_policy': dict(CONTEXT_POLICY), 'changed_sources': changes}
            for name, change in changes.items():
                shutil.copyfile(REPO/name, root/'frozen_sources'/name)
                protocol['sources'][name] = change['after']
            protocol['context_policy'] = dict(CONTEXT_POLICY)
            protocol['context'] = 'independent chain; current base program and bounded recent task feedback; no tools'
            protocol['context_revision'] = {'revision': stamp,
                'previous_protocol_sha256': record['previous_protocol_sha256'],
                'effective_from_version': record['effective_from_version']}
            save(root/'protocol.json', protocol)
            receipt = read(archive/'freeze_receipt.json')
            receipt.update(protocol_sha256=digest(root/'protocol.json'), context_revision=stamp,
                           previous_receipt_sha256=record['previous_receipt_sha256'])
            save(root/'freeze_receipt.json', receipt)
            record['new_protocol_sha256'] = digest(root/'protocol.json')
            save(archive/'revision.json', record)
            validate(root)
            records.append(dict(record, root=str(root)))
        save(campaign/f'context_revision_{stamp}.json', {'chains': records})
        return records


def refresh_preparation_sources(campaign):
    campaign = Path(campaign)
    manifest = read(campaign/'manifest.json')
    for item in manifest['chains']:
        root = Path(item['root'])
        state = read(root/'state.json')
        if (root/'freeze_receipt.json').exists() or state['events'] or any((root/'candidates').iterdir()):
            raise ValueError('source revision forbidden after any formal proposal')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    revisions = []
    for item in manifest['chains']:
        root = Path(item['root']); protocol=read(root/'protocol.json')
        archive=root/'preparation_revisions'/stamp;archive.mkdir(parents=True)
        shutil.copyfile(root/'protocol.json',archive/'protocol.json')
        sources={}
        for path in source_files():
            name=str(path.relative_to(REPO));sources[name]=digest(path)
        api_name=protocol['scene']['prefix']+'_skill_api.txt'
        api=REPO/'integrations/aspire'/api_name
        if digest(api)!=protocol.get('api_source_sha256',protocol['api_sha256']):
            raise ValueError('public API revision requires a new experiment protocol')
        shutil.copyfile(root/'API.txt',archive/'API-before.txt')
        (root/'API.txt').write_text(natural_api(api.read_text()))
        protocol['api_source_sha256']=digest(api)
        protocol['api_sha256']=digest(root/'API.txt')
        sources[str(api.relative_to(REPO))]=digest(api)
        changes=[]
        for name,expected in sources.items():
            destination=root/'frozen_sources'/name
            if destination.exists() and digest(destination)!=expected:
                old=archive/name;old.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(destination,old);changes.append(name)
            elif not destination.exists():
                changes.append(name)
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,destination)
        protocol['sources']=sources
        protocol['preparation_revision']=stamp
        save(root/'protocol.json',protocol)
        revisions.append({'chain':str(root),'changed_sources':changes,'archived_protocol':str(archive/'protocol.json')})
    save(campaign/f'preparation_revision_{stamp}.json',{'formal_proposals':0,'revisions':revisions})
    return revisions
