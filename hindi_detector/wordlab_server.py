"""Read-only private listening lab; no annotation writes or arbitrary file access."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import urlparse

from hindi_detector.review_server import read_items

LOG = logging.getLogger(__name__)


def audio_byte_range(body, header):
    """Single HTTP byte ranges let browser audio seek to the selected word."""
    total=len(body)
    if header is None:return 200,body,{'Accept-Ranges':'bytes'}
    invalid=(416,b'',{'Accept-Ranges':'bytes','Content-Range':f'bytes */{total}'})
    if len(header)>128:return invalid
    match=re.fullmatch(r'bytes=(\d*)-(\d*)',header.strip())
    if match is None or not any(match.groups()) or not total:return invalid
    first,last=match.groups()
    if first:
        start=int(first);end=min(int(last),total-1) if last else total-1
    else:
        suffix=int(last)
        if not suffix:return invalid
        start=max(0,total-suffix);end=total-1
    if start>=total or end<start:return invalid
    return 206,body[start:end+1],{'Accept-Ranges':'bytes','Content-Range':f'bytes {start}-{end}/{total}'}


def vowel_directory(root):
    pointer=root/'active_vowel_model.json'
    if not pointer.is_file():return root/'reports/vowel_detector_v19'
    selected=json.loads(pointer.read_text());directory=(root/selected['model_directory']).resolve(strict=True)
    if not directory.is_relative_to((root/'reports').resolve()):raise ValueError('Selected vowel model outside owned reports')
    report=json.loads((directory/'report.json').read_text())
    if report['model_sha256']!=selected['model_sha256']:raise ValueError('Selected vowel model metadata differs')
    return directory


def review_case_additions(case,directory):
    path=directory/'review_additions.json'
    if not path.is_file():return case
    additions=json.loads(path.read_text()).get(case['id'],{})
    for source,reading in case['analysis'].items():
        words=additions.get(source)
        if words is None:continue
        if len(words)!=len(reading['words']):raise ValueError('Review addition word geometry differs')
        for word,addition in zip(reading['words'],words,strict=True):
            word['secondary_phone_probe_sequence']=addition['secondary_phone_probe_sequence']
            def key(site):return site['kind'],site['index'],tuple(site['native_phones'])
            sites={key(s):s for s in addition['sites']}
            for site in word['sites']:
                data=sites.get((site['site']['kind'],site['site']['index'],tuple(site['native_phones'])))
                if data is None:raise ValueError('Review phonetic site identity differs')
                site.update({k:v for k,v in data.items() if k not in {'kind','index','native_phones'}})
    return case


class ReadOnlyServer(ThreadingHTTPServer):
    # A human-word case opens up to eight audio requests together over Tailscale.
    request_queue_size = 64
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack', type=Path, action='append', required=True)
    parser.add_argument('--page', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8770)
    args = parser.parse_args()
    registry = {}
    for pack in args.pack:
        items, _ = read_items(pack)
        for identifier, row in items.items():
            if identifier in registry:
                raise ValueError('Duplicate audio identity across lab source packs')
            registry[identifier] = (pack / row['audio'], row['sha256'])
    index = json.loads((args.data / 'index.json').read_text())
    allowed = {row['id'] for row in index}
    manifest = json.loads((args.data / 'manifest.json').read_text())
    for member in manifest['files']:
        p = (args.data / member['path']).resolve(strict=True)
        if not p.is_relative_to(args.data.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest() != member['sha256']:
            raise ValueError('Lab data inventory mismatch')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            LOG.info('%s %s', self.client_address[0], fmt % values)

        def send(self, code, body, mime):
            extra={}
            if code==200 and mime=='audio/wav':
                code,body,extra=audio_byte_range(body,self.headers.get('Range'))
            self.send_response(code); self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store')
            for name,value in extra.items():self.send_header(name,value)
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            try:
                route = urlparse(self.path).path
                if route == '/':
                    if (args.page.resolve().parent.parent/'active_vowel_model.json').is_file():
                        self.send(200,(args.page.parent/'swar-words.html').read_bytes(),'text/html; charset=utf-8');return
                    current=args.page.parent/'corrected-pair.html'
                    plan=args.page.resolve().parent.parent/'data/teacher_indicf5_v14/model_pair.json'
                    self.send(200, current.read_bytes() if current.is_file() and plan.is_file() else args.page.read_bytes(), 'text/html; charset=utf-8')
                elif route == '/legacy':
                    self.send(200, args.page.read_bytes(), 'text/html; charset=utf-8')
                elif route == '/progress' and (args.page.parent / 'wordlab-progress.html').is_file():
                    if (args.page.resolve().parent.parent/'active_vowel_model.json').is_file():
                        self.send(200,(args.page.parent/'swar-words.html').read_bytes(),'text/html; charset=utf-8');return
                    self.send(200, (args.page.parent / 'wordlab-progress.html').read_bytes(), 'text/html; charset=utf-8')
                elif route == '/overnight':
                    self.send(200, (args.page.parent / 'overnight.html').read_bytes(), 'text/html; charset=utf-8')
                elif route == '/comparison':
                    self.send(200, (args.page.parent / 'corrected-pair.html').read_bytes(), 'text/html; charset=utf-8')
                elif route == '/swar':
                    self.send(200, (args.page.parent / 'swar-words.html').read_bytes(), 'text/html; charset=utf-8')
                elif route == '/api/swar':
                    root = args.page.resolve().parent.parent
                    model = vowel_directory(root)
                    def read_optional(path):
                        return json.loads(path.read_text()) if path.is_file() else None
                    actual = model / 'actual_swar'
                    cases = read_optional(actual / 'cases.json')
                    selection = cases if cases is not None else read_optional(root / 'data/swar_speech_v18/selection.json') or []
                    vowel_index = [{'id': r['id'], 'text': r['text'], 'voice': r['voice'],
                              'user_pick': r['user_pick'], 'scored': bool(r.get('analysis')),
                              'swar_flags': sum(w['decision'] == 'strong_vowel_change' for w in
                                  r.get('analysis', {}).get('swar', {}).get('words', []))} for r in selection]
                    review=read_optional(model/'review_report.json')
                    payload = {'model_report': review if review is not None else read_optional(model / 'report.json'),
                               'model_status': read_optional(model / 'status.json'),
                               'control_status': read_optional(root / 'reports/vowel_controls_v19/status.json'),
                               'actual_report': read_optional(actual / 'report.json') if cases is not None else None,
                               'cases': vowel_index}
                    self.send(200, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8')
                elif route.startswith('/api/swar-case/'):
                    root = args.page.resolve().parent.parent
                    identifier = route.removeprefix('/api/swar-case/')
                    model=vowel_directory(root)
                    path = model/'actual_swar/cases.json'
                    cases = json.loads(path.read_text()) if path.is_file() else []
                    found = [r for r in cases if r['id'] == identifier]
                    if len(found) != 1:
                        self.send(404, b'Unknown or unfinished Swar case', 'text/plain'); return
                    self.send(200, json.dumps(review_case_additions(found[0],model), ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8')
                elif route.startswith('/swar-audio/'):
                    parts = route.removeprefix('/swar-audio/').split('/')
                    if len(parts) != 2 or parts[1] not in {'real', 'swar', 'a450', 'prompt'}:
                        self.send(404, b'Unknown Swar audio', 'text/plain'); return
                    root = args.page.resolve().parent.parent / 'data/swar_speech_v18'
                    inventory = json.loads((root / 'inventory.json').read_text())
                    matches = [r for r in inventory['assets'] if r['id'] == parts[0] and r['kind'] == parts[1]]
                    if len(matches) != 1:
                        self.send(404, b'Unknown Swar audio', 'text/plain'); return
                    row = matches[0]; path = (root / row['path']).resolve(strict=True)
                    if not path.is_relative_to(root.resolve() / 'assets'):
                        raise ValueError('Swar audio outside verified asset inventory')
                    body = path.read_bytes()
                    if len(body) != row['bytes'] or hashlib.sha256(body).hexdigest() != row['sha256']:
                        raise ValueError('Swar source audio identity changed')
                    self.send(200, body, 'audio/wav')
                elif route == '/swar-model':
                    root = vowel_directory(args.page.resolve().parent.parent)
                    report = root / 'report.json'
                    if not report.is_file():
                        self.send(404, b'Model is not ready', 'text/plain'); return
                    body = (root / 'model.pt').read_bytes()
                    if hashlib.sha256(body).hexdigest() != json.loads(report.read_text())['model_sha256']:
                        raise ValueError('Selected vowel model hash changed')
                    self.send(200, body, 'application/octet-stream')
                elif route == '/swar-validation':
                    root=vowel_directory(args.page.resolve().parent.parent)
                    path=root/'review_report.json'
                    if not path.is_file():path=root/'report.json'
                    self.send(200,path.read_bytes(),'application/json; charset=utf-8')
                elif route == '/api/comparison':
                    root=args.page.resolve().parent.parent
                    plan=root/'data/teacher_indicf5_v14';report=root/'reports/corrected_pair_v14'
                    def read(p):
                        return json.loads(p.read_text()) if p.is_file() else None
                    def journal(p):
                        return [json.loads(line) for line in p.read_bytes().split(b'\n')[:-1]] if p.is_file() else []
                    pair=read(plan/'model_pair.json');sources={};grouped={}
                    content={r['clip_id']:r['content_quality'] for r in journal(report/'content_items.jsonl')}
                    for system,folder in [('pocket_teacher_250000_24l','teacher'),('indicf5_stock','indicf5')]:
                        rows=journal(plan/folder/'items.jsonl');runtime=read(plan/folder/'runtime.json') or {}
                        sources[system]={'status':read(plan/folder/'run.json'),'runtime':{k:runtime[k] for k in ('weights_sha256','loaded_layers','repo','revision','settings') if k in runtime}}
                        for r in rows:
                            if r['status']=='ok_unreviewed':
                                grouped.setdefault(r['id'],{'id':r['id'],'text':r['text'],'target_words':r['target_words'],'split':r['split'],'speaker_id':r['speaker_id'],'systems':{}})['systems'][system]={
                                    'clip_id':r['clip_id'],'seconds':r['seconds'],'reference_sha256':r['reference_sha256'],
                                    'content_quality':content.get(r['clip_id'])}
                    cases=[r for r in grouped.values() if len(r['systems'])==2][:24]
                    training=root/'reports/corrected_training_v16'
                    preview=read(training/'preview/report.json')
                    examples_group='preview_examples' if preview is not None else 'pilot_examples'
                    example_info=read(training/examples_group/'report.json') or {}
                    training_data={'status':read(training/'status.json'),'protocol':read(training/'protocol.json'),
                        'summary':read(training/'final_summary.json'),'finalizer_status':read(training/'finalizer_status.json'),
                        'extraction':read(root/'reports/structured_evidence_v15/status.json'),
                        'pilot':read(training/'pilot/report.json'),'preview':preview,'final':read(training/'final/report.json'),
                        'native_variation_status':read(root/'reports/native_variation_audit_v16/status.json'),
                        'native_variation_report':read(root/'reports/native_variation_audit_v16/report.json'),
                        'development_cases_model_sha256':example_info.get('model_sha256'),
                        'development_cases':(read(training/examples_group/'predictions.json') or [])[:8]}
                    payload={'observed_utc':datetime.now(timezone.utc).isoformat(),'model_pair':pair,'plan':read(plan/'inputs.json'),
                        'controller':read(report/'status.json'),'smoke':read(report/'smoke_report.json'),'report':read(report/'report.json'),'sources':sources,'cases':cases,'training':training_data}
                    self.send(200,json.dumps(payload,ensure_ascii=False,allow_nan=False).encode(),'application/json; charset=utf-8')
                elif route.startswith('/training-audio/'):
                    identifier=route.removeprefix('/training-audio/');root=args.page.resolve().parent.parent/'reports/structured_evidence_v15'
                    if len(identifier)!=24 or any(c not in '0123456789abcdef' for c in identifier):
                        self.send(404,b'Unknown training clip','text/plain');return
                    matches=[]
                    for line in (root/'physical_items.jsonl').read_bytes().split(b'\n')[:-1]:
                        row=json.loads(line)
                        if row['clip_id']==identifier and row['feature_status']=='ok_structured_features':matches.append((root,row))
                    recovered=root/'cpu_recovery/items.jsonl'
                    if recovered.is_file():
                        for line in recovered.read_bytes().split(b'\n')[:-1]:
                            receipt=json.loads(line)
                            if receipt['kind']=='physical' and receipt['status']=='ok_structured_features' and receipt['clip_id']==identifier:
                                matches.append((root/'cpu_recovery',receipt['row']))
                    if len(matches)!=1:
                        self.send(404,b'Unknown training clip','text/plain');return
                    pack,row=matches[0];p=(pack/row['audio']).resolve(strict=True)
                    if not p.is_relative_to(pack.resolve()/'controls'):raise ValueError('Training audio outside whitelist')
                    body=p.read_bytes()
                    if hashlib.sha256(body).hexdigest()!=row['sha256']:raise ValueError('Training audio identity changed')
                    self.send(200,body,'audio/wav')
                elif route.startswith('/comparison-audio/'):
                    identifier=route.removeprefix('/comparison-audio/');root=args.page.resolve().parent.parent/'data/teacher_indicf5_v14'
                    if len(identifier)!=24 or any(c not in '0123456789abcdef' for c in identifier):
                        self.send(404,b'Unknown corrected clip','text/plain');return
                    matches=[]
                    for folder in ('teacher','indicf5'):
                        pack=root/folder;p=pack/'items.jsonl'
                        if not p.is_file():continue
                        for line in p.read_bytes().split(b'\n')[:-1]:
                            row=json.loads(line)
                            if row['clip_id']==identifier and row['status']=='ok_unreviewed':matches.append((pack,row))
                    if len(matches)!=1:
                        self.send(404,b'Unknown corrected clip','text/plain');return
                    pack,row=matches[0];p=(pack/row['audio']).resolve(strict=True)
                    if not p.is_relative_to(pack.resolve()/'audio'):raise ValueError('Corrected audio path outside whitelist')
                    body=p.read_bytes()
                    if hashlib.sha256(body).hexdigest()!=row['sha256']:raise ValueError('Corrected audio identity changed')
                    self.send(200,body,'audio/wav')
                elif route == '/api/overnight':
                    root = args.page.resolve().parent.parent
                    def read(name):
                        p = root / name
                        return json.loads(p.read_text()) if p.is_file() else None
                    model = 'reports/paired_detector_v11' if (root / 'reports/paired_detector_v11/report.json').is_file() else 'reports/paired_detector_pilot_v11'
                    predictions = read(model + '/predictions.json') or []
                    cases = [r for r in predictions if r['condition'] in {'native_control', 'full_hindi_source_audit'}][:80]
                    payload = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'corpus': read('data/pocket_bulk_plan_v11/report.json'),
                        'generation': [read(f'data/pocket_bulk_v11/worker_{i}/run.json') for i in range(4)],
                        'asr': [read(f'reports/bulk_asr_v11_{i}/status.json') for i in range(4)],
                        'features': read('reports/bulk_words_v11/status.json'), 'model': read(model + '/report.json'),
                        'operating_points': read('reports/paired_operating_points_v11/report.json'),
                        'g2p_audit': read('reports/g2p_source_audit_v11/report.json'),
                        'physical_audit': read('reports/physical_word_audit_v11/report.json'),
                        'physical_training': read('reports/physical_detector_v12/status.json'),
                        'physical_model': read('reports/physical_detector_v12/report.json'),
                        'physical_content': read('reports/physical_content_v12/report.json'),
                        'physical_cases': (read('reports/physical_detector_v12/predictions.json') or [])[:12],
                        'cases': cases}
                    self.send(200, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8')
                elif route.startswith('/physical-audio/'):
                    identifier = route.removeprefix('/physical-audio/')
                    root = args.page.resolve().parent.parent / 'reports/physical_detector_v12'
                    if not identifier.replace('_', '').isalnum() or not identifier.startswith('night11_'):
                        self.send(404, b'Unknown control', 'text/plain'); return
                    records = json.loads((root / 'predictions.json').read_text())
                    matches = [r for r in records if Path(r['audio']).stem == identifier]
                    if len(matches) != 1:
                        self.send(404, b'Unknown control', 'text/plain'); return
                    row = matches[0]; p = (root / row['audio']).resolve(strict=True)
                    if not p.is_relative_to(root.resolve() / 'controls'):
                        raise ValueError('Physical control path outside whitelist')
                    body = p.read_bytes()
                    if hashlib.sha256(body).hexdigest() != row['sha256']:
                        raise ValueError('Physical control checksum changed')
                    self.send(200, body, 'audio/wav')
                elif route.startswith('/overnight-audio/'):
                    identifier = route.removeprefix('/overnight-audio/')
                    root = args.page.resolve().parent.parent
                    if not identifier.isalnum() or len(identifier) != 24:
                        self.send(404, b'Unknown clip', 'text/plain'); return
                    matches = []
                    for line in (root / 'reports/bulk_words_v11/items.jsonl').read_text().splitlines():
                        record = json.loads(line)
                        if record['clip_id'] == identifier:
                            matches.append(record['row'])
                    if len(matches) != 1:
                        self.send(404, b'Unknown clip', 'text/plain'); return
                    row = matches[0]; pack = Path(row['source_pack']).resolve(strict=True)
                    p = (pack / row['audio']).resolve(strict=True)
                    if not pack.is_relative_to(root / 'data/pocket_bulk_v11') or not p.is_relative_to(pack):
                        raise ValueError('Unregistered overnight audio path')
                    body = p.read_bytes()
                    if hashlib.sha256(body).hexdigest() != row['sha256']:
                        raise ValueError('Overnight audio identity changed')
                    self.send(200, body, 'audio/wav')
                elif route == '/api/index':
                    self.send(200, json.dumps(index, ensure_ascii=False).encode(), 'application/json; charset=utf-8')
                elif route == '/api/current-run':
                    root = args.page.resolve().parent.parent

                    def optional(name):
                        p = root / name
                        return json.loads(p.read_text()) if p.is_file() else None

                    def extraction(name, planned, success_status='ok_unreviewed_word_evidence'):
                        p = root / name
                        body = p.read_text() if p.is_file() else ''
                        lines = body.splitlines()
                        if body and not body.endswith('\n'):
                            lines = lines[:-1]
                        rows = [json.loads(line) for line in lines]
                        return {'processed': len(rows), 'planned': planned,
                                'failed': sum(row['status'] != success_status for row in rows)}

                    status = {'observed_utc': datetime.now(timezone.utc).isoformat(),
                              'generation': optional('data/pocket_wordmatrix_v4/pocket/run.json'),
                              'pipeline': optional('reports/pocket_pipeline_v4/status.json'),
                              'native': optional('reports/pocket_native_validation_v4/status.json'),
                              'word_evidence': extraction('reports/wordlab_pocket_matrix_v4/items.jsonl', 1024),
                              'native_evidence': extraction('reports/wordlab_pocket_native_v4/items.jsonl', 64),
                              'model_report': optional('reports/pocket_crossed_models_v4/report.json') if (root / 'reports/pocket_crossed_models_v4/README.md').is_file() else None,
                              'native_report': optional('reports/pocket_native_validation_v4/report.json')}
                    status.update(human_metadata_status=optional('data/human_regions_v5/status.json'),
                                  human_metadata_audit=optional('data/human_regions_v5/audit.json'),
                                  human_bank_status=optional('reports/human_g2p_selection_v5/status.json'),
                                  human_bank_report=optional('data/human_g2p_bank_v5/report.json') if (root / 'data/human_g2p_bank_v5/README.md').is_file() else None,
                                  human_evidence_status=optional('reports/human_g2p_evidence_v5/job.json'),
                                  human_evidence=extraction('reports/human_g2p_evidence_v5/items.jsonl', 232),
                                  human_evidence_report=optional('reports/human_g2p_summary_v5/report.json') if (root / 'reports/human_g2p_summary_v5/README.md').is_file() else None,
                                  original_audio_pilot=optional('data/human_original_audio_pilot_v6/report.json'),
                                  original_audio_status=optional('data/human_original_audio_v6/status.json'),
                                  original_audio_report=optional('data/human_original_audio_v6/report.json'),
                                  original_audio_failure=optional('data/human_original_audio_v6/failure.json'),
                                  boundary_audit_status=optional('reports/human_boundary_audit_v6/status.json'),
                                  boundary_audit_report=optional('reports/human_boundary_audit_v6/result/report.json') if (root / 'reports/human_boundary_audit_v6/result/README.md').is_file() else None,
                                  intervention_render_report=optional('data/schwa_interventions_v7/render/result/report.json') if (root / 'data/schwa_interventions_v7/render/result/README.md').is_file() else None,
                                  intervention_feature_status=optional('reports/intervention_features_v7/status.json'),
                                  intervention_head_report=optional('reports/intervention_heads_v7/evaluation/report.json') if (root / 'reports/intervention_heads_v7/evaluation/README.md').is_file() else None,
                                  real_word_transfer_status=optional('reports/real_word_transfer_v8/status.json'),
                                  real_word_transfer_report=optional('reports/real_word_transfer_v8/result/report.json') if (root / 'reports/real_word_transfer_v8/result/README.md').is_file() else None,
                                  real_word_transfer_summary=optional('reports/real_word_transfer_summary_v8/report.json') if (root / 'reports/real_word_transfer_summary_v8/README.md').is_file() else None,
                                  human_matched_plan=optional('data/human_matched_v9/report.json') if (root / 'data/human_matched_v9/README.md').is_file() else None,
                                  human_matched_f5_run=optional('data/human_matched_f5_v9/indicf5/run.json'),
                                  human_matched_pocket_run=optional('data/human_matched_pocket_v9/pocket/run.json'),
                                  human_matched_continuation=optional('reports/human_matched_v9/continuation.json'))
                    if status['human_matched_plan'] is not None:
                        planned = status['human_matched_plan']['jobs_per_tts']
                        status.update(human_matched_f5=extraction('data/human_matched_f5_v9/indicf5/items.jsonl', planned, 'ok_unreviewed'),
                                      human_matched_pocket=extraction('data/human_matched_pocket_v9/pocket/items.jsonl', planned, 'ok_unreviewed'),
                                      human_matched_f5_evidence=extraction('reports/human_matched_f5_evidence_v9/items.jsonl', planned),
                                      human_matched_pocket_evidence=extraction('reports/human_matched_pocket_evidence_v9/items.jsonl', planned))
                    self.send(200, json.dumps(status, ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8')
                elif route.startswith('/api/sentence/') and route.removeprefix('/api/sentence/') in allowed:
                    identifier = route.removeprefix('/api/sentence/')
                    self.send(200, (args.data / (identifier + '.json')).read_bytes(), 'application/json; charset=utf-8')
                elif route.startswith('/audio/') and route.removeprefix('/audio/') in registry:
                    path, sha = registry[route.removeprefix('/audio/')]
                    body = path.read_bytes()
                    if hashlib.sha256(body).hexdigest() != sha:
                        raise ValueError('Lab audio identity mismatch')
                    self.send(200, body, 'audio/wav')
                elif route == '/favicon.ico':
                    self.send(204, b'', 'image/x-icon')
                else:
                    self.send(404, b'{"error":"resource not found"}', 'application/json')
            except (BrokenPipeError, ConnectionResetError):
                LOG.info('Lab client disconnected')
            except Exception:
                LOG.exception('Word lab read failed')
                self.send(500, b'{"error":"lab read failed; inspect logged traceback"}', 'application/json')

    server = ReadOnlyServer(('127.0.0.1', args.port), Handler)
    LOG.info('Read-only word lab listening on loopback:%s', args.port)
    server.serve_forever()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
